"""
app.py
Unified Flask backend serving all four forecasting models.

Routes:
    GET /api/predict/<ticker>?model=transformer|lstm|rnn|rf
    GET /api/predict/all/<ticker>
    GET /api/sentiment/<ticker>
    GET /api/fundamentals/<ticker>
    GET /api/health

Monte Carlo Dropout is used for the Transformer and LSTM to produce
confidence intervals instead of single-point estimates. The model is run
N_MC times with dropout active during inference. The spread across those
runs gives p10 and p90 bounds.

Run locally:
    python app.py
"""

import os
import math
import io
import base64
import joblib

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch import nn
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.join(BASE_DIR, "transformer_final")
LSTM_DIR = os.path.join(BASE_DIR, "lstm_final_project")
RNN_DIR = os.path.join(BASE_DIR, "rnn_final")
RF_DIR = os.path.join(BASE_DIR, "rf_final")

HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "6m": 126, "1y": 252}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of forward passes for Monte Carlo Dropout uncertainty estimation.
# Higher N = more stable intervals but slower response. 50 is a good default.
N_MC = 50

FEATS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_10",
    "SMA_50",
    "SMA_200",
    "RSI_14",
    "MOM_1",
    "ROC_14",
    "MACD",
]


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["MOM_1"] = df["Close"].diff(1)
    df["ROC_14"] = df["Close"].pct_change(14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()


def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    sym = ticker.upper().replace(".", "-")
    df = yf.download(sym, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def build_chart_b64(df_tech: pd.DataFrame, ticker: str) -> str:
    """Renders a 1-year price chart with SMA lines and returns it as base64 PNG."""
    d = df_tech.tail(252)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d.index, d["Close"], label="Close", linewidth=1.5)
    ax.plot(d.index, d["SMA_50"], label="SMA 50", linewidth=1, linestyle="--")
    ax.plot(d.index, d["SMA_200"], label="SMA 200", linewidth=1, linestyle="--")
    ax.set_title(f"{ticker.upper()} — Last 1 Year")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    buf.close()
    return b64


def price_data_dict(df_tech: pd.DataFrame) -> dict:
    """Returns the last year of OHLCV and SMA data in a format the frontend chart expects."""
    d = df_tech.tail(252)
    return {
        "dates": [dt.strftime("%Y-%m-%d") for dt in d.index],
        "close": d["Close"].tolist(),
        "sma_50": d["SMA_50"].tolist(),
        "sma_200": d["SMA_200"].tolist(),
    }


def generate_recommendation(close_price: float, preds: dict, tech: dict) -> dict:
    """
    Scores the stock on a simple scale using predicted price changes and
    moving average signals, then maps the score to BUY / HOLD / SELL.

    Score range:
        >= 3   BUY  (High confidence)
         1-2   BUY  (Medium confidence)
        -1 to 0 HOLD
        -3 to -2 SELL (Medium confidence)
        <= -4  SELL (High confidence)
    """
    changes = {
        h: round(((preds[h] - close_price) / close_price) * 100, 2) for h in preds
    }

    sma_50 = tech.get("sma_50", [])
    sma_200 = tech.get("sma_200", [])

    if sma_50 and sma_200:
        cur_sma50 = float(sma_50[-1])
        cur_sma200 = float(sma_200[-1])
        golden_cross = cur_sma50 > cur_sma200
        above_sma50 = float(close_price) > cur_sma50
        above_sma200 = float(close_price) > cur_sma200
    else:
        golden_cross = above_sma50 = above_sma200 = False

    score = 0
    reasons = []

    short = changes.get("1d", 0)
    long = changes.get("1y", 0)

    if short > 2:
        score += 2
        reasons.append(f"Short-term forecast up {short:.1f}%")
    elif short > 0:
        score += 1
        reasons.append(f"Short-term forecast slightly positive ({short:.1f}%)")
    elif short < -2:
        score -= 2
        reasons.append(f"Short-term forecast down {short:.1f}%")
    elif short < 0:
        score -= 1
        reasons.append(f"Short-term forecast slightly negative ({short:.1f}%)")

    if long > 10:
        score += 3
        reasons.append(f"Strong long-term upside ({long:.1f}%)")
    elif long > 5:
        score += 2
        reasons.append(f"Good long-term potential ({long:.1f}%)")
    elif long < -10:
        score -= 3
        reasons.append(f"Significant long-term downside ({long:.1f}%)")
    elif long < -5:
        score -= 2
        reasons.append(f"Weak long-term outlook ({long:.1f}%)")

    if golden_cross:
        score += 1
        reasons.append("SMA 50 is above SMA 200 (bullish crossover)")

    if above_sma50 and above_sma200:
        score += 1
        reasons.append("Price is above both moving averages")
    elif not above_sma50 and not above_sma200:
        score -= 1
        reasons.append("Price is below both moving averages")

    if score >= 3:
        rec, conf = "BUY", "High"
    elif score >= 1:
        rec, conf = "BUY", "Medium"
    elif score >= -1:
        rec, conf = "HOLD", "Medium"
    elif score >= -3:
        rec, conf = "SELL", "Medium"
    else:
        rec, conf = "SELL", "High"

    return {
        "recommendation": rec,
        "confidence": conf,
        "score": score,
        "reasons": reasons,
        "price_changes": changes,
    }


def get_fundamentals(ticker: str) -> dict:
    """Pulls key company fundamentals from yfinance and formats large numbers."""
    try:
        info = yf.Ticker(ticker.upper().replace(".", "-")).info

        def fmt(v):
            if not isinstance(v, (int, float)) or v == 0:
                return v
            if v >= 1e9:
                return f"${v / 1e9:.2f}B"
            if v >= 1e6:
                return f"${v / 1e6:.2f}M"
            return round(v, 2)

        return {
            "company_info": {
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": fmt(info.get("marketCap", 0)),
                "website": info.get("website", "N/A"),
                "country": info.get("country", "N/A"),
            },
            "valuation": {
                "pe_ratio": round(info.get("trailingPE", 0) or 0, 2),
                "forward_pe": round(info.get("forwardPE", 0) or 0, 2),
                "price_to_book": round(info.get("priceToBook", 0) or 0, 2),
                "price_to_sales": round(
                    info.get("priceToSalesTrailing12Months", 0) or 0, 2
                ),
            },
            "financials": {
                "revenue": fmt(info.get("totalRevenue", 0)),
                "net_income": fmt(info.get("netIncomeToCommon", 0)),
                "total_cash": fmt(info.get("totalCash", 0)),
                "total_debt": fmt(info.get("totalDebt", 0)),
                "debt_equity": round(info.get("debtToEquity", 0) or 0, 2),
                "current_ratio": round(info.get("currentRatio", 0) or 0, 2),
            },
            "growth": {
                "revenue_growth": round((info.get("revenueGrowth", 0) or 0) * 100, 2),
                "earnings_growth": round((info.get("earningsGrowth", 0) or 0) * 100, 2),
                "profit_margin": round((info.get("profitMargins", 0) or 0) * 100, 2),
            },
            "trading": {
                "beta": round(info.get("beta", 0) or 0, 2),
                "52w_change_pct": round((info.get("52WeekChange", 0) or 0) * 100, 2),
                "dividend_yield_pct": round(
                    (info.get("dividendYield", 0) or 0) * 100, 2
                ),
            },
        }
    except Exception as e:
        return {"error": str(e)}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=756):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feat_size=12,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.regressor = nn.Linear(d_model, len(HORIZONS))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :])


class LSTMForecast(nn.Module):
    def __init__(
        self, input_size=12, hidden_size=128, num_layers=2, out_size=5, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class RNNForecast(nn.Module):
    def __init__(
        self, input_size=12, hidden_size=128, num_layers=2, out_size=5, dropout=0.2
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


print("Loading model artifacts...")

tf_meta = joblib.load(os.path.join(TRANSFORMER_DIR, "transformer_meta.pkl"))
tf_window = tf_meta["window"]
tf_sc_feat = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_feat.pkl"))
tf_sc_ret = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_ret.pkl"))
tf_model = TimeSeriesTransformer().to(DEVICE)
tf_model.load_state_dict(
    torch.load(
        os.path.join(TRANSFORMER_DIR, "transformer_multi_horizon.pth"),
        map_location=DEVICE,
    )
)
tf_model.eval()

lstm_meta = joblib.load(os.path.join(LSTM_DIR, "lstm_meta.pkl"))
lstm_window = lstm_meta["window"]
lstm_sc_feat = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_feat.pkl"))
lstm_sc_targ = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_targ.pkl"))
lstm_model = LSTMForecast().to(DEVICE)
lstm_model.load_state_dict(
    torch.load(
        os.path.join(LSTM_DIR, "lstm_multi_horizon.pth"),
        map_location=DEVICE,
    )
)
lstm_model.eval()

rnn_meta = joblib.load(os.path.join(RNN_DIR, "rnn_meta.pkl"))
rnn_window = rnn_meta["window"]
rnn_sc_feat = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_feat.pkl"))
rnn_sc_targ = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_targ.pkl"))
rnn_model = RNNForecast().to(DEVICE)
rnn_model.load_state_dict(
    torch.load(
        os.path.join(RNN_DIR, "rnn_multi_horizon.pth"),
        map_location=DEVICE,
    )
)
rnn_model.eval()

rf_model = joblib.load(os.path.join(RF_DIR, "rf_multi_horizon.pkl"))
rf_features = joblib.load(os.path.join(RF_DIR, "feature_list_multi.pkl"))
rf_window = 252

print("All models loaded.")


def _enable_dropout(model: nn.Module):
    """Sets all Dropout layers to train mode so they stay active during inference.
    This is what makes Monte Carlo Dropout work. The rest of the model stays in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _mc_predict(model, x_tensor, sc_ret, n=N_MC, is_return_model=True):
    """
    Runs N_MC forward passes with dropout active to get a distribution of predictions.
    Returns p10, p50, and p90 prices for each horizon.

    For models trained on returns (Transformer), we inverse-transform with sc_ret
    and then multiply by close price to get absolute prices.
    For models trained on absolute prices (LSTM), sc_ret is the target scaler.
    """
    _enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(n):
            out = model(x_tensor).cpu().numpy()
            samples.append(out)
    model.eval()

    samples = np.stack(samples, axis=0)  # shape (N_MC, 1, n_horizons)

    if is_return_model:
        samples_unscaled = np.stack(
            [sc_ret.inverse_transform(s) for s in samples], axis=0
        )
    else:
        samples_unscaled = np.stack(
            [sc_ret.inverse_transform(s) for s in samples], axis=0
        )

    p10 = np.percentile(samples_unscaled, 10, axis=0).flatten()
    p50 = np.percentile(samples_unscaled, 50, axis=0).flatten()
    p90 = np.percentile(samples_unscaled, 90, axis=0).flatten()

    return p10, p50, p90


def predict_transformer(df_tech: pd.DataFrame, close: float) -> dict:
    if len(df_tech) < tf_window:
        raise ValueError(f"Need at least {tf_window} rows.")

    window = df_tech[FEATS].tail(tf_window).values
    window_s = tf_sc_feat.transform(window)
    x = torch.tensor(window_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    p10_r, p50_r, p90_r = _mc_predict(tf_model, x, tf_sc_ret, is_return_model=True)

    # Convert returns to prices
    p50 = {k: round(close * (1 + p50_r[i]), 2) for i, k in enumerate(HORIZONS)}
    p10 = {k: round(close * (1 + p10_r[i]), 2) for i, k in enumerate(HORIZONS)}
    p90 = {k: round(close * (1 + p90_r[i]), 2) for i, k in enumerate(HORIZONS)}

    return {"p50": p50, "p10": p10, "p90": p90}


def predict_lstm(df_tech: pd.DataFrame, close: float) -> dict:
    if len(df_tech) < lstm_window:
        raise ValueError(f"Need at least {lstm_window} rows.")

    window = df_tech[FEATS].tail(lstm_window).values.astype(np.float32)
    window_s = lstm_sc_feat.transform(window.reshape(-1, window.shape[-1])).reshape(
        1, *window.shape
    )
    x = torch.tensor(window_s, dtype=torch.float32).to(DEVICE)

    p10_s, p50_s, p90_s = _mc_predict(
        lstm_model, x, lstm_sc_targ, is_return_model=False
    )

    p50 = {k: round(float(p50_s[i]), 2) for i, k in enumerate(HORIZONS)}
    p10 = {k: round(float(p10_s[i]), 2) for i, k in enumerate(HORIZONS)}
    p90 = {k: round(float(p90_s[i]), 2) for i, k in enumerate(HORIZONS)}

    return {"p50": p50, "p10": p10, "p90": p90}


def predict_rnn(df_tech: pd.DataFrame, close: float) -> dict:
    """RNN does not use MC Dropout (it was trained with a shorter window and no
    dropout layers in its recurrent stack). Returns a single point estimate."""
    if len(df_tech) < rnn_window:
        raise ValueError(f"Need at least {rnn_window} rows.")

    window = df_tech[FEATS].tail(rnn_window).values.astype(np.float32)
    window_s = rnn_sc_feat.transform(window.reshape(-1, window.shape[-1])).reshape(
        1, *window.shape
    )
    x = torch.tensor(window_s, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_s = rnn_model(x).cpu().numpy()

    pred = rnn_sc_targ.inverse_transform(pred_s).flatten()
    p50 = {k: round(float(pred[i]), 2) for i, k in enumerate(HORIZONS)}

    return {"p50": p50, "p10": p50, "p90": p50}


def predict_rf(df_ohlcv: pd.DataFrame) -> dict:
    """Random Forest does not have dropout. Returns a single point estimate."""
    if len(df_ohlcv) < rf_window:
        raise ValueError(f"Need at least {rf_window} rows.")

    flat = (
        df_ohlcv[["Open", "High", "Low", "Close", "Volume"]]
        .tail(rf_window)
        .values.flatten()
    )
    X_df = pd.DataFrame([flat], columns=rf_features)
    pred = rf_model.predict(X_df)[0]
    p50 = {k: round(float(pred[i]), 2) for i, k in enumerate(HORIZONS)}

    return {"p50": p50, "p10": p50, "p90": p50}


app = Flask(__name__)
CORS(app)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "models": ["transformer", "lstm", "rnn", "rf"],
            "device": str(DEVICE),
        }
    )


@app.route("/api/predict/<ticker>")
def predict_single(ticker):
    """
    GET /api/predict/<ticker>?model=transformer|lstm|rnn|rf

    Returns p50 (median), p10 (lower bound), p90 (upper bound) for all 5 horizons,
    along with the recommendation, chart data, and current price.
    """
    model_name = request.args.get("model", "transformer").lower()

    try:
        df_ohlcv = fetch_ohlcv(ticker)
        df_tech = compute_technicals(df_ohlcv)
        close = float(df_tech["Close"].iloc[-1])
        pd_data = price_data_dict(df_tech)
        chart = build_chart_b64(df_tech, ticker)

        if model_name == "transformer":
            result = predict_transformer(df_tech, close)
        elif model_name == "lstm":
            result = predict_lstm(df_tech, close)
        elif model_name == "rnn":
            result = predict_rnn(df_tech, close)
        elif model_name == "rf":
            result = predict_rf(df_ohlcv)
        else:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        rec = generate_recommendation(close, result["p50"], pd_data)

        return jsonify(
            {
                "ticker": ticker.upper(),
                "model": model_name,
                "last_close_price": round(close, 2),
                "current_price": round(close, 2),
                "predictions": result["p50"],
                "predicted_prices": result["p50"],
                "p10": result["p10"],
                "p90": result["p90"],
                "recommendation": rec,
                "price_data": pd_data,
                "technical_data": pd_data,
                "chart_base64": chart,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/all/<ticker>")
def predict_all(ticker):
    """
    GET /api/predict/all/<ticker>

    Runs all 4 models and returns their predictions side by side.
    The frontend uses this for the model comparison tab.
    """
    try:
        df_ohlcv = fetch_ohlcv(ticker)
        df_tech = compute_technicals(df_ohlcv)
        close = float(df_tech["Close"].iloc[-1])
        pd_data = price_data_dict(df_tech)
        chart = build_chart_b64(df_tech, ticker)

        results = {}
        errors = {}

        for name, fn in [
            ("transformer", lambda: predict_transformer(df_tech, close)),
            ("lstm", lambda: predict_lstm(df_tech, close)),
            ("rnn", lambda: predict_rnn(df_tech, close)),
            ("rf", lambda: predict_rf(df_ohlcv)),
        ]:
            try:
                results[name] = fn()
            except Exception as e:
                errors[name] = str(e)

        return jsonify(
            {
                "ticker": ticker.upper(),
                "last_close_price": round(close, 2),
                "predictions": results,
                "errors": errors,
                "price_data": pd_data,
                "chart_base64": chart,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment/<ticker>")
def sentiment(ticker):
    """
    GET /api/sentiment/<ticker>

    Fetches the last 20 news headlines from yfinance and runs VADER sentiment
    on each one. Returns per-article scores and an aggregate compound score.
    This feeds the SentimentAnalysis.js component in the frontend.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        ticker_obj = yf.Ticker(ticker.upper().replace(".", "-"))
        news = ticker_obj.news or []

        articles = []
        scores = []

        for item in news[:20]:
            title = item.get("title", "")
            summary = item.get("summary", "") or item.get("description", "")
            link = item.get("link", "")
            pub_ts = item.get("providerPublishTime", 0)
            date = pd.Timestamp(pub_ts, unit="s").strftime("%Y-%m-%d") if pub_ts else ""

            score = analyzer.polarity_scores(f"{title}. {summary}")["compound"]
            scores.append(score)
            articles.append(
                {
                    "title": title,
                    "summary": summary[:200],
                    "link": link,
                    "date": date,
                    "sentiment_score": round(score, 3),
                }
            )

        avg = round(float(np.mean(scores)), 3) if scores else 0.0
        label = "positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral"

        return jsonify(
            {
                "ticker": ticker.upper(),
                "score": avg,
                "sentiment": label,
                "articles_analyzed": len(articles),
                "articles": articles,
            }
        )

    except ImportError:
        return jsonify(
            {"error": "vaderSentiment not installed. pip install vaderSentiment"}
        ), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fundamentals/<ticker>")
def fundamentals(ticker):
    """
    GET /api/fundamentals/<ticker>

    Returns key company fundamentals from yfinance: valuation ratios,
    financials, growth rates, and trading info.
    """
    try:
        return jsonify(get_fundamentals(ticker))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_all_models():
    """Loads all model artifacts from disk into the module-level variables.
    Called once at startup and again by /api/reload after retraining."""
    global tf_model, tf_sc_feat, tf_sc_ret, tf_window
    global lstm_model, lstm_sc_feat, lstm_sc_targ, lstm_window
    global rnn_model, rnn_sc_feat, rnn_sc_targ, rnn_window
    global rf_model, rf_features, rf_window

    tf_meta = joblib.load(os.path.join(TRANSFORMER_DIR, "transformer_meta.pkl"))
    tf_window = tf_meta["window"]
    tf_sc_feat = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_feat.pkl"))
    tf_sc_ret = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_ret.pkl"))
    tf_model = TimeSeriesTransformer().to(DEVICE)
    tf_model.load_state_dict(
        torch.load(
            os.path.join(TRANSFORMER_DIR, "transformer_multi_horizon.pth"),
            map_location=DEVICE,
        )
    )
    tf_model.eval()

    lstm_meta = joblib.load(os.path.join(LSTM_DIR, "lstm_meta.pkl"))
    lstm_window = lstm_meta["window"]
    lstm_sc_feat = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_feat.pkl"))
    lstm_sc_targ = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_targ.pkl"))
    lstm_model = LSTMForecast().to(DEVICE)
    lstm_model.load_state_dict(
        torch.load(
            os.path.join(LSTM_DIR, "lstm_multi_horizon.pth"),
            map_location=DEVICE,
        )
    )
    lstm_model.eval()

    rnn_meta = joblib.load(os.path.join(RNN_DIR, "rnn_meta.pkl"))
    rnn_window = rnn_meta["window"]
    rnn_sc_feat = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_feat.pkl"))
    rnn_sc_targ = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_targ.pkl"))
    rnn_model = RNNForecast().to(DEVICE)
    rnn_model.load_state_dict(
        torch.load(
            os.path.join(RNN_DIR, "rnn_multi_horizon.pth"),
            map_location=DEVICE,
        )
    )
    rnn_model.eval()

    rf_model = joblib.load(os.path.join(RF_DIR, "rf_multi_horizon.pkl"))
    rf_features = joblib.load(os.path.join(RF_DIR, "feature_list_multi.pkl"))
    rf_window = 252

    print("Models reloaded from disk.")


@app.route("/api/reload", methods=["POST"])
def reload_models():
    """
    POST /api/reload

    Reloads all model weights from disk without restarting the server.
    Called automatically by the GitHub Actions retrain workflow after
    new checkpoints are committed and the server is redeployed.

    Protected by a secret token stored in the RELOAD_TOKEN environment
    variable so random people on the internet can't trigger it.
    """
    secret = os.environ.get("RELOAD_TOKEN", "")
    if secret and request.headers.get("X-Reload-Token") != secret:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        _load_all_models()
        return jsonify({"status": "ok", "message": "Models reloaded successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
