"""
app.py
------
Unified Flask API serving all four forecasting models:
  - Transformer  (most accurate, 756-day window)
  - LSTM         (sequence model, 756-day window)
  - RNN          (simple recurrent, 252-day window)
  - Random Forest (tree-based baseline, 252-day window)

Routes:
  GET /api/predict/<ticker>?model=transformer|lstm|rnn|rf
  GET /api/predict/all/<ticker>          — all 4 models in one call
  GET /api/sentiment/<ticker>            — VADER sentiment on yfinance news
  GET /api/health                        — sanity check

Run locally:
  python app.py
  # open http://localhost:5000
"""

import os
import sys
import math
import joblib
import io
import base64

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.join(BASE_DIR, "transformer_final")
LSTM_DIR = os.path.join(BASE_DIR, "lstm_final_project")
RNN_DIR = os.path.join(BASE_DIR, "rnn_final")
RF_DIR = os.path.join(BASE_DIR, "rf_final")

# ---------------------------------------------------------------------------
# Shared feature engineering
# ---------------------------------------------------------------------------
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
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
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


def build_chart_b64(df: pd.DataFrame, ticker: str) -> str:
    """Return a base64-encoded 1-year price + SMA chart."""
    df_plot = df.tail(252)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_plot.index, df_plot["Close"], label="Close", linewidth=1.5)
    ax.plot(
        df_plot.index, df_plot["SMA_50"], label="SMA 50", linewidth=1, linestyle="--"
    )
    ax.plot(
        df_plot.index, df_plot["SMA_200"], label="SMA 200", linewidth=1, linestyle="--"
    )
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
    d = df_tech.tail(252)
    return {
        "dates": [dt.strftime("%Y-%m-%d") for dt in d.index],
        "close": d["Close"].tolist(),
        "sma_50": d["SMA_50"].tolist(),
        "sma_200": d["SMA_200"].tolist(),
    }


# ---------------------------------------------------------------------------
# Model definitions (must match training code exactly)
# ---------------------------------------------------------------------------
HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "6m": 126, "1y": 252}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ---------------------------------------------------------------------------
# Load all model artifacts at startup
# ---------------------------------------------------------------------------
print("Loading model artifacts...")

# Transformer
_tf_meta = joblib.load(os.path.join(TRANSFORMER_DIR, "transformer_meta.pkl"))
_tf_window = _tf_meta["window"]  # 756
_tf_sc_feat = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_feat.pkl"))
_tf_sc_ret = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_ret.pkl"))
_tf_model = TimeSeriesTransformer().to(DEVICE)
_tf_model.load_state_dict(
    torch.load(
        os.path.join(TRANSFORMER_DIR, "transformer_multi_horizon.pth"),
        map_location=DEVICE,
    )
)
_tf_model.eval()

# LSTM
_lstm_meta = joblib.load(os.path.join(LSTM_DIR, "lstm_meta.pkl"))
_lstm_window = _lstm_meta["window"]  # 756
_lstm_sc_feat = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_feat.pkl"))
_lstm_sc_targ = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_targ.pkl"))
_lstm_model = LSTMForecast().to(DEVICE)
_lstm_model.load_state_dict(
    torch.load(os.path.join(LSTM_DIR, "lstm_multi_horizon.pth"), map_location=DEVICE)
)
_lstm_model.eval()

# RNN
_rnn_meta = joblib.load(os.path.join(RNN_DIR, "rnn_meta.pkl"))
_rnn_window = _rnn_meta["window"]  # 252
_rnn_sc_feat = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_feat.pkl"))
_rnn_sc_targ = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_targ.pkl"))
_rnn_model = RNNForecast().to(DEVICE)
_rnn_model.load_state_dict(
    torch.load(os.path.join(RNN_DIR, "rnn_multi_horizon.pth"), map_location=DEVICE)
)
_rnn_model.eval()

# Random Forest
_rf_model = joblib.load(os.path.join(RF_DIR, "rf_multi_horizon.pkl"))
_rf_features = joblib.load(os.path.join(RF_DIR, "feature_list_multi.pkl"))
_rf_window = 252

print("All models loaded.")


# ---------------------------------------------------------------------------
# Per-model inference helpers
# ---------------------------------------------------------------------------
def _predict_transformer(df_tech: pd.DataFrame, close_price: float) -> dict:
    if len(df_tech) < _tf_window:
        raise ValueError(f"Need at least {_tf_window} rows of data.")
    window = df_tech[FEATS].tail(_tf_window).values
    window_s = _tf_sc_feat.transform(window)
    x = torch.tensor(window_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        ret_s = _tf_model(x).cpu().numpy()
    ret = _tf_sc_ret.inverse_transform(ret_s).flatten()
    return {k: round(close_price * (1 + ret[i]), 2) for i, k in enumerate(HORIZONS)}


def _predict_lstm(df_tech: pd.DataFrame, close_price: float) -> dict:
    if len(df_tech) < _lstm_window:
        raise ValueError(f"Need at least {_lstm_window} rows of data.")
    window = df_tech[FEATS].tail(_lstm_window).values.astype(np.float32)
    window_s = _lstm_sc_feat.transform(window.reshape(-1, window.shape[-1])).reshape(
        1, *window.shape
    )
    x = torch.tensor(window_s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_s = _lstm_model(x).cpu().numpy()
    pred = _lstm_sc_targ.inverse_transform(pred_s).flatten()
    return {k: round(float(pred[i]), 2) for i, k in enumerate(HORIZONS)}


def _predict_rnn(df_tech: pd.DataFrame, close_price: float) -> dict:
    if len(df_tech) < _rnn_window:
        raise ValueError(f"Need at least {_rnn_window} rows of data.")
    window = df_tech[FEATS].tail(_rnn_window).values.astype(np.float32)
    window_s = _rnn_sc_feat.transform(window.reshape(-1, window.shape[-1])).reshape(
        1, *window.shape
    )
    x = torch.tensor(window_s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_s = _rnn_model(x).cpu().numpy()
    pred = _rnn_sc_targ.inverse_transform(pred_s).flatten()
    return {k: round(float(pred[i]), 2) for i, k in enumerate(HORIZONS)}


def _predict_rf(df_ohlcv: pd.DataFrame) -> dict:
    if len(df_ohlcv) < _rf_window:
        raise ValueError(f"Need at least {_rf_window} rows of data.")
    window_flat = (
        df_ohlcv[["Open", "High", "Low", "Close", "Volume"]]
        .tail(_rf_window)
        .values.flatten()
    )
    X_df = pd.DataFrame([window_flat], columns=_rf_features)
    pred = _rf_model.predict(X_df)[0]
    return {k: round(float(pred[i]), 2) for i, k in enumerate(HORIZONS)}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models": ["transformer", "lstm", "rnn", "rf"]})


@app.route("/api/predict/<ticker>")
def predict_single(ticker):
    """
    GET /api/predict/<ticker>?model=transformer|lstm|rnn|rf
    Returns predictions from one model for all 5 horizons.
    """
    model_name = request.args.get("model", "transformer").lower()
    try:
        df_ohlcv = fetch_ohlcv(ticker)
        df_tech = compute_technicals(df_ohlcv)
        close = float(df_tech["Close"].iloc[-1])
        chart = build_chart_b64(df_tech, ticker)
        pd_data = price_data_dict(df_tech)

        if model_name == "transformer":
            preds = _predict_transformer(df_tech, close)
        elif model_name == "lstm":
            preds = _predict_lstm(df_tech, close)
        elif model_name == "rnn":
            preds = _predict_rnn(df_tech, close)
        elif model_name == "rf":
            preds = _predict_rf(df_ohlcv)
        else:
            return jsonify(
                {
                    "error": f"Unknown model '{model_name}'. Use: transformer, lstm, rnn, rf"
                }
            ), 400

        return jsonify(
            {
                "ticker": ticker.upper(),
                "model": model_name,
                "last_close_price": round(close, 2),
                "predicted_prices": preds,
                "chart_base64": chart,
                "price_data": pd_data,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/all/<ticker>")
def predict_all(ticker):
    """
    GET /api/predict/all/<ticker>
    Returns predictions from all 4 models side by side.
    Useful for the frontend comparison dashboard tab.
    """
    try:
        df_ohlcv = fetch_ohlcv(ticker)
        df_tech = compute_technicals(df_ohlcv)
        close = float(df_tech["Close"].iloc[-1])
        chart = build_chart_b64(df_tech, ticker)
        pd_data = price_data_dict(df_tech)

        results = {}
        errors = {}

        for name, fn in [
            ("transformer", lambda: _predict_transformer(df_tech, close)),
            ("lstm", lambda: _predict_lstm(df_tech, close)),
            ("rnn", lambda: _predict_rnn(df_tech, close)),
            ("rf", lambda: _predict_rf(df_ohlcv)),
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
                "chart_base64": chart,
                "price_data": pd_data,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment/<ticker>")
def sentiment(ticker):
    """
    GET /api/sentiment/<ticker>
    Fetches recent news from yfinance and runs VADER sentiment analysis.
    Returns per-article scores and an aggregate score — feeds SentimentAnalysis.js.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()

        sym = ticker.upper().replace(".", "-")
        ticker_obj = yf.Ticker(sym)
        news = ticker_obj.news or []

        articles = []
        scores = []
        for item in news[:20]:
            title = item.get("title", "")
            summary = item.get("summary", "") or item.get("description", "")
            link = item.get("link", "")
            pub_ts = item.get("providerPublishTime", 0)
            date = pd.Timestamp(pub_ts, unit="s").strftime("%Y-%m-%d") if pub_ts else ""

            text = f"{title}. {summary}"
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)
            articles.append(
                {
                    "title": title,
                    "summary": summary[:200] if summary else "",
                    "link": link,
                    "date": date,
                    "sentiment_score": round(score, 3),
                }
            )

        avg_score = round(float(np.mean(scores)), 3) if scores else 0.0
        if avg_score > 0.05:
            sentiment_label = "positive"
        elif avg_score < -0.05:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return jsonify(
            {
                "ticker": ticker.upper(),
                "score": avg_score,
                "sentiment": sentiment_label,
                "articles_analyzed": len(articles),
                "articles": articles,
            }
        )

    except ImportError:
        return jsonify(
            {"error": "vaderSentiment not installed. Run: pip install vaderSentiment"}
        ), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
