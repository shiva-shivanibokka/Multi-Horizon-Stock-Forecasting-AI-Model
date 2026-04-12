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
import re
import math
import io
import base64
import logging
import joblib
import time
from functools import wraps
from collections import defaultdict

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.join(BASE_DIR, "transformer_final")
LSTM_DIR = os.path.join(BASE_DIR, "lstm_final_project")
RNN_DIR = os.path.join(BASE_DIR, "rnn_final")
RF_DIR = os.path.join(BASE_DIR, "rf_final")

# All models now use 3 horizons — 1d and 1y removed from all models:
#   1d: dominated by news/microstructure — price history cannot reliably predict it
#   1y: too much macro uncertainty — prediction interval is too wide to be actionable
HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
HORIZONS_TRANSFORMER = (
    HORIZONS  # same — kept for backward compatibility in predict_transformer
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of forward passes for Monte Carlo Dropout uncertainty estimation.
# Higher N = more stable intervals but slower response. 50 is a good default.
N_MC = 50

# Full 36-feature set — must match build_dataset.py FEATS exactly
FEATS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_10",
    "SMA_50",
    "SMA_200",
    "MACD",
    "MACD_signal",
    "RSI_14",
    "MOM_5",
    "ROC_21",
    "Williams_R",
    "ATR_14",
    "BB_upper",
    "BB_lower",
    "BB_width",
    "OBV_norm",
    "Volume_SMA_20",
    "Volume_ratio",
    "body_size",
    "upper_shadow",
    "lower_shadow",
    "body_pct",
    "doji",
    "hammer",
    "shooting_star",
    "engulfing",
    "pct_from_52w_high",
    "pct_from_52w_low",
    "price_range_pct",
    "vix_close",
    "sp500_ret_21d",
    "sp500_ret_63d",
    "rel_strength_21d",
]
N_FEATS = len(FEATS)

# Sector map — used at inference to pass sector ID to PatchTST
SECTOR_MAP = {
    "Communication Services": 0,
    "Consumer Discretionary": 1,
    "Consumer Staples": 2,
    "Energy": 3,
    "Financials": 4,
    "Health Care": 5,
    "Industrials": 6,
    "Information Technology": 7,
    "Materials": 8,
    "Real Estate": 9,
    "Utilities": 10,
}

# Cached market data (VIX + S&P500) — refreshed once per server process
_market_cache: dict = {}


def get_market_data() -> tuple:
    """Returns (vix_series, sp500_ret_21d, sp500_ret_63d) cached for the session."""
    if "vix" not in _market_cache:
        try:
            vix = yf.download("^VIX", period="2y", interval="1d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            _market_cache["vix"] = vix["Close"].dropna()
        except Exception:
            _market_cache["vix"] = pd.Series(dtype=float)

        try:
            sp = yf.download("^GSPC", period="2y", interval="1d", progress=False)
            if isinstance(sp.columns, pd.MultiIndex):
                sp.columns = sp.columns.get_level_values(0)
            sp_close = sp["Close"].dropna()
            _market_cache["sp500_ret_21d"] = sp_close.pct_change(21)
            _market_cache["sp500_ret_63d"] = sp_close.pct_change(63)
        except Exception:
            _market_cache["sp500_ret_21d"] = pd.Series(dtype=float)
            _market_cache["sp500_ret_63d"] = pd.Series(dtype=float)

    return (
        _market_cache["vix"],
        _market_cache["sp500_ret_21d"],
        _market_cache["sp500_ret_63d"],
    )


def compute_features(df: pd.DataFrame, sector_id: int = 7) -> pd.DataFrame:
    """
    Computes the full 36-feature set for a single ticker at inference time.
    This must produce exactly the same features as build_dataset.compute_features().
    """
    vix, sp500_ret_21d, sp500_ret_63d = get_market_data()

    d = df.copy()

    # Trend
    d["SMA_10"] = d["Close"].rolling(10).mean()
    d["SMA_50"] = d["Close"].rolling(50).mean()
    d["SMA_200"] = d["Close"].rolling(200).mean()
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    # Momentum
    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    d["RSI_14"] = 100 - (100 / (1 + rs))
    d["MOM_5"] = d["Close"].diff(5)
    d["ROC_21"] = d["Close"].pct_change(21)
    high14 = d["High"].rolling(14).max()
    low14 = d["Low"].rolling(14).min()
    d["Williams_R"] = -100 * (high14 - d["Close"]) / (high14 - low14 + 1e-9)

    # Volatility
    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift(1)).abs()
    tr3 = (d["Low"] - d["Close"].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR_14"] = true_range.rolling(14).mean()
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["BB_upper"] = sma20 + 2 * std20
    d["BB_lower"] = sma20 - 2 * std20
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (sma20 + 1e-9)

    # Volume
    obv = (np.sign(d["Close"].diff()) * d["Volume"]).fillna(0).cumsum()
    obv_max = obv.rolling(252).max().replace(0, 1)
    d["OBV_norm"] = obv / obv_max.abs()
    d["Volume_SMA_20"] = d["Volume"].rolling(20).mean()
    d["Volume_ratio"] = d["Volume"] / (d["Volume_SMA_20"] + 1e-9)

    # Candlestick
    body = d["Close"] - d["Open"]
    full_range = (d["High"] - d["Low"]).replace(0, 1e-9)
    d["body_size"] = body.abs()
    d["upper_shadow"] = d["High"] - d[["Close", "Open"]].max(axis=1)
    d["lower_shadow"] = d[["Close", "Open"]].min(axis=1) - d["Low"]
    d["body_pct"] = body / full_range
    d["doji"] = (body.abs() < 0.1 * full_range).astype(float)
    prev_down = d["Close"].shift(1) < d["SMA_50"].shift(1)
    small_body = body.abs() < 0.3 * full_range
    long_lower = d["lower_shadow"] > 2 * body.abs()
    tiny_upper = d["upper_shadow"] < 0.1 * full_range
    d["hammer"] = (prev_down & small_body & long_lower & tiny_upper).astype(float)
    prev_up = d["Close"].shift(1) > d["SMA_50"].shift(1)
    long_upper = d["upper_shadow"] > 2 * body.abs()
    tiny_lower = d["lower_shadow"] < 0.1 * full_range
    d["shooting_star"] = (prev_up & small_body & long_upper & tiny_lower).astype(float)
    prev_body = d["Close"].shift(1) - d["Open"].shift(1)
    d["engulfing"] = (
        (body > 0)
        & (prev_body < 0)
        & (d["Close"] > d["Open"].shift(1))
        & (d["Open"] < d["Close"].shift(1))
    ).astype(float)

    # Price structure
    high_52w = d["High"].rolling(252).max()
    low_52w = d["Low"].rolling(252).min()
    d["pct_from_52w_high"] = (d["Close"] - high_52w) / (high_52w + 1e-9)
    d["pct_from_52w_low"] = (d["Close"] - low_52w) / (low_52w + 1e-9)
    d["price_range_pct"] = full_range / (d["Close"] + 1e-9)

    # Market features
    d["vix_close"] = vix.reindex(d.index).ffill().bfill()
    d["sp500_ret_21d"] = sp500_ret_21d.reindex(d.index).ffill().bfill()
    d["sp500_ret_63d"] = sp500_ret_63d.reindex(d.index).ffill().bfill()

    # Relative strength — at inference we don't have sector peers, use 0
    d["rel_strength_21d"] = 0.0

    return d[FEATS].dropna()


def fetch_ohlcv(ticker: str, period: str = "6y") -> pd.DataFrame:
    """Fetches 6y of OHLCV — extra year needed for 52-week high/low rolling windows."""
    sym = ticker.upper().replace(".", "-")
    df = yf.download(sym, period=period, interval="1d", progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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


class PatchEmbedding(nn.Module):
    def __init__(self, n_features=12, d_model=128, patch_len=16, stride=8):
        super().__init__()
        self.proj = nn.Conv1d(n_features, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class PatchTST(nn.Module):
    """
    PatchTST with separate output heads per horizon.
    Must match train_transformer.py exactly.
    """

    def __init__(
        self,
        n_features=36,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_ff=512,
        dropout=0.3,
        n_sectors=11,
        sector_dim=8,
        n_quantiles=3,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_horizons = 3  # 1w, 1m, 6m
        self.patch_embed = PatchEmbedding(n_features, d_model)
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, d_model)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        def _head(out_size, deep=False):
            if deep:
                return nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, out_size),
                )
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, out_size),
            )

        self.head_1w = _head(n_quantiles)
        self.head_1m = _head(n_quantiles)
        self.head_6m = _head(n_quantiles, deep=True)

    def encode(self, x, sector):
        patches = self.patch_embed(x)
        sec_emb = self.sector_proj(self.sector_embed(sector)).unsqueeze(1)
        patches = self.drop(patches + sec_emb)
        out = self.norm(self.encoder(patches))
        return out.mean(dim=1)

    def forward(self, x, sector):
        enc = self.encode(x, sector)
        q1w = self.head_1w(enc)
        q1m = self.head_1m(enc)
        q6m = self.head_6m(enc)
        return torch.stack([q1w, q1m, q6m], dim=1)  # (B, 3, 3)


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

# Each model is loaded inside a try/except so a missing checkpoint for one
# model does not prevent the other three from starting. If a model fails to
# load it is set to None and its predict route will return a clear error.

tf_model = tf_sc_feat = tf_sc_ret = None
tf_window = 756
try:
    tf_meta = joblib.load(os.path.join(TRANSFORMER_DIR, "transformer_meta.pkl"))
    tf_window = tf_meta.get("window", 756)
    tf_sc_feat = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_feat.pkl"))
    tf_sc_ret = joblib.load(os.path.join(TRANSFORMER_DIR, "scaler_ret.pkl"))
    tf_model = PatchTST().to(DEVICE)
    tf_model.load_state_dict(
        torch.load(
            os.path.join(TRANSFORMER_DIR, "transformer_multi_horizon.pth"),
            map_location=DEVICE,
            weights_only=False,
        )
    )
    tf_model.eval()
    print("PatchTST Transformer loaded.")
except FileNotFoundError as e:
    print(f"Transformer not loaded (missing file): {e}")
    print("Run: cd transformer_final && python train_transformer.py")

lstm_model = lstm_sc_feat = lstm_sc_targ = None
lstm_window = 756
try:
    lstm_meta = joblib.load(os.path.join(LSTM_DIR, "lstm_meta.pkl"))
    lstm_window = lstm_meta["window"]
    lstm_n_feat = lstm_meta.get("n_features", N_FEATS)
    lstm_n_out = len(lstm_meta.get("horizons", HORIZONS))
    lstm_sc_feat = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_feat.pkl"))
    lstm_sc_targ = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_targ.pkl"))
    lstm_model = LSTMForecast(input_size=lstm_n_feat, out_size=lstm_n_out).to(DEVICE)
    lstm_model.load_state_dict(
        torch.load(
            os.path.join(LSTM_DIR, "lstm_multi_horizon.pth"),
            map_location=DEVICE,
            weights_only=False,
        )
    )
    lstm_model.eval()
    print(f"LSTM loaded (input={lstm_n_feat} features, output={lstm_n_out} horizons).")
except FileNotFoundError as e:
    print(f"LSTM not loaded (missing file): {e}")

rnn_model = rnn_sc_feat = rnn_sc_targ = None
rnn_window = 252
try:
    rnn_meta = joblib.load(os.path.join(RNN_DIR, "rnn_meta.pkl"))
    rnn_window = rnn_meta["window"]
    rnn_n_feat = rnn_meta.get("n_features", N_FEATS)
    rnn_n_out = len(rnn_meta.get("horizons", HORIZONS))
    rnn_sc_feat = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_feat.pkl"))
    rnn_sc_targ = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_targ.pkl"))
    rnn_model = RNNForecast(input_size=rnn_n_feat, out_size=rnn_n_out).to(DEVICE)
    rnn_model.load_state_dict(
        torch.load(
            os.path.join(RNN_DIR, "rnn_multi_horizon.pth"),
            map_location=DEVICE,
            weights_only=False,
        )
    )
    rnn_model.eval()
    print(f"RNN loaded (input={rnn_n_feat} features, output={rnn_n_out} horizons).")
except FileNotFoundError as e:
    print(f"RNN not loaded (missing file): {e}")

rf_model = rf_features = None
rf_window = 252
try:
    rf_model = joblib.load(os.path.join(RF_DIR, "rf_multi_horizon.pkl"))
    rf_features = joblib.load(os.path.join(RF_DIR, "feature_list_multi.pkl"))
    print("Random Forest loaded.")
except FileNotFoundError as e:
    print(f"Random Forest not loaded (missing file): {e}")
    print("Run: python rf_final/train_rf.py  to train and save the RF model.")

print("Startup complete.")


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


def predict_transformer(
    df_ohlcv: pd.DataFrame, close: float, sector_id: int = 7
) -> dict:
    """
    Runs PatchTST on the full 36-feature set for the last tf_window days.
    Returns p10/p50/p90 for 1w, 1m, 6m only.
    1d and 1y are not predicted — 1d is too noisy, 1y has too much uncertainty.
    """
    if tf_model is None:
        raise ValueError(
            "Transformer not available. Run: cd transformer_final && python train_transformer.py"
        )

    df_feat = compute_features(df_ohlcv, sector_id)

    if len(df_feat) < tf_window:
        raise ValueError(
            f"Need at least {tf_window} rows after feature computation. Got {len(df_feat)}."
        )

    window = df_feat[FEATS].tail(tf_window).values
    window_s = tf_sc_feat.transform(window)
    x = torch.tensor(window_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    sec = torch.tensor([sector_id], dtype=torch.long).to(DEVICE)

    _enable_dropout(tf_model)
    samples = []
    with torch.no_grad():
        for _ in range(30):
            samples.append(tf_model(x, sec).cpu().numpy())
    tf_model.eval()

    samples = np.stack(samples, axis=0)  # (30, 1, 3, 3)
    p10_scaled = np.percentile(samples[:, 0, :, 0], 10, axis=0)
    p50_scaled = np.median(samples[:, 0, :, 1], axis=0)
    p90_scaled = np.percentile(samples[:, 0, :, 2], 90, axis=0)

    p50_ret = tf_sc_ret.inverse_transform(p50_scaled.reshape(1, -1)).flatten()
    p10_ret = tf_sc_ret.inverse_transform(p10_scaled.reshape(1, -1)).flatten()
    p90_ret = tf_sc_ret.inverse_transform(p90_scaled.reshape(1, -1)).flatten()

    p50 = {
        k: round(close * (1 + p50_ret[i]), 2)
        for i, k in enumerate(HORIZONS_TRANSFORMER)
    }
    p10 = {
        k: round(close * (1 + p10_ret[i]), 2)
        for i, k in enumerate(HORIZONS_TRANSFORMER)
    }
    p90 = {
        k: round(close * (1 + p90_ret[i]), 2)
        for i, k in enumerate(HORIZONS_TRANSFORMER)
    }

    return {"p50": p50, "p10": p10, "p90": p90}


def predict_lstm(df_tech: pd.DataFrame, close: float) -> dict:
    if lstm_model is None:
        raise ValueError(
            "LSTM not available. Run: python lstm_final_project/train_lstm.py"
        )
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
    if rnn_model is None:
        raise ValueError("RNN not available. Run: python rnn_final/train_rnn.py")
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
    if rf_model is None:
        raise ValueError(
            "Random Forest not available. Run: python rf_final/train_rf.py"
        )
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


# ---------------------------------------------------------------------------
# Guardrail 1 — Ticker validation
# Valid tickers are 1-5 uppercase letters, optionally followed by a dot and
# 1-2 more letters (e.g. BRK.B). We reject anything that doesn't match
# before making any network calls to yfinance.
# ---------------------------------------------------------------------------
TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")


def validate_ticker(ticker: str):
    """Raises ValueError if the ticker format is invalid."""
    t = ticker.strip().upper()
    if not t:
        raise ValueError("Ticker is required.")
    if len(t) > 10:
        raise ValueError(f"Ticker '{t}' is too long.")
    if not TICKER_RE.match(t):
        raise ValueError(
            f"Invalid ticker format: '{t}'. "
            "Expected 1-5 uppercase letters, optionally followed by a dot and 1-2 letters (e.g. AAPL, BRK.B)."
        )
    return t


# ---------------------------------------------------------------------------
# Guardrail 2 — In-memory rate limiter
# Limits each IP to 20 predict requests per minute and 5 /predict/all
# requests per minute (since /all runs 4 models × 50 MC passes).
# Uses a simple sliding-window counter stored in a dict.
# For production, replace with Redis-backed Flask-Limiter.
# ---------------------------------------------------------------------------
_rate_store: dict = defaultdict(list)


def _is_rate_limited(
    ip: str, endpoint: str, max_calls: int, window_secs: int = 60
) -> bool:
    key = f"{ip}:{endpoint}"
    now = time.time()
    calls = [t for t in _rate_store[key] if now - t < window_secs]
    _rate_store[key] = calls
    if len(calls) >= max_calls:
        return True
    _rate_store[key].append(now)
    return False


def rate_limit(max_calls: int, window_secs: int = 60):
    """Decorator that rate-limits a route per IP address."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = request.headers.get(
                "X-Forwarded-For", request.remote_addr or "unknown"
            )
            ip = ip.split(",")[0].strip()
            if _is_rate_limited(ip, fn.__name__, max_calls, window_secs):
                logger.warning("Rate limit hit: ip=%s endpoint=%s", ip, fn.__name__)
                return jsonify(
                    {
                        "error": f"Rate limit exceeded. Max {max_calls} requests per {window_secs}s."
                    }
                ), 429
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Guardrail 3 — Prediction sanity checker
# Flags predictions that are physically implausible — more than 10x or
# less than 10% of the current price. These almost always mean the model
# received bad input data (delisted stock, extreme outlier in the window).
# The prediction is still returned but a warning is added to the response.
# ---------------------------------------------------------------------------
def sanity_check_predictions(predictions: dict, current_price: float) -> list:
    """Returns a list of warning strings for any horizon with an implausible prediction."""
    warnings = []
    if current_price <= 0:
        return ["Current price is zero or negative — data may be invalid."]
    for horizon, price in predictions.items():
        if price is None:
            continue
        ratio = price / current_price
        if ratio > 10:
            warnings.append(
                f"{horizon}: predicted ${price:.2f} is >10x current price (${current_price:.2f}). "
                "Model may have received bad input data."
            )
        elif ratio < 0.1:
            warnings.append(
                f"{horizon}: predicted ${price:.2f} is <10% of current price (${current_price:.2f}). "
                "Model may have received bad input data."
            )
    return warnings


# ---------------------------------------------------------------------------
# Guardrail 4 — Request logger
# Logs every incoming predict request with IP, ticker, model, and latency.
# Useful for monitoring abuse patterns and debugging production issues.
# ---------------------------------------------------------------------------
def log_request(ticker: str, model: str, latency_ms: float, status: str):
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    logger.info(
        "predict ticker=%s model=%s status=%s latency_ms=%.0f ip=%s",
        ticker,
        model,
        status,
        latency_ms,
        ip,
    )


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
@rate_limit(max_calls=20, window_secs=60)
def predict_single(ticker):
    """
    GET /api/predict/<ticker>?model=transformer|lstm|rnn|rf

    Returns p50 (median), p10 (lower bound), p90 (upper bound) for all 5 horizons,
    along with the recommendation, chart data, and current price.
    Rate limited to 20 requests per minute per IP.
    """
    t_start = time.time()
    model_name = request.args.get("model", "transformer").lower()

    try:
        ticker = validate_ticker(ticker)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if model_name not in ("transformer", "lstm", "rnn", "rf"):
        return jsonify(
            {"error": f"Unknown model: '{model_name}'. Use: transformer, lstm, rnn, rf"}
        ), 400

    try:
        df_ohlcv = fetch_ohlcv(ticker)
        df_tech = compute_features(df_ohlcv)  # 36-feature DataFrame for all models
        close = float(df_tech["Close"].iloc[-1])
        pd_data = price_data_dict(df_tech)
        chart = build_chart_b64(df_tech, ticker)

        if model_name == "transformer":
            result = predict_transformer(df_ohlcv, close)
        elif model_name == "lstm":
            result = predict_lstm(df_tech, close)
        elif model_name == "rnn":
            result = predict_rnn(df_tech, close)
        else:
            result = predict_rf(df_ohlcv)

        rec = generate_recommendation(close, result["p50"], pd_data)
        warnings = sanity_check_predictions(result["p50"], close)

        latency = (time.time() - t_start) * 1000
        log_request(ticker, model_name, latency, "ok")

        return jsonify(
            {
                "ticker": ticker,
                "model": model_name,
                "last_close_price": round(close, 2),
                "current_price": round(close, 2),
                "predictions": result["p50"],
                "predicted_prices": result["p50"],
                "p10": result["p10"],
                "p90": result["p90"],
                "recommendation": rec,
                "warnings": warnings,
                "price_data": pd_data,
                "technical_data": pd_data,
                "chart_base64": chart,
            }
        )

    except Exception as e:
        latency = (time.time() - t_start) * 1000
        log_request(ticker, model_name, latency, f"error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/all/<ticker>")
@rate_limit(max_calls=5, window_secs=60)
def predict_all(ticker):
    """
    GET /api/predict/all/<ticker>

    Runs all 4 models and returns their predictions side by side.
    Rate limited to 5 requests per minute per IP because this route runs
    4 models × 50 MC passes each — it is significantly more expensive
    than the single-model route.
    """
    t_start = time.time()

    try:
        ticker = validate_ticker(ticker)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        df_ohlcv = fetch_ohlcv(ticker)
        df_tech = compute_features(df_ohlcv)
        close = float(df_tech["Close"].iloc[-1])
        pd_data = price_data_dict(df_tech)
        chart = build_chart_b64(df_tech, ticker)

        results = {}
        errors = {}
        all_warn = {}

        for name, fn in [
            ("transformer", lambda: predict_transformer(df_ohlcv, close)),
            ("lstm", lambda: predict_lstm(df_tech, close)),
            ("rnn", lambda: predict_rnn(df_tech, close)),
            ("rf", lambda: predict_rf(df_ohlcv)),
        ]:
            try:
                r = fn()
                results[name] = r
                w = sanity_check_predictions(r["p50"], close)
                if w:
                    all_warn[name] = w
            except Exception as e:
                errors[name] = str(e)

        latency = (time.time() - t_start) * 1000
        log_request(ticker, "all", latency, "ok")

        return jsonify(
            {
                "ticker": ticker,
                "last_close_price": round(close, 2),
                "predictions": results,
                "errors": errors,
                "warnings": all_warn,
                "price_data": pd_data,
                "chart_base64": chart,
            }
        )

    except Exception as e:
        latency = (time.time() - t_start) * 1000
        log_request(ticker, "all", latency, f"error: {e}")
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
    tf_model = PatchTST().to(DEVICE)
    tf_model.load_state_dict(
        torch.load(
            os.path.join(TRANSFORMER_DIR, "transformer_multi_horizon.pth"),
            map_location=DEVICE,
        )
    )
    tf_model.eval()

    lstm_meta = joblib.load(os.path.join(LSTM_DIR, "lstm_meta.pkl"))
    lstm_window = lstm_meta["window"]
    lstm_n_feat = lstm_meta.get("n_features", N_FEATS)
    lstm_n_out = len(lstm_meta.get("horizons", HORIZONS))
    lstm_sc_feat = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_feat.pkl"))
    lstm_sc_targ = joblib.load(os.path.join(LSTM_DIR, "lstm_scaler_targ.pkl"))
    lstm_model = LSTMForecast(input_size=lstm_n_feat, out_size=lstm_n_out).to(DEVICE)
    lstm_model.load_state_dict(
        torch.load(
            os.path.join(LSTM_DIR, "lstm_multi_horizon.pth"), map_location=DEVICE
        )
    )
    lstm_model.eval()

    rnn_meta = joblib.load(os.path.join(RNN_DIR, "rnn_meta.pkl"))
    rnn_window = rnn_meta["window"]
    rnn_n_feat = rnn_meta.get("n_features", N_FEATS)
    rnn_n_out = len(rnn_meta.get("horizons", HORIZONS))
    rnn_sc_feat = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_feat.pkl"))
    rnn_sc_targ = joblib.load(os.path.join(RNN_DIR, "rnn_scaler_targ.pkl"))
    rnn_model = RNNForecast(input_size=rnn_n_feat, out_size=rnn_n_out).to(DEVICE)
    rnn_model.load_state_dict(
        torch.load(os.path.join(RNN_DIR, "rnn_multi_horizon.pth"), map_location=DEVICE)
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
