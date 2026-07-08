import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50", "SMA_200", "MACD", "MACD_signal",
    "RSI_14", "MOM_5", "ROC_21", "Williams_R",
    "ATR_14", "BB_upper", "BB_lower", "BB_width",
    "OBV_norm", "Volume_SMA_20", "Volume_ratio",
    "body_size", "upper_shadow", "lower_shadow", "body_pct",
    "doji", "hammer", "shooting_star", "engulfing",
    "pct_from_52w_high", "pct_from_52w_low", "price_range_pct",
    "vix_close", "sp500_ret_21d", "sp500_ret_63d",
]


def compute_features(df: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.copy()

    d["SMA_10"] = d["Close"].rolling(10).mean()
    d["SMA_50"] = d["Close"].rolling(50).mean()
    d["SMA_200"] = d["Close"].rolling(200).mean()
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI_14"] = 100 - 100 / (1 + rs)
    d["MOM_5"] = d["Close"].diff(5)
    d["ROC_21"] = d["Close"].pct_change(21)
    high14 = d["High"].rolling(14).max()
    low14 = d["Low"].rolling(14).min()
    d["Williams_R"] = -100 * (high14 - d["Close"]) / (high14 - low14 + 1e-9)

    tr = pd.concat(
        [(d["High"] - d["Low"]),
         (d["High"] - d["Close"].shift(1)).abs(),
         (d["Low"] - d["Close"].shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["BB_upper"] = sma20 + 2 * std20
    d["BB_lower"] = sma20 - 2 * std20
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (sma20 + 1e-9)

    obv = (np.sign(d["Close"].diff()).fillna(0) * d["Volume"]).cumsum()
    d["OBV_norm"] = obv / obv.rolling(252).max().replace(0, 1).abs()
    d["Volume_SMA_20"] = d["Volume"].rolling(20).mean()
    d["Volume_ratio"] = d["Volume"] / (d["Volume_SMA_20"] + 1e-9)

    body = d["Close"] - d["Open"]
    full = (d["High"] - d["Low"]).replace(0, 1e-9)
    d["body_size"] = body.abs()
    d["upper_shadow"] = d["High"] - d[["Close", "Open"]].max(axis=1)
    d["lower_shadow"] = d[["Close", "Open"]].min(axis=1) - d["Low"]
    d["body_pct"] = body / full
    small = body.abs() < 0.3 * full
    d["doji"] = (body.abs() < 0.1 * full).astype(float)
    d["hammer"] = (
        (d["Close"].shift(1) < d["SMA_50"].shift(1)) & small
        & (d["lower_shadow"] > 2 * body.abs()) & (d["upper_shadow"] < 0.1 * full)
    ).astype(float)
    d["shooting_star"] = (
        (d["Close"].shift(1) > d["SMA_50"].shift(1)) & small
        & (d["upper_shadow"] > 2 * body.abs()) & (d["lower_shadow"] < 0.1 * full)
    ).astype(float)
    d["engulfing"] = (
        (body > 0) & ((d["Close"].shift(1) - d["Open"].shift(1)) < 0)
        & (d["Close"] > d["Open"].shift(1)) & (d["Open"] < d["Close"].shift(1))
    ).astype(float)

    high52 = d["High"].rolling(252).max()
    low52 = d["Low"].rolling(252).min()
    d["pct_from_52w_high"] = (d["Close"] - high52) / (high52 + 1e-9)
    d["pct_from_52w_low"] = (d["Close"] - low52) / (low52 + 1e-9)
    d["price_range_pct"] = full / (d["Close"] + 1e-9)

    if market is not None:
        aligned = market.reindex(d.index).ffill()  # ffill ONLY — never bfill
        d["vix_close"] = aligned.get("vix_close")
        d["sp500_ret_21d"] = aligned.get("sp500_ret_21d")
        d["sp500_ret_63d"] = aligned.get("sp500_ret_63d")
    else:
        d["vix_close"] = np.nan
        d["sp500_ret_21d"] = np.nan
        d["sp500_ret_63d"] = np.nan

    return d[FEATURES].dropna()
