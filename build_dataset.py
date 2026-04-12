"""
build_dataset.py
Downloads S&P 500 data and builds enriched feature windows for all four models.

Feature set (per day, per stock):
  OHLCV base:         Open, High, Low, Close, Volume
  Trend indicators:   SMA_10, SMA_50, SMA_200, MACD, MACD_signal
  Momentum:           RSI_14, MOM_5, ROC_21, Williams_%R
  Volatility:         ATR_14, BB_upper, BB_lower, BB_width
  Volume signals:     OBV_norm, Volume_SMA_20, Volume_ratio
  Candlestick:        body_size, upper_shadow, lower_shadow, body_pct,
                      doji, hammer, shooting_star, engulfing
  Price structure:    pct_from_52w_high, pct_from_52w_low, price_range_pct
  Market features:    vix_close, sp500_return_21d, sp500_return_63d
  Relative strength:  rel_strength_vs_sector_21d (how stock moved vs sector)

Horizons: 1 week (5d), 1 month (21d), 6 months (126d)
  1-day and 1-year removed:
    - 1-day is dominated by news/microstructure — price history cannot predict it
    - 1-year has too much macro uncertainty to be actionable

Run:
    python build_dataset.py           # use cached data if exists
    python build_dataset.py --refresh # force fresh download
"""

import os
import io
import json
import argparse
import requests
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

from data_guards import check_price_data, check_feature_array, log_dataset_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")

# Horizons: 1 week, 1 month, 6 months only
HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
MAX_H = max(HORIZONS.values())

WINDOW_LONG = 756  # Transformer + LSTM
WINDOW_SHORT = 252  # RNN + RF

IND_WIN = 200
MIN_ROWS_AFTER_TECH = WINDOW_LONG + MAX_H

MAX_WORKERS = 32

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
N_SECTORS = len(SECTOR_MAP)

# All feature names — used by training scripts to know column order
FEATS = [
    # OHLCV
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    # Trend
    "SMA_10",
    "SMA_50",
    "SMA_200",
    "MACD",
    "MACD_signal",
    # Momentum
    "RSI_14",
    "MOM_5",
    "ROC_21",
    "Williams_R",
    # Volatility
    "ATR_14",
    "BB_upper",
    "BB_lower",
    "BB_width",
    # Volume
    "OBV_norm",
    "Volume_SMA_20",
    "Volume_ratio",
    # Candlestick
    "body_size",
    "upper_shadow",
    "lower_shadow",
    "body_pct",
    "doji",
    "hammer",
    "shooting_star",
    "engulfing",
    # Price structure
    "pct_from_52w_high",
    "pct_from_52w_low",
    "price_range_pct",
    # Market (VIX + S&P500)
    "vix_close",
    "sp500_ret_21d",
    "sp500_ret_63d",
    # Relative strength
    "rel_strength_21d",
]
N_FEATS = len(FEATS)


def fetch_sp500_tickers() -> tuple:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (research project; contact via GitHub)"}
    html = requests.get(url, headers=headers, timeout=15).text
    df = pd.read_html(io.StringIO(html), header=0)[0]
    tickers = df["Symbol"].tolist()
    sc_col = "GICS Sector" if "GICS Sector" in df.columns else df.columns[3]
    sector_ids = {
        row["Symbol"]: SECTOR_MAP.get(str(row[sc_col]), 7) for _, row in df.iterrows()
    }
    return tickers, sector_ids


def fetch_market_data() -> tuple:
    """
    Downloads VIX and S&P 500 daily data.
    These are market-wide features added to every stock's feature window.
    VIX = fear index (high = elevated uncertainty, low = complacency)
    S&P 500 rolling return = regime indicator (bull vs bear)
    """
    logger.info("Downloading VIX and S&P 500 market data...")
    try:
        vix = yf.download("^VIX", period="6y", interval="1d", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix["Close"].dropna()
        vix.name = "vix_close"
    except Exception:
        vix = pd.Series(dtype=float, name="vix_close")

    try:
        sp = yf.download("^GSPC", period="6y", interval="1d", progress=False)
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = sp.columns.get_level_values(0)
        sp = sp["Close"].dropna()
        sp500_ret_21d = sp.pct_change(21)
        sp500_ret_21d.name = "sp500_ret_21d"
        sp500_ret_63d = sp.pct_change(63)
        sp500_ret_63d.name = "sp500_ret_63d"
    except Exception:
        sp500_ret_21d = pd.Series(dtype=float, name="sp500_ret_21d")
        sp500_ret_63d = pd.Series(dtype=float, name="sp500_ret_63d")

    return vix, sp500_ret_21d, sp500_ret_63d


def compute_sector_returns(raw_data: dict, sector_ids: dict) -> dict:
    """
    For each sector, compute the equal-weight daily return of all stocks in it.
    Used to compute each stock's return relative to its sector (relative strength).
    """
    sector_dfs = {i: [] for i in range(N_SECTORS)}
    for sym, df in raw_data.items():
        sid = sector_ids.get(sym, 7)
        ret = df["Close"].pct_change()
        ret.name = sym
        sector_dfs[sid].append(ret)

    sector_returns = {}
    for sid, series_list in sector_dfs.items():
        if series_list:
            sector_returns[sid] = pd.concat(series_list, axis=1).mean(axis=1)
        else:
            sector_returns[sid] = pd.Series(dtype=float)
    return sector_returns


def compute_features(
    df: pd.DataFrame,
    vix: pd.Series,
    sp500_ret_21d: pd.Series,
    sp500_ret_63d: pd.Series,
    sector_ret: pd.Series,
) -> pd.DataFrame:
    """
    Computes all features for one ticker.
    Returns a DataFrame with columns matching FEATS exactly.
    """
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

    # Candlestick features
    body = d["Close"] - d["Open"]
    full_range = (d["High"] - d["Low"]).replace(0, 1e-9)
    d["body_size"] = body.abs()
    d["upper_shadow"] = d["High"] - d[["Close", "Open"]].max(axis=1)
    d["lower_shadow"] = d[["Close", "Open"]].min(axis=1) - d["Low"]
    d["body_pct"] = body / full_range

    # Doji: body is < 10% of total range — indecision candle
    d["doji"] = (body.abs() < 0.1 * full_range).astype(float)

    # Hammer: small body at the top of range, long lower shadow, in downtrend
    prev_close_below_sma = d["Close"].shift(1) < d["SMA_50"].shift(1)
    small_body = body.abs() < 0.3 * full_range
    long_lower = d["lower_shadow"] > 2 * body.abs()
    tiny_upper = d["upper_shadow"] < 0.1 * full_range
    d["hammer"] = (prev_close_below_sma & small_body & long_lower & tiny_upper).astype(
        float
    )

    # Shooting star: small body at the bottom of range, long upper shadow, in uptrend
    prev_close_above_sma = d["Close"].shift(1) > d["SMA_50"].shift(1)
    long_upper = d["upper_shadow"] > 2 * body.abs()
    tiny_lower = d["lower_shadow"] < 0.1 * full_range
    d["shooting_star"] = (
        prev_close_above_sma & small_body & long_upper & tiny_lower
    ).astype(float)

    # Bullish engulfing: current candle body completely engulfs previous candle body
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

    # Market features — align by date index
    d["vix_close"] = vix.reindex(d.index).ffill().bfill()
    d["sp500_ret_21d"] = sp500_ret_21d.reindex(d.index).ffill().bfill()
    d["sp500_ret_63d"] = sp500_ret_63d.reindex(d.index).ffill().bfill()

    # Relative strength vs sector: stock 21d return minus sector 21d return
    stock_ret_21d = d["Close"].pct_change(21)
    sec_ret_21d = sector_ret.reindex(d.index).pct_change(21).ffill().bfill()
    d["rel_strength_21d"] = stock_ret_21d - sec_ret_21d

    # Return only the feature columns, drop rows with NaN
    return d[FEATS].dropna()


def _download_one(sym: str):
    try:
        yf_sym = sym.replace(".", "-").upper()
        df = yf.download(yf_sym, period="6y", interval="1d", progress=False)
        if df is None or df.empty:
            return sym, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return sym, df if not df.empty else None
    except Exception:
        return sym, None


def download_all(tickers: list) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, sym): sym for sym in tickers}
        done = 0
        for future in as_completed(futures):
            sym, df = future.result()
            done += 1
            if df is not None:
                results[sym] = df
            if done % 50 == 0 or done == len(tickers):
                logger.info(
                    "Downloaded %d/%d (%d succeeded)", done, len(tickers), len(results)
                )
    return results


def build_windows_long(tech: pd.DataFrame, sym: str, sector_id: int) -> tuple:
    arr = tech[FEATS].values
    date_index = tech.index

    X_list, Y_ret_list, Y_px_list, LC_list = [], [], [], []
    date_list, sector_list, ticker_list = [], [], []

    for i in range(len(arr) - WINDOW_LONG - MAX_H + 1):
        win = arr[i : i + WINDOW_LONG]
        base = float(win[-1, 3])  # last Close in window
        fut = [float(arr[i + WINDOW_LONG + h - 1, 3]) for h in HORIZONS.values()]
        rets = [(f / base - 1) for f in fut]
        X_list.append(win)
        Y_ret_list.append(rets)
        Y_px_list.append(fut)
        LC_list.append(base)
        date_list.append(str(date_index[i + WINDOW_LONG - 1].date()))
        sector_list.append(sector_id)
        ticker_list.append(sym)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(Y_ret_list, dtype=np.float32),
        np.array(Y_px_list, dtype=np.float32),
        np.array(LC_list, dtype=np.float32),
        np.array(date_list),
        np.array(sector_list, dtype=np.int32),
        np.array(ticker_list),
    )


def build_windows_short(tech: pd.DataFrame) -> tuple:
    arr = tech[FEATS].values
    ohlcv = tech[["Open", "High", "Low", "Close", "Volume"]].values

    X_seq_list, X_flat_list, Y_list = [], [], []

    for i in range(len(arr) - WINDOW_SHORT - MAX_H + 1):
        X_seq_list.append(arr[i : i + WINDOW_SHORT])
        X_flat_list.append(ohlcv[i : i + WINDOW_SHORT].flatten())
        Y_list.append(
            [float(arr[i + WINDOW_SHORT + h - 1, 3]) for h in HORIZONS.values()]
        )

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_flat_list, dtype=np.float32),
        np.array(Y_list, dtype=np.float32),
    )


def build(refresh: bool = False):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    meta_path = os.path.join(DATASET_DIR, "meta.json")
    long_path = os.path.join(DATASET_DIR, "windows_756.npz")
    short_path = os.path.join(DATASET_DIR, "windows_252.npz")

    if not refresh and os.path.exists(meta_path) and os.path.exists(long_path):
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(
            "Dataset cache found (built %s, %d tickers, %d long windows). "
            "Pass --refresh to rebuild.",
            meta.get("built_at", "unknown"),
            meta.get("n_tickers", 0),
            meta.get("n_windows_756", 0),
        )
        return

    logger.info("Fetching S&P 500 ticker list...")
    tickers, sector_ids = fetch_sp500_tickers()
    logger.info("Found %d tickers.", len(tickers))

    # Download market-wide data first (VIX + S&P500)
    vix, sp500_ret_21d, sp500_ret_63d = fetch_market_data()

    logger.info("Downloading ticker data in parallel...")
    raw_data = download_all(list(tickers))
    logger.info("Download complete: %d/%d tickers.", len(raw_data), len(tickers))

    # Compute sector average returns for relative strength feature
    logger.info("Computing sector returns...")
    sector_returns = compute_sector_returns(raw_data, sector_ids)

    # Use on-disk temporary files to accumulate windows without holding
    # everything in RAM. 150K windows × 756 × 36 = ~15 GB in memory.
    # Instead we write each ticker to a temp .npy file and concatenate at the end.
    tmp_dir = os.path.join(DATASET_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Lists of temp file paths — much lighter than arrays
    long_tmp_files = []  # each element is a dict of {key: path}
    short_tmp_files = []

    tickers_used = []
    date_min = date_max = None
    n_long = 0
    n_short = 0

    for sym, raw_df in raw_data.items():
        try:
            df = check_price_data(raw_df, sym)
            sid = sector_ids.get(sym, 7)
            sec_ret = sector_returns.get(sid, pd.Series(dtype=float))
            tech = compute_features(df, vix, sp500_ret_21d, sp500_ret_63d, sec_ret)

            if len(tech) < MIN_ROWS_AFTER_TECH:
                continue

            tech.to_csv(os.path.join(RAW_DIR, f"{sym}.csv"))

            if len(tech) >= WINDOW_LONG + MAX_H:
                X_l, Y_r, Y_p, LC, dates, secs, tks = build_windows_long(tech, sym, sid)
                if len(X_l) > 0:
                    prefix = os.path.join(tmp_dir, f"long_{sym}")
                    np.save(f"{prefix}_X.npy", X_l)
                    np.save(f"{prefix}_Yr.npy", Y_r)
                    np.save(f"{prefix}_Yp.npy", Y_p)
                    np.save(f"{prefix}_LC.npy", LC)
                    np.save(f"{prefix}_dt.npy", dates)
                    np.save(f"{prefix}_sec.npy", secs)
                    np.save(f"{prefix}_tk.npy", tks)
                    long_tmp_files.append(prefix)
                    n_long += len(X_l)

            if len(tech) >= WINDOW_SHORT + MAX_H:
                X_seq, X_flat, Y_s = build_windows_short(tech)
                if len(X_seq) > 0:
                    prefix = os.path.join(tmp_dir, f"short_{sym}")
                    np.save(f"{prefix}_Xs.npy", X_seq)
                    np.save(f"{prefix}_Xf.npy", X_flat)
                    np.save(f"{prefix}_Y.npy", Y_s)
                    short_tmp_files.append(prefix)
                    n_short += len(X_seq)

            tickers_used.append(sym)
            idx_min, idx_max = tech.index.min(), tech.index.max()
            if date_min is None or idx_min < date_min:
                date_min = idx_min
            if date_max is None or idx_max > date_max:
                date_max = idx_max

        except Exception as e:
            logger.debug("Skip %s: %s", sym, e)

    if not long_tmp_files:
        raise RuntimeError(
            "No windows created. Check internet connection and yfinance availability."
        )

    logger.info(
        "Concatenating %d long windows from %d tickers...", n_long, len(long_tmp_files)
    )

    # Concatenate on disk — load each file and write directly to the output npz
    # This avoids holding all data in RAM at once
    X_long = np.empty((n_long, WINDOW_LONG, N_FEATS), dtype=np.float32)
    Y_ret = np.empty((n_long, len(HORIZONS)), dtype=np.float32)
    Y_px_long = np.empty((n_long, len(HORIZONS)), dtype=np.float32)
    LC_arr = np.empty((n_long,), dtype=np.float32)
    dates_arr = np.empty((n_long,), dtype=object)
    sectors_arr = np.empty((n_long,), dtype=np.int32)
    tickers_arr = np.empty((n_long,), dtype=object)

    idx = 0
    for prefix in long_tmp_files:
        X_l = np.load(f"{prefix}_X.npy")
        n = len(X_l)
        X_long[idx : idx + n] = X_l
        Y_ret[idx : idx + n] = np.load(f"{prefix}_Yr.npy")
        Y_px_long[idx : idx + n] = np.load(f"{prefix}_Yp.npy")
        LC_arr[idx : idx + n] = np.load(f"{prefix}_LC.npy")
        dates_arr[idx : idx + n] = np.load(f"{prefix}_dt.npy", allow_pickle=True)
        sectors_arr[idx : idx + n] = np.load(f"{prefix}_sec.npy")
        tickers_arr[idx : idx + n] = np.load(f"{prefix}_tk.npy", allow_pickle=True)
        idx += n

    logger.info("Concatenating %d short windows...", n_short)
    X_seq_arr = np.empty((n_short, WINDOW_SHORT, N_FEATS), dtype=np.float32)
    X_flat_arr = np.empty((n_short, WINDOW_SHORT * 5), dtype=np.float32)
    Y_short = np.empty((n_short, len(HORIZONS)), dtype=np.float32)

    idx = 0
    for prefix in short_tmp_files:
        Xs = np.load(f"{prefix}_Xs.npy")
        n = len(Xs)
        X_seq_arr[idx : idx + n] = Xs
        X_flat_arr[idx : idx + n] = np.load(f"{prefix}_Xf.npy")
        Y_short[idx : idx + n] = np.load(f"{prefix}_Y.npy")
        idx += n

    # Spot-check a sample for NaN/Inf
    sample_idx = np.random.choice(len(X_long), min(1000, len(X_long)), replace=False)
    check_feature_array(X_long[sample_idx], "X_long sample (756)")
    check_feature_array(X_seq_arr[sample_idx[: len(X_seq_arr)]], "X_seq sample (252)")
    log_dataset_summary(X_long, Y_ret, n_tickers=len(tickers_used))

    np.savez_compressed(
        long_path,
        X=X_long,
        Y_ret=Y_ret,
        Y_px=Y_px_long,
        LC=LC_arr,
        dates=dates_arr,
        sectors=sectors_arr,
        tickers=tickers_arr,
    )
    np.savez_compressed(short_path, X_seq=X_seq_arr, X_flat=X_flat_arr, Y=Y_short)

    # Clean up temp files
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Temp files cleaned up.")

    meta = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "n_tickers": len(tickers_used),
        "tickers": tickers_used,
        "date_min": str(date_min.date()) if date_min else None,
        "date_max": str(date_max.date()) if date_max else None,
        "horizons": HORIZONS,
        "features": FEATS,
        "n_features": N_FEATS,
        "window_long": WINDOW_LONG,
        "window_short": WINDOW_SHORT,
        "n_windows_756": int(n_long),
        "n_windows_252": int(n_short),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Dataset saved.\n"
        "  Features: %d per day\n"
        "  Long windows (756): %s\n"
        "  Short windows (252): %s\n"
        "  Tickers: %d  Date range: %s to %s",
        N_FEATS,
        X_long.shape,
        X_seq.shape,
        len(tickers_used),
        meta["date_min"],
        meta["date_max"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build shared dataset for all 4 models"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-download even if cache exists"
    )
    args = parser.parse_args()
    build(refresh=args.refresh)
