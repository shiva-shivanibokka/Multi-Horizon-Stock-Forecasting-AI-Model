"""
build_dataset.py
Downloads 5 years of S&P 500 data once and builds the window arrays
that all four training scripts need.

Run this before training any model:
    python build_dataset.py

What it does:
    1. Fetches all S&P 500 tickers from Wikipedia
    2. Downloads 5y of daily OHLCV data for all tickers in parallel (32 workers)
    3. Runs data_guards.check_price_data() on each ticker — cleans bad rows,
       clips extreme returns, rejects tickers with too little or too much missing data
    4. Computes 12 technical indicators for each ticker
    5. Builds four separate window arrays (one per model) and saves them to dataset/
    6. Saves a metadata file so training scripts know what was built and when

Output files (saved to dataset/):
    raw/              One CSV per ticker — raw cleaned OHLCV+indicators
    windows_756.npz   Window arrays for Transformer and LSTM (WINDOW=756)
    windows_252.npz   Window arrays for RNN and RF (WINDOW=252)
    meta.json         Build metadata: tickers used, date, data range, shapes

Training scripts load from dataset/ instead of downloading again. This means:
    - Data is downloaded once, not four times
    - All models train on exactly the same tickers and the same date range
    - Re-running training after a hyperparameter change is fast (no re-download)
    - The dataset can be inspected and versioned separately from model weights

To force a fresh download (e.g. weekly retraining):
    python build_dataset.py --refresh
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

HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "6m": 126, "1y": 252}
MAX_H = max(HORIZONS.values())

# Window sizes per model group
WINDOW_LONG = 756  # Transformer + LSTM (3 years)
WINDOW_SHORT = 252  # RNN + RF (1 year)

# Minimum rows a ticker needs after cleaning
# Must cover at least one full long window + the longest forecast horizon + indicator warmup
IND_WIN = 200
MIN_ROWS = WINDOW_LONG + MAX_H + IND_WIN

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

MAX_WORKERS = 32


def fetch_sp500_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (research project; contact via GitHub)"}
    html = requests.get(url, headers=headers, timeout=15).text
    return pd.read_html(io.StringIO(html), header=0)[0]["Symbol"].tolist()


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
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["MOM_1"] = df["Close"].diff(1)
    df["ROC_14"] = df["Close"].pct_change(14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()


def _download_one(sym: str):
    """Downloads 5y OHLCV for one ticker. Returns (sym, df) or (sym, None)."""
    try:
        yf_sym = sym.replace(".", "-").upper()
        df = yf.download(yf_sym, period="5y", interval="1d", progress=False)
        if df is None or df.empty:
            return sym, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return sym, df if not df.empty else None
    except Exception:
        return sym, None


def download_all(tickers: list) -> dict:
    """Downloads all tickers in parallel. Returns {sym: df}."""
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
                    "Downloaded %d/%d tickers (%d succeeded so far)",
                    done,
                    len(tickers),
                    len(results),
                )
    return results


def build_windows_long(tech: pd.DataFrame) -> tuple:
    """
    Builds sliding windows for Transformer and LSTM (WINDOW=756).

    Returns:
        X      (n_windows, 756, 12)  — feature windows
        Y_ret  (n_windows, 5)        — future log returns (for Transformer)
        Y_px   (n_windows, 5)        — future absolute prices (for LSTM)
        LC     (n_windows,)          — last close price of each window
    """
    arr = tech[FEATS].values
    X_list, Y_ret_list, Y_px_list, LC_list = [], [], [], []

    for i in range(len(arr) - WINDOW_LONG - MAX_H + 1):
        win = arr[i : i + WINDOW_LONG]
        base = win[-1, 3]
        fut = [arr[i + WINDOW_LONG + h - 1, 3] for h in HORIZONS.values()]
        rets = [(f / base - 1) for f in fut]
        X_list.append(win)
        Y_ret_list.append(rets)
        Y_px_list.append(fut)
        LC_list.append(base)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(Y_ret_list, dtype=np.float32),
        np.array(Y_px_list, dtype=np.float32),
        np.array(LC_list, dtype=np.float32),
    )


def build_windows_short(tech: pd.DataFrame) -> tuple:
    """
    Builds sliding windows for RNN and RF (WINDOW=252).

    Returns:
        X_seq  (n_windows, 252, 12)   — sequential windows (for RNN)
        X_flat (n_windows, 252*5)     — flattened OHLCV windows (for RF)
        Y      (n_windows, 5)         — future absolute prices
    """
    arr = tech[FEATS].values
    ohlcv = tech[["Open", "High", "Low", "Close", "Volume"]].values

    X_seq_list, X_flat_list, Y_list = [], [], []

    for i in range(len(arr) - WINDOW_SHORT - MAX_H + 1):
        win_seq = arr[i : i + WINDOW_SHORT]
        win_flat = ohlcv[i : i + WINDOW_SHORT].flatten()
        fut = [arr[i + WINDOW_SHORT + h - 1, 3] for h in HORIZONS.values()]
        X_seq_list.append(win_seq)
        X_flat_list.append(win_flat)
        Y_list.append(fut)

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

    # Skip rebuild if cache already exists and --refresh was not passed
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

    logger.info("Fetching S&P 500 ticker list from Wikipedia...")
    tickers = fetch_sp500_tickers()
    logger.info("Found %d tickers. Starting parallel download...", len(tickers))

    raw_data = download_all(tickers)
    logger.info(
        "Download complete: %d/%d tickers succeeded.", len(raw_data), len(tickers)
    )

    # Accumulate windows across all tickers
    X_long_list = []
    Y_ret_list = []
    Y_px_long_list = []
    LC_list = []
    X_seq_list = []
    X_flat_list = []
    Y_short_list = []

    tickers_used = []
    date_min = None
    date_max = None

    for sym, raw_df in raw_data.items():
        try:
            df = check_price_data(raw_df, sym)
            if len(df) < MIN_ROWS:
                logger.debug(
                    "Skip %s: only %d rows after cleaning (need %d)",
                    sym,
                    len(df),
                    MIN_ROWS,
                )
                continue

            tech = compute_technicals(df)
            if len(tech) < MIN_ROWS:
                continue

            # Save the cleaned ticker data as CSV for inspection
            csv_path = os.path.join(RAW_DIR, f"{sym}.csv")
            tech.to_csv(csv_path)

            # Build long windows (Transformer + LSTM)
            if len(tech) >= WINDOW_LONG + MAX_H:
                X_l, Y_r, Y_p, LC = build_windows_long(tech)
                if len(X_l) > 0:
                    X_long_list.append(X_l)
                    Y_ret_list.append(Y_r)
                    Y_px_long_list.append(Y_p)
                    LC_list.append(LC)

            # Build short windows (RNN + RF)
            if len(tech) >= WINDOW_SHORT + MAX_H:
                X_seq, X_flat, Y_s = build_windows_short(tech)
                if len(X_seq) > 0:
                    X_seq_list.append(X_seq)
                    X_flat_list.append(X_flat)
                    Y_short_list.append(Y_s)

            tickers_used.append(sym)
            idx_min = tech.index.min()
            idx_max = tech.index.max()
            if date_min is None or idx_min < date_min:
                date_min = idx_min
            if date_max is None or idx_max > date_max:
                date_max = idx_max

        except Exception as e:
            logger.debug("Skip %s: %s", sym, e)
            continue

    if not X_long_list:
        raise RuntimeError(
            "No windows were created. Check your internet connection "
            "and that yfinance is returning data."
        )

    # Concatenate all tickers
    X_long = np.concatenate(X_long_list, axis=0)
    Y_ret = np.concatenate(Y_ret_list, axis=0)
    Y_px_long = np.concatenate(Y_px_long_list, axis=0)
    LC = np.concatenate(LC_list, axis=0)
    X_seq = np.concatenate(X_seq_list, axis=0)
    X_flat = np.concatenate(X_flat_list, axis=0)
    Y_short = np.concatenate(Y_short_list, axis=0)

    # Validate arrays before saving
    check_feature_array(X_long, "X_long (756)")
    check_feature_array(X_seq, "X_seq (252)")
    check_feature_array(X_flat, "X_flat (252 OHLCV)")

    log_dataset_summary(X_long, Y_ret, n_tickers=len(tickers_used))

    # Save window arrays
    np.savez_compressed(
        long_path,
        X=X_long,
        Y_ret=Y_ret,
        Y_px=Y_px_long,
        LC=LC,
    )
    np.savez_compressed(
        short_path,
        X_seq=X_seq,
        X_flat=X_flat,
        Y=Y_short,
    )

    # Save metadata
    meta = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "n_tickers": len(tickers_used),
        "tickers": tickers_used,
        "date_min": str(date_min.date()) if date_min else None,
        "date_max": str(date_max.date()) if date_max else None,
        "window_long": WINDOW_LONG,
        "window_short": WINDOW_SHORT,
        "horizons": HORIZONS,
        "features": FEATS,
        "n_windows_756": int(len(X_long)),
        "n_windows_252": int(len(X_seq)),
        "x_long_shape": list(X_long.shape),
        "x_seq_shape": list(X_seq.shape),
        "x_flat_shape": list(X_flat.shape),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Dataset saved to dataset/\n"
        "  Long windows (756): %s  ->  %s\n"
        "  Short windows (252): seq=%s  flat=%s\n"
        "  Tickers: %d  Date range: %s to %s",
        X_long.shape,
        long_path,
        X_seq.shape,
        X_flat.shape,
        len(tickers_used),
        meta["date_min"],
        meta["date_max"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build shared dataset for all 4 models"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if a cached dataset already exists",
    )
    args = parser.parse_args()
    build(refresh=args.refresh)
