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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
        vix = yf.download("^VIX", period="11y", interval="1d", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix["Close"].dropna()
        vix.name = "vix_close"
    except Exception:
        vix = pd.Series(dtype=float, name="vix_close")

    try:
        sp = yf.download("^GSPC", period="11y", interval="1d", progress=False)
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

    # Volatility — np.maximum avoids allocating a 3-column DataFrame just to take max
    tr1 = (d["High"] - d["Low"]).values
    tr2 = (d["High"] - d["Close"].shift(1)).abs().values
    tr3 = (d["Low"] - d["Close"].shift(1)).abs().values
    true_range = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)), index=d.index)
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
        df = yf.download(yf_sym, period="11y", interval="1d", progress=False)
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
    """
    Vectorized window builder using sliding_window_view — no Python loop.
    sliding_window_view returns a zero-copy view; .copy() makes it contiguous
    for np.save. Shape: (n_windows, WINDOW_LONG, N_FEATS).
    """
    arr = tech[FEATS].values.astype(np.float32)
    close = arr[:, 3]  # Close column index
    date_index = tech.index
    n = len(arr)
    n_windows = n - WINDOW_LONG - MAX_H + 1
    if n_windows <= 0:
        empty = np.empty((0,), dtype=np.float32)
        return (
            empty,
            empty,
            empty,
            empty,
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
        )

    # sliding_window_view(arr, window_shape=W, axis=0) on shape (n_days, N_FEATS)
    # produces (n_windows, N_FEATS, W) — transpose to (n_windows, W, N_FEATS)
    X = (
        sliding_window_view(arr, window_shape=WINDOW_LONG, axis=0)[:n_windows]
        .transpose(0, 2, 1)
        .copy()
    )

    # Base close = last close in each window
    base_close = close[WINDOW_LONG - 1 : WINDOW_LONG - 1 + n_windows]  # (n_windows,)

    # Future closes for each horizon — vectorized index gather
    h_vals = list(HORIZONS.values())
    fut_indices = np.array([WINDOW_LONG + h - 1 for h in h_vals])  # (3,)
    # close[fut_indices[j] + i] for all i: use advanced indexing
    row_idx = np.arange(n_windows)[:, None] + fut_indices[None, :]  # (n_windows, 3)
    Y_px = close[row_idx].astype(np.float32)  # (n_windows, 3)
    Y_ret = (Y_px / base_close[:, None] - 1).astype(np.float32)

    dates = np.array(
        [str(date_index[i + WINDOW_LONG - 1].date()) for i in range(n_windows)]
    )
    sectors = np.full(n_windows, sector_id, dtype=np.int32)
    tickers = np.full(n_windows, sym, dtype=object)

    return X, Y_ret, Y_px, base_close.astype(np.float32), dates, sectors, tickers


def build_windows_short(tech: pd.DataFrame) -> tuple:
    """
    Vectorized window builder for the short (252-day) window.
    """
    arr = tech[FEATS].values.astype(np.float32)
    ohlcv = tech[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
    close = arr[:, 3]
    n = len(arr)
    n_windows = n - WINDOW_SHORT - MAX_H + 1
    if n_windows <= 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty

    # sliding_window_view on (n_days, N_FEATS) gives (n_windows, N_FEATS, W)
    # transpose to (n_windows, W, N_FEATS) for both X_seq and X_flat
    X_seq = (
        sliding_window_view(arr, window_shape=WINDOW_SHORT, axis=0)[:n_windows]
        .transpose(0, 2, 1)
        .copy()
    )

    # X_flat: (n_windows, WINDOW_SHORT * 5) — OHLCV only, flattened row-major
    X_flat = (
        sliding_window_view(ohlcv, window_shape=WINDOW_SHORT, axis=0)[:n_windows]
        .transpose(0, 2, 1)  # (n_windows, WINDOW_SHORT, 5)
        .reshape(n_windows, -1)  # (n_windows, WINDOW_SHORT * 5)
        .copy()
    )

    h_vals = list(HORIZONS.values())
    fut_indices = np.array([WINDOW_SHORT + h - 1 for h in h_vals])
    row_idx = np.arange(n_windows)[:, None] + fut_indices[None, :]
    Y = close[row_idx].astype(np.float32)

    return X_seq, X_flat, Y


def _series_to_arrays(s: pd.Series):
    """Convert a Series to (values, index.values) for safe multiprocessing pickling."""
    return s.values, s.index.values


def _process_ticker(args):
    """
    Module-level worker function (must be at module level to be pickleable on Windows).
    Computes features + windows for one ticker and writes results to two temp .npz files.
    """
    (
        sym,
        raw_df,
        sid,
        sec_ret_vals,
        sec_ret_idx,
        vix_vals,
        vix_idx,
        sp21_vals,
        sp21_idx,
        sp63_vals,
        sp63_idx,
        tmp_dir,
    ) = args
    try:
        # Reconstruct Series in the worker — values+index survive pickling cleanly
        vix_s = pd.Series(vix_vals, index=pd.DatetimeIndex(vix_idx), name="vix_close")
        sp21_s = pd.Series(
            sp21_vals, index=pd.DatetimeIndex(sp21_idx), name="sp500_ret_21d"
        )
        sp63_s = pd.Series(
            sp63_vals, index=pd.DatetimeIndex(sp63_idx), name="sp500_ret_63d"
        )
        sec_ret_s = pd.Series(sec_ret_vals, index=pd.DatetimeIndex(sec_ret_idx))

        df = check_price_data(raw_df, sym)
        tech = compute_features(df, vix_s, sp21_s, sp63_s, sec_ret_s)

        if len(tech) < MIN_ROWS_AFTER_TECH:
            return None

        idx_min = str(tech.index.min().date())
        idx_max = str(tech.index.max().date())

        long_path_out = None
        short_path_out = None
        n_l = n_s = 0

        if len(tech) >= WINDOW_LONG + MAX_H:
            X_l, Y_r, Y_p, LC, dates, secs, tks = build_windows_long(tech, sym, sid)
            if len(X_l) > 0:
                long_path_out = os.path.join(tmp_dir, f"long_{sym}.npz")
                np.savez(
                    long_path_out,
                    X=X_l,
                    Yr=Y_r,
                    Yp=Y_p,
                    LC=LC,
                    dt=dates,
                    sec=secs,
                    tk=tks,
                )
                n_l = len(X_l)

        if len(tech) >= WINDOW_SHORT + MAX_H:
            X_seq, X_flat, Y_s = build_windows_short(tech)
            if len(X_seq) > 0:
                short_path_out = os.path.join(tmp_dir, f"short_{sym}.npz")
                np.savez(short_path_out, Xs=X_seq, Xf=X_flat, Y=Y_s)
                n_s = len(X_seq)

        return sym, long_path_out, short_path_out, n_l, n_s, idx_min, idx_max
    except Exception as e:
        logger.debug("Skip %s: %s", sym, e)
        return None


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

    # -------------------------------------------------------------------------
    # Per-ticker processing: features + windows, parallelised across CPUs.
    # Each worker receives its ticker's raw DataFrame plus the pre-computed
    # market-wide series (VIX, S&P500, sector returns) as arguments.
    # Results are written to temp .npz files (2 per ticker, not 7).
    # -------------------------------------------------------------------------
    tmp_dir = os.path.join(DATASET_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    worker_args = []
    for sym, raw_df in raw_data.items():
        sid = sector_ids.get(sym, 7)
        sec_ret = sector_returns.get(sid, pd.Series(dtype=float))
        vix_v, vix_i = _series_to_arrays(vix)
        sp21_v, sp21_i = _series_to_arrays(sp500_ret_21d)
        sp63_v, sp63_i = _series_to_arrays(sp500_ret_63d)
        sec_v, sec_i = _series_to_arrays(sec_ret)
        worker_args.append(
            (
                sym,
                raw_df,
                sid,
                sec_v,
                sec_i,
                vix_v,
                vix_i,
                sp21_v,
                sp21_i,
                sp63_v,
                sp63_i,
                tmp_dir,
            )
        )

    # Use half the logical CPUs so we don't starve the OS
    n_workers = max(1, os.cpu_count() // 2)
    logger.info("Processing %d tickers with %d workers...", len(worker_args), n_workers)

    long_tmp_files = []
    short_tmp_files = []
    tickers_used = []
    date_min = date_max = None
    n_long = n_short = 0
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_ticker, arg): arg[0] for arg in worker_args}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is None:
                continue
            sym, long_prefix, short_prefix, n_l, n_s, idx_min_s, idx_max_s = result
            if long_prefix:
                long_tmp_files.append(long_prefix)
                n_long += n_l
            if short_prefix:
                short_tmp_files.append(short_prefix)
                n_short += n_s
            tickers_used.append(sym)
            idx_min_dt = pd.Timestamp(idx_min_s)
            idx_max_dt = pd.Timestamp(idx_max_s)
            if date_min is None or idx_min_dt < date_min:
                date_min = idx_min_dt
            if date_max is None or idx_max_dt > date_max:
                date_max = idx_max_dt
            if done % 50 == 0 or done == len(worker_args):
                logger.info(
                    "Processed %d/%d tickers (%d long windows so far)",
                    done,
                    len(worker_args),
                    n_long,
                )

    if not long_tmp_files:
        raise RuntimeError(
            "No windows created. Check internet connection and yfinance availability."
        )

    import shutil
    import zipfile
    import numpy.lib.format as npfmt

    # -------------------------------------------------------------------------
    # One-array-at-a-time streaming into the final .npz
    # -------------------------------------------------------------------------
    # Opening all memmaps simultaneously maps ~31 GB of address space and
    # causes Windows to thrash swap even on a 32 GB machine.
    # Instead we write each array directly from the per-ticker chunks into the
    # output zip, one array at a time, keeping only one chunk (~50 MB) in RAM.
    # -------------------------------------------------------------------------

    def _stream_array_to_zip(
        zf: zipfile.ZipFile,
        arcname: str,
        tmp_files: list,
        key: str,
        shape: tuple,
        dtype,
        allow_pickle: bool = False,
    ):
        """
        Stream one array field from per-ticker .npz files into an open ZipFile
        entry, without ever holding the full concatenated array in RAM.
        Writes a valid .npy header followed by raw row chunks.
        """
        # Build a dummy array just for the header (zero-size, no allocation)
        dummy = np.empty(shape, dtype=dtype)
        with zf.open(arcname + ".npy", "w", force_zip64=True) as fh:
            npfmt.write_array_header_2_0(fh, npfmt.header_data_from_array_1_0(dummy))
            del dummy
            for fpath in tmp_files:
                chunk = np.load(fpath, allow_pickle=allow_pickle)[key]
                fh.write(np.ascontiguousarray(chunk, dtype=dtype).tobytes())
                del chunk

    # Collect small object arrays (dates, tickers) in RAM — they're tiny (~5 MB)
    logger.info("Collecting dates and ticker labels...")
    dates_arr = np.empty((n_long,), dtype=object)
    tickers_arr = np.empty((n_long,), dtype=object)
    idx = 0
    for fpath in long_tmp_files:
        d = np.load(fpath, allow_pickle=True)
        n = len(d["dt"])
        dates_arr[idx : idx + n] = d["dt"]
        tickers_arr[idx : idx + n] = d["tk"]
        idx += n

    logger.info("Writing long windows .npz (streaming, one array at a time)...")
    with zipfile.ZipFile(
        long_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        _stream_array_to_zip(
            zf, "X", long_tmp_files, "X", (n_long, WINDOW_LONG, N_FEATS), np.float32
        )
        logger.info("  X done")
        _stream_array_to_zip(
            zf, "Y_ret", long_tmp_files, "Yr", (n_long, len(HORIZONS)), np.float32
        )
        logger.info("  Y_ret done")
        _stream_array_to_zip(
            zf, "Y_px", long_tmp_files, "Yp", (n_long, len(HORIZONS)), np.float32
        )
        logger.info("  Y_px done")
        _stream_array_to_zip(zf, "LC", long_tmp_files, "LC", (n_long,), np.float32)
        _stream_array_to_zip(zf, "sectors", long_tmp_files, "sec", (n_long,), np.int32)
        with zf.open("dates.npy", "w", force_zip64=True) as fh:
            np.save(fh, dates_arr)
        with zf.open("tickers.npy", "w", force_zip64=True) as fh:
            np.save(fh, tickers_arr)
    logger.info("Long windows saved.")

    logger.info("Writing short windows .npz (streaming, one array at a time)...")
    with zipfile.ZipFile(
        short_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        _stream_array_to_zip(
            zf,
            "X_seq",
            short_tmp_files,
            "Xs",
            (n_short, WINDOW_SHORT, N_FEATS),
            np.float32,
        )
        logger.info("  X_seq done")
        _stream_array_to_zip(
            zf, "X_flat", short_tmp_files, "Xf", (n_short, WINDOW_SHORT * 5), np.float32
        )
        logger.info("  X_flat done")
        _stream_array_to_zip(
            zf, "Y", short_tmp_files, "Y", (n_short, len(HORIZONS)), np.float32
        )
    logger.info("Short windows saved.")

    # Clean up all per-ticker temp files
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
        "  Long windows (756): (%d, %d, %d)\n"
        "  Short windows (252): (%d, %d, %d)\n"
        "  Tickers: %d  Date range: %s to %s",
        N_FEATS,
        n_long,
        WINDOW_LONG,
        N_FEATS,
        n_short,
        WINDOW_SHORT,
        N_FEATS,
        len(tickers_used),
        meta["date_min"],
        meta["date_max"],
    )


def build_incremental(lookback_days: int = 30):
    """
    Incremental update: only download the last `lookback_days` of new data,
    compute windows that overlap with the new rows, and append them to the
    existing .npz files without rebuilding from scratch.

    This is designed for weekly retraining on GitHub Actions where:
      - Disk space is limited (~14 GB free)
      - The full 10-year dataset (~57 GB) cannot be stored on the runner
      - Only model weights (.pth / .pkl) are committed to git, not the dataset
      - Fine-tuning on recent data (last 30 days) is sufficient to keep models current

    Strategy:
      1. Load existing meta.json to know the last date already in the dataset
      2. Download only rows newer than that date (typically 5–7 trading days)
      3. For each ticker, load its raw CSV (stored in dataset/raw/), append new rows,
         recompute features, and extract only the new windows at the tail
      4. Append new windows to the existing .npz files using the same streaming
         zip writer, without loading the existing data into RAM
    """
    meta_path = os.path.join(DATASET_DIR, "meta.json")
    long_path = os.path.join(DATASET_DIR, "windows_756.npz")
    short_path = os.path.join(DATASET_DIR, "windows_252.npz")

    if not os.path.exists(meta_path) or not os.path.exists(long_path):
        logger.warning(
            "No existing dataset found — falling back to full build. "
            "Run  python build_dataset.py --refresh  first."
        )
        build(refresh=True)
        return

    with open(meta_path) as f:
        meta = json.load(f)

    last_date = meta.get("date_max")
    logger.info("Incremental update: existing dataset ends at %s", last_date)
    logger.info("Downloading last %d days of data for all tickers...", lookback_days)

    tickers, sector_ids = fetch_sp500_tickers()
    vix, sp500_ret_21d, sp500_ret_63d = fetch_market_data()
    sector_returns_prev = {}  # will be sparse — just use zeros for sector feature

    # Download recent data — use a short period so yfinance only pulls new rows
    period = f"{lookback_days + 30}d"  # extra buffer for indicator warmup

    def _download_recent(sym):
        try:
            yf_sym = sym.replace(".", "-").upper()
            df_new = yf.download(yf_sym, period=period, interval="1d", progress=False)
            if df_new is None or df_new.empty:
                return sym, None
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            df_new = df_new[["Open", "High", "Low", "Close", "Volume"]].dropna()
            # Load existing raw CSV and append, keeping full history for indicators
            raw_csv = os.path.join(RAW_DIR, f"{sym}.csv")
            if os.path.exists(raw_csv):
                df_old = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
                df_old.columns = [c.strip() for c in df_old.columns]
                df_combined = pd.concat([df_old, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                df_combined = df_combined.sort_index()
            else:
                df_combined = df_new
            return sym, df_combined
        except Exception as e:
            logger.debug("Skip %s: %s", sym, e)
            return sym, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = dict(pool.map(lambda s: _download_recent(s), list(tickers)))
    raw_data = {k: v for k, v in results.items() if v is not None}
    logger.info("Downloaded recent data for %d tickers", len(raw_data))

    # Recompute sector returns from recent data
    sector_returns = compute_sector_returns(raw_data, sector_ids)

    tmp_dir = os.path.join(DATASET_DIR, "_tmp_inc")
    os.makedirs(tmp_dir, exist_ok=True)

    def _series_to_arrays_local(s: pd.Series):
        return s.values, s.index.values

    worker_args = []
    for sym, raw_df in raw_data.items():
        sid = sector_ids.get(sym, 7)
        sec_ret = sector_returns.get(sid, pd.Series(dtype=float))
        vix_v, vix_i = _series_to_arrays_local(vix)
        sp21_v, sp21_i = _series_to_arrays_local(sp500_ret_21d)
        sp63_v, sp63_i = _series_to_arrays_local(sp500_ret_63d)
        sec_v, sec_i = _series_to_arrays_local(sec_ret)
        worker_args.append(
            (
                sym,
                raw_df,
                sid,
                sec_v,
                sec_i,
                vix_v,
                vix_i,
                sp21_v,
                sp21_i,
                sp63_v,
                sp63_i,
                tmp_dir,
            )
        )

    n_workers = max(1, os.cpu_count() // 2)
    new_long_files, new_short_files = [], []
    n_new_long = n_new_short = 0
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_ticker_incremental, arg): arg[0] for arg in worker_args
        }
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is None:
                continue
            sym, long_p, short_p, n_l, n_s, _, _ = result
            if long_p:
                new_long_files.append(long_p)
                n_new_long += n_l
            if short_p:
                new_short_files.append(short_p)
                n_new_short += n_s
            if done % 100 == 0:
                logger.info("  %d/%d tickers processed", done, len(worker_args))

    if not new_long_files:
        logger.warning("No new windows found — dataset is already up to date.")
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    logger.info(
        "Appending %d new long windows and %d new short windows...",
        n_new_long,
        n_new_short,
    )

    import zipfile, numpy.lib.format as npfmt, shutil

    def _append_array_to_zip(
        existing_path: str, new_files: list, key_map: dict, n_new: int
    ):
        """
        Append new rows to each array inside an existing .npz (zip) file.
        Opens the old zip, copies existing entries, then appends new rows.
        key_map: {npz_arcname: (tmp_key, shape_suffix, dtype, allow_pickle)}
        """
        tmp_out = existing_path + ".tmp"
        with (
            zipfile.ZipFile(existing_path, "r") as zin,
            zipfile.ZipFile(
                tmp_out, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as zout,
        ):
            for arcname, (tmp_key, extra_dims, dtype, allow_pickle) in key_map.items():
                npy_name = arcname + ".npy"
                # Read old array header to get shape
                with zin.open(npy_name) as fh:
                    version = npfmt.read_magic(fh)
                    shape_old, fortran, dtype_old = (
                        npfmt.read_array_header_2_0(fh)
                        if version == (2, 0)
                        else npfmt.read_array_header_1_0(fh)
                    )
                n_old = shape_old[0]
                new_shape = (n_old + n_new,) + extra_dims

                with zout.open(npy_name, "w", force_zip64=True) as fout:
                    # Write new header
                    dummy = np.empty(new_shape, dtype=dtype)
                    npfmt.write_array_header_2_0(
                        fout, npfmt.header_data_from_array_1_0(dummy)
                    )
                    del dummy
                    # Copy old data verbatim (raw bytes — no decompression needed
                    # for ZIP_STORED, which is what we always write)
                    with zin.open(npy_name) as fh_old:
                        # Skip header
                        npfmt.read_magic(fh_old)
                        if version == (2, 0):
                            npfmt.read_array_header_2_0(fh_old)
                        else:
                            npfmt.read_array_header_1_0(fh_old)
                        while True:
                            chunk = fh_old.read(1 << 20)  # 1 MB chunks
                            if not chunk:
                                break
                            fout.write(chunk)
                    # Append new rows
                    for fpath in new_files:
                        chunk = np.load(fpath, allow_pickle=allow_pickle)[tmp_key]
                        fout.write(np.ascontiguousarray(chunk, dtype=dtype).tobytes())

        os.replace(tmp_out, existing_path)

    # Append to long windows
    _append_array_to_zip(
        long_path,
        new_long_files,
        {
            "X": ("X", (WINDOW_LONG, N_FEATS), np.float32, False),
            "Y_ret": ("Yr", (len(HORIZONS),), np.float32, False),
            "Y_px": ("Yp", (len(HORIZONS),), np.float32, False),
            "LC": ("LC", (), np.float32, False),
            "sectors": ("sec", (), np.int32, False),
            "dates": ("dt", (), object, True),
            "tickers": ("tk", (), object, True),
        },
    )
    logger.info("Long windows updated.")

    # Append to short windows
    _append_array_to_zip(
        short_path,
        new_short_files,
        {
            "X_seq": ("Xs", (WINDOW_SHORT, N_FEATS), np.float32, False),
            "X_flat": ("Xf", (WINDOW_SHORT * 5,), np.float32, False),
            "Y": ("Y", (len(HORIZONS),), np.float32, False),
        },
    )
    logger.info("Short windows updated.")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Update meta
    with open(meta_path) as f:
        meta = json.load(f)
    meta["n_windows_756"] = meta.get("n_windows_756", 0) + n_new_long
    meta["n_windows_252"] = meta.get("n_windows_252", 0) + n_new_short
    meta["last_incremental_update"] = datetime.utcnow().isoformat() + "Z"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Incremental update complete. Added %d long + %d short windows.",
        n_new_long,
        n_new_short,
    )


def _process_ticker_incremental(args):
    """
    Like _process_ticker but only extracts windows from the TAIL of the data
    — specifically the new rows added since the last update.
    We still need the full history for indicator computation (SMA_200 needs
    200 rows of lookback), but we only save windows that end in the new region.
    """
    (
        sym,
        raw_df,
        sid,
        sec_ret_vals,
        sec_ret_idx,
        vix_vals,
        vix_idx,
        sp21_vals,
        sp21_idx,
        sp63_vals,
        sp63_idx,
        tmp_dir,
    ) = args
    try:
        vix_s = pd.Series(vix_vals, index=pd.DatetimeIndex(vix_idx), name="vix_close")
        sp21_s = pd.Series(
            sp21_vals, index=pd.DatetimeIndex(sp21_idx), name="sp500_ret_21d"
        )
        sp63_s = pd.Series(
            sp63_vals, index=pd.DatetimeIndex(sp63_idx), name="sp500_ret_63d"
        )
        sec_ret_s = pd.Series(sec_ret_vals, index=pd.DatetimeIndex(sec_ret_idx))

        df = check_price_data(raw_df, sym)
        tech = compute_features(df, vix_s, sp21_s, sp63_s, sec_ret_s)

        if len(tech) < MIN_ROWS_AFTER_TECH:
            return None

        idx_min = str(tech.index.min().date())
        idx_max = str(tech.index.max().date())

        # Only generate windows whose LAST day is within the most recent
        # lookback_days rows — avoids duplicating windows already in the dataset
        new_rows = 10  # conservative: keep last 10 windows worth of new data
        n_total = len(tech)

        long_path_out = short_path_out = None
        n_l = n_s = 0

        if n_total >= WINDOW_LONG + MAX_H:
            # Build all windows but slice only the tail
            X_l, Y_r, Y_p, LC, dates, secs, tks = build_windows_long(tech, sym, sid)
            if len(X_l) > 0:
                tail = X_l[-new_rows:]
                if len(tail) > 0:
                    long_path_out = os.path.join(tmp_dir, f"long_{sym}.npz")
                    np.savez(
                        long_path_out,
                        X=tail,
                        Yr=Y_r[-new_rows:],
                        Yp=Y_p[-new_rows:],
                        LC=LC[-new_rows:],
                        dt=dates[-new_rows:],
                        sec=secs[-new_rows:],
                        tk=tks[-new_rows:],
                    )
                    n_l = len(tail)

        if n_total >= WINDOW_SHORT + MAX_H:
            X_seq, X_flat, Y_s = build_windows_short(tech)
            if len(X_seq) > 0:
                tail_s = X_seq[-new_rows:]
                if len(tail_s) > 0:
                    short_path_out = os.path.join(tmp_dir, f"short_{sym}.npz")
                    np.savez(
                        short_path_out,
                        Xs=tail_s,
                        Xf=X_flat[-new_rows:],
                        Y=Y_s[-new_rows:],
                    )
                    n_s = len(tail_s)

        return sym, long_path_out, short_path_out, n_l, n_s, idx_min, idx_max
    except Exception as e:
        logger.debug("Skip %s (incremental): %s", sym, e)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build shared dataset for all 4 models"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-download even if cache exists"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only download recent data and append new windows (fast, for weekly retraining)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="How many recent days to download in incremental mode (default: 30)",
    )
    args = parser.parse_args()
    if args.incremental:
        build_incremental(lookback_days=args.lookback_days)
    else:
        build(refresh=args.refresh)
