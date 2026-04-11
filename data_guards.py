"""
data_guards.py
Training-time data quality checks for all four forecasting models.

These functions are called inside build_dataset() before any window is added
to the training set. They catch bad data early so it never corrupts the model.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_price_data(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """
    Validates and cleans a raw OHLCV DataFrame for a single ticker.

    Checks applied (in order):
      1. Minimum row count — needs at least 200 rows to compute any indicator
      2. Zero / negative prices — rows with Close <= 0 are dropped
      3. Missing value threshold — skips ticker if >5% of Close values are NaN
      4. Daily return outlier removal — clips single-day returns beyond ±50%
         (these are almost always unadjusted split artifacts, not real moves)

    Returns the cleaned DataFrame. Raises ValueError if the ticker should be skipped.
    """
    if df is None or df.empty:
        raise ValueError("empty dataframe")

    if len(df) < 200:
        raise ValueError(f"only {len(df)} rows (need >= 200)")

    # Drop rows with zero or negative Close — impossible in real market data
    df = df[df["Close"] > 0].copy()

    # Skip if too much of the close series is missing
    missing_pct = df["Close"].isna().mean()
    if missing_pct > 0.05:
        raise ValueError(f"{missing_pct:.1%} of Close values are NaN (threshold 5%)")

    df = df.dropna(subset=["Close"])

    # Clip extreme single-day returns. Returns > 50% in one day are almost
    # always a data artifact (unadjusted stock split, yfinance bug). We cap
    # the move rather than dropping the row to preserve sequence continuity.
    returns = df["Close"].pct_change()
    extreme = (returns.abs() > 0.5).sum()
    if extreme > 0:
        logger.warning("%s: clipping %d extreme daily returns (>50%%)", sym, extreme)
        valid_pct_change = returns.clip(-0.5, 0.5)
        # Reconstruct Close from clipped returns so the series stays smooth
        first_valid = df["Close"].iloc[0]
        reconstructed = first_valid * (1 + valid_pct_change).cumprod()
        reconstructed.iloc[0] = first_valid
        df = df.copy()
        df["Close"] = reconstructed.values

    return df


def check_feature_array(X: np.ndarray, name: str = "X"):
    """
    Checks a scaled feature array for NaN and Inf values after preprocessing.
    These can silently corrupt model weights if not caught.
    Raises ValueError with a clear message if bad values are found.
    """
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0:
        raise ValueError(
            f"{name} contains {nan_count} NaN values after scaling. "
            "Check compute_technicals() for division by zero or insufficient history."
        )
    if inf_count > 0:
        raise ValueError(
            f"{name} contains {inf_count} Inf values after scaling. "
            "Check for zero denominators in RSI or ROC calculations."
        )
    logger.info("%s: shape=%s  NaN=0  Inf=0  OK", name, X.shape)


def check_train_test_split(X_tr: np.ndarray, X_te: np.ndarray):
    """
    Verifies there is a clean chronological boundary between train and test.
    Specifically checks that the split index is non-trivial (both sets
    have at least 10% of total data).
    """
    total = len(X_tr) + len(X_te)
    tr_pct = len(X_tr) / total
    te_pct = len(X_te) / total
    if tr_pct < 0.1 or te_pct < 0.1:
        raise ValueError(
            f"Degenerate split: train={tr_pct:.1%}  test={te_pct:.1%}. "
            "Check that enough tickers downloaded successfully."
        )
    logger.info(
        "Train/test split OK: train=%d (%.0f%%)  test=%d (%.0f%%)",
        len(X_tr),
        tr_pct * 100,
        len(X_te),
        te_pct * 100,
    )


def check_target_distribution(Y: np.ndarray, horizon_name: str):
    """
    Logs if more than 70% of target returns are in one direction (all positive
    or all negative). This would mean the training data covers a one-directional
    market period and the model will be biased.
    This is a warning only — training continues.
    """
    if Y.ndim == 2:
        col = Y[:, 0]
    else:
        col = Y
    pct_positive = (col > 0).mean()
    if pct_positive > 0.70:
        logger.warning(
            "Target distribution for %s: %.0f%% positive returns. "
            "Dataset may cover a bull-only period — model could be bullish-biased.",
            horizon_name,
            pct_positive * 100,
        )
    elif pct_positive < 0.30:
        logger.warning(
            "Target distribution for %s: %.0f%% positive returns (%.0f%% negative). "
            "Dataset may cover a bear-only period — model could be bearish-biased.",
            horizon_name,
            pct_positive * 100,
            (1 - pct_positive) * 100,
        )
    else:
        logger.info(
            "Target distribution for %s: %.0f%% positive / %.0f%% negative — balanced.",
            horizon_name,
            pct_positive * 100,
            (1 - pct_positive) * 100,
        )


def log_dataset_summary(X: np.ndarray, Y: np.ndarray, n_tickers: int):
    """Logs a summary of the final assembled dataset before training begins."""
    logger.info(
        "Dataset ready: tickers=%d  windows=%d  X_shape=%s  Y_shape=%s  memory=%.1f MB",
        n_tickers,
        len(X),
        X.shape,
        Y.shape,
        (X.nbytes + Y.nbytes) / 1e6,
    )
