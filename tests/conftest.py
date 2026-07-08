import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """400 trading days of synthetic, strictly-positive OHLCV for one ticker.

    Needs to be long enough that after compute_features()'s rolling-252
    warmup dropna (~251 rows consumed), enough rows remain for window+horizon
    tests (e.g. window=60, max horizon=21 -> needs >= 81 post-dropna rows).
    """
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, size=400))
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0, 2, size=400)
    low = close - rng.uniform(0, 2, size=400)
    open_ = close + rng.uniform(-1, 1, size=400)
    vol = rng.integers(1_000_000, 5_000_000, size=400).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


@pytest.fixture
def market() -> pd.DataFrame:
    idx = pd.bdate_range("2019-06-01", periods=520)  # starts before ohlcv, wider range
    rng = np.random.default_rng(1)
    vix = 15 + np.cumsum(rng.normal(0, 0.3, size=520))
    sp = 3000 + np.cumsum(rng.normal(0, 5, size=520))
    sp = pd.Series(sp, index=idx)
    return pd.DataFrame(
        {"vix_close": np.clip(vix, 9, None),
         "sp500_ret_21d": sp.pct_change(21).values,
         "sp500_ret_63d": sp.pct_change(63).values},
        index=idx,
    )
