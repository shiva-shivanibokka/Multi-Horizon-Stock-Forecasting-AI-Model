import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """300 trading days of synthetic, strictly-positive OHLCV for one ticker."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, size=300))
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0, 2, size=300)
    low = close - rng.uniform(0, 2, size=300)
    open_ = close + rng.uniform(-1, 1, size=300)
    vol = rng.integers(1_000_000, 5_000_000, size=300).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )
