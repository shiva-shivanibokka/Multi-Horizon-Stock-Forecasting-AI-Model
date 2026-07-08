import numpy as np
import pandas as pd
import pytest

from mhf.data import build


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """700 trading days of synthetic OHLCV, local to this test module.

    build_ticker() requires len(feats) >= settings.window_short (252) +
    settings.max_horizon (126) = 378 rows AFTER compute_features's
    rolling-252 warmup (~251 rows consumed by dropna). The shared 400-row
    `ohlcv` fixture in tests/conftest.py only survives that warmup down to
    ~149 rows -- nowhere near enough for a non-empty WindowSet at
    window_short=252. Rather than bump the shared fixture (used by other
    tasks' tests) or weaken this test, this fixture shadows the root one
    for tests in this module only.
    """
    idx = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, size=700))
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0, 2, size=700)
    low = close - rng.uniform(0, 2, size=700)
    open_ = close + rng.uniform(-1, 1, size=700)
    vol = rng.integers(1_000_000, 5_000_000, size=700).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


@pytest.fixture
def market() -> pd.DataFrame:
    """Wider/earlier-starting market series so it fully covers the local `ohlcv` range."""
    idx = pd.bdate_range("2019-06-01", periods=820)
    rng = np.random.default_rng(1)
    vix = 15 + np.cumsum(rng.normal(0, 0.3, size=820))
    sp = 3000 + np.cumsum(rng.normal(0, 5, size=820))
    sp = pd.Series(sp, index=idx)
    return pd.DataFrame(
        {
            "vix_close": np.clip(vix, 9, None),
            "sp500_ret_21d": sp.pct_change(21).values,
            "sp500_ret_63d": sp.pct_change(63).values,
        },
        index=idx,
    )


def test_build_ticker_happy_path(monkeypatch, ohlcv, market):
    monkeypatch.setattr(build, "download_ohlcv", lambda t, refresh=False: ohlcv)
    ws = build.build_ticker("AAPL", market)
    assert ws is not None
    assert ws.X.shape[0] > 0


def test_build_ticker_skips_bad_ticker(monkeypatch, market):
    monkeypatch.setattr(build, "download_ohlcv", lambda t, refresh=False: None)
    assert build.build_ticker("BAD", market) is None
