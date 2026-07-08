import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS
from mhf.models.baselines import HistoricalQuantile, RandomWalk, garch_volatility


def _panel(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0.01 * i, 0.05, size=n) for i, c in enumerate(Y_COLS)})
    df["end_date"] = pd.bdate_range("2018-01-01", periods=n)
    return df


def test_random_walk_zero_median_and_monotonic():
    train = _panel()
    m = RandomWalk().fit(train)
    out = m.predict_quantiles(train.iloc[:5])
    assert out.shape == (5, len(Y_COLS), len(QUANTILES))
    assert (np.diff(out, axis=2) >= 0).all()
    np.testing.assert_allclose(out[:, :, 1], 0.0, atol=1e-9)  # p50 == 0 (random walk)


def test_historical_quantile_shape_and_monotonic():
    train = _panel()
    m = HistoricalQuantile().fit(train)
    out = m.predict_quantiles(train.iloc[:10])
    assert out.shape == (10, len(Y_COLS), len(QUANTILES))
    # p10 <= p50 <= p90 for every row/horizon
    assert (np.diff(out, axis=2) >= 0).all()


def test_historical_quantile_matches_empirical():
    train = _panel(seed=1)
    m = HistoricalQuantile().fit(train)
    out = m.predict_quantiles(train.iloc[:1])
    expected = np.quantile(train[Y_COLS[0]].to_numpy(), QUANTILES)
    np.testing.assert_allclose(out[0, 0, :], expected, rtol=1e-6)


def test_garch_volatility_positive():
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0, 0.02, size=800))
    vol = garch_volatility(rets, horizon=21)
    assert vol > 0 and np.isfinite(vol)
