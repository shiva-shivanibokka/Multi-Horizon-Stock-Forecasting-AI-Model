import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES
from mhf.models.gbm import GBMQuantile


def _panel(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)
    # y depends on feature 0 so the model has real signal to learn
    signal = 0.03 * X[:, 0]
    for i, c in enumerate(Y_COLS):
        df[c] = signal + rng.normal(0, 0.01, size=n)
    df["ticker"] = "AAA"
    df["end_date"] = pd.bdate_range("2017-01-01", periods=n)
    return df


def test_gbm_shape_and_quantile_monotonic():
    train = _panel()
    m = GBMQuantile(n_estimators=50).fit(train)
    out = m.predict_quantiles(train.iloc[:20])
    assert out.shape == (20, len(Y_COLS), len(QUANTILES))
    assert (np.diff(out, axis=2) >= -1e-9).all()  # sorted p10<=p50<=p90


def test_gbm_learns_direction():
    train = _panel(seed=3)
    m = GBMQuantile(n_estimators=100).fit(train)
    hi = train[train[FEATURES[0]] > 1.0].iloc[:30]
    lo = train[train[FEATURES[0]] < -1.0].iloc[:30]
    p50_hi = m.predict_quantiles(hi)[:, 0, 1].mean()
    p50_lo = m.predict_quantiles(lo)[:, 0, 1].mean()
    assert p50_hi > p50_lo  # higher feature-0 -> higher predicted median
