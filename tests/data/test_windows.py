import numpy as np

from mhf.data.features import FEATURES
from mhf.data.windows import build_windows


def _feats(ohlcv, market):
    from mhf.data.features import compute_features
    return compute_features(ohlcv, market)


def test_window_shapes_and_dates(ohlcv, market):
    feats = _feats(ohlcv, market)
    ws = build_windows(feats, window=60, horizons={"1w": 5, "1m": 21})
    n = ws.X.shape[0]
    assert ws.X.shape == (n, 60, len(FEATURES))
    assert ws.y_ret.shape == (n, 2)
    assert ws.end_dates.shape == (n,)
    # end date of sample i is the last input row of that window
    assert ws.end_dates[0] == np.datetime64(feats.index[59])
    # no target reaches past the data
    assert n == len(feats) - 60 - 21 + 1


def test_target_is_forward_return(ohlcv, market):
    feats = _feats(ohlcv, market)
    ws = build_windows(feats, window=60, horizons={"1w": 5})
    close = feats["Close"].values
    expected0 = close[59 + 5] / close[59] - 1
    # rtol=1e-5: y_ret is float32 (~1e-7 precision)
    assert np.isclose(ws.y_ret[0, 0], expected0, rtol=1e-5)
