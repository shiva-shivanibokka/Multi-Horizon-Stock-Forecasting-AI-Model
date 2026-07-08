import numpy as np
import pytest

from mhf.data.quality import DataQualityError, clean_ohlcv


def test_rejects_too_few_rows(ohlcv):
    with pytest.raises(DataQualityError):
        clean_ohlcv(ohlcv.iloc[:50], "X", min_rows=200)


def test_drops_nonpositive_close(ohlcv):
    df = ohlcv.copy()
    df.iloc[10, df.columns.get_loc("Close")] = -1.0
    out = clean_ohlcv(df, "X")
    assert (out["Close"] > 0).all()


def test_clips_extreme_daily_move(ohlcv):
    df = ohlcv.copy()
    df.iloc[100, df.columns.get_loc("Close")] = df["Close"].iloc[99] * 3  # +200% spike
    out = clean_ohlcv(df, "X")
    assert out["Close"].pct_change().abs().max() <= 0.5 + 1e-9
    assert (out["Close"] > 0).all()
    assert not np.isnan(out["Close"]).any()
