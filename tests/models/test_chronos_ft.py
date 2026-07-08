import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.models.chronos_ft import forecast_to_return_quantiles, to_tsdf


def test_to_tsdf_long_format():
    import pytest
    pytest.importorskip("autogluon.timeseries")

    s1 = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    s2 = pd.Series([4.0, 5.0], index=pd.bdate_range("2020-01-01", periods=2))
    tsdf = to_tsdf({"AAA": s1, "BBB": s2})
    df = tsdf.reset_index() if hasattr(tsdf, "reset_index") else tsdf
    assert set(df["item_id"]) == {"AAA", "BBB"}
    assert "target" in df.columns and "timestamp" in df.columns
    assert len(df) == 5


def test_forecast_to_return_quantiles_math():
    # Build a fake 126-step forecast where log-price rises linearly by 0.001/step
    steps = 126
    ts = pd.bdate_range("2021-01-01", periods=steps)
    anchor = 4.0  # log price at anchor
    logp = anchor + 0.001 * np.arange(1, steps + 1)
    idx = pd.MultiIndex.from_product([["AAA"], ts], names=["item_id", "timestamp"])
    pred = pd.DataFrame(
        {"0.1": logp - 0.01, "0.5": logp, "0.9": logp + 0.01}, index=idx
    )
    out = forecast_to_return_quantiles(pred, anchor_log_price=anchor)
    assert out.shape == (3, len(QUANTILES))
    # median return at 21d = exp(0.001*21) - 1
    assert abs(out[1, 1] - (np.exp(0.001 * 21) - 1)) < 1e-6
    assert (np.diff(out, axis=1) >= -1e-9).all()  # monotone quantiles
