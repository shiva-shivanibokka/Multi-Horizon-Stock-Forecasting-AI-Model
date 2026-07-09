import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.serve.export import _forecast_records, _weights_and_deltas


def test_weights_and_deltas_ordered_by_horizon():
    names = list(settings.horizons)  # e.g. ["1m","3m","6m"]
    metrics = {
        "blend_weight_chronos": {names[0]: 0.8, names[1]: 1.0, names[2]: 1.0},
        "conformal_delta": {names[0]: 0.001, names[1]: -0.01, names[2]: 0.06},
    }
    w, delta = _weights_and_deltas(metrics)
    assert list(w) == [0.8, 1.0, 1.0]
    np.testing.assert_allclose(delta, [0.001, -0.01, 0.06])


def test_weights_and_deltas_defaults_when_missing():
    # a GBM-only / pre-conformal run has neither key -> pure Chronos weight, no shift
    w, delta = _weights_and_deltas({})
    assert (w == 1.0).all() and (delta == 0.0).all()


def test_forecast_records_shape_and_price_fan_inputs():
    names = list(settings.horizons)
    latest = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "end_date": pd.to_datetime(["2025-12-26", "2025-12-26"]),
        "base_close": [100.0, 50.0],
    })
    ens_q = np.zeros((2, len(names), 3))
    ens_q[0, 0] = [-0.05, 0.02, 0.09]  # AAA, first horizon
    recs = _forecast_records(latest, ens_q, {"AAA": "Tech"}, {"AAA": "Alpha Inc"})
    assert [r["ticker"] for r in recs] == ["AAA", "BBB"]  # sorted
    aaa = recs[0]
    assert aaa["sector"] == "Tech" and recs[1]["sector"] == "Unknown"
    assert aaa["name"] == "Alpha Inc" and recs[1]["name"] == "BBB"  # falls back to ticker
    assert aaa["anchor_price"] == 100.0 and aaa["anchor_date"] == "2025-12-26"
    assert set(aaa["horizons"]) == set(names)
    h0 = aaa["horizons"][names[0]]
    assert h0["q10"] == -0.05 and h0["q50"] == 0.02 and h0["q90"] == 0.09
