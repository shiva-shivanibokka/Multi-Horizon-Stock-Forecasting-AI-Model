import numpy as np
import pandas as pd

from mhf.eval.backtest import build_from_signal, _stats


def test_stats_constant_series():
    m = pd.Series([0.01] * 24, index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    s = _stats(m)
    assert abs(s["ann_return"] - 0.12) < 1e-9
    assert s["ann_vol"] == 0.0
    assert s["sharpe"] is None          # zero volatility
    assert s["max_drawdown"] == 0.0     # never drops
    assert s["hit_rate"] == 1.0


def test_backtest_rewards_a_good_signal():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2018-01-01", periods=36, freq="MS"):
        realized = rng.normal(0.0, 0.05, size=100)
        pred = realized + rng.normal(0.0, 0.01, size=100)  # strongly rank-predictive
        for i in range(100):
            rows.append({"end_date": d, "ticker": f"T{i}", "pred": pred[i], "realized": realized[i]})
    out = build_from_signal(pd.DataFrame(rows), decile=0.1)

    assert out["n_months"] == 36
    assert len(out["equity"]["long_short"]) == 36
    assert out["stats"]["long_short"]["sharpe"] > 1.0        # a real edge shows up
    # long-short should beat the equal-weight benchmark's mean return
    assert out["stats"]["long_short"]["ann_return"] > out["stats"]["benchmark"]["ann_return"]
    # transaction costs can only reduce the Sharpe
    cs = out["cost_sensitivity"]
    assert cs[0]["sharpe"] >= cs[-1]["sharpe"]


def test_backtest_neutral_on_noise():
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.date_range("2018-01-01", periods=36, freq="MS"):
        realized = rng.normal(0.0, 0.05, size=100)
        pred = rng.normal(0.0, 0.05, size=100)  # unrelated to realized
        for i in range(100):
            rows.append({"end_date": d, "ticker": f"T{i}", "pred": pred[i], "realized": realized[i]})
    out = build_from_signal(pd.DataFrame(rows), decile=0.1)
    assert abs(out["stats"]["long_short"]["ann_return"]) < 0.15  # no systematic edge from noise
