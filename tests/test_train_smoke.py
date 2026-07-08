import numpy as np
import pandas as pd

from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES
from mhf.train import monthly_anchors, run_training


def _panel(n_per=900, tickers=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for t in tickers:
        X = rng.normal(size=(n_per, len(FEATURES)))
        df = pd.DataFrame(X, columns=FEATURES)
        sig = 0.02 * X[:, 0]
        for c in Y_COLS:
            df[c] = sig + rng.normal(0, 0.02, size=n_per)
        df["ticker"] = t
        df["end_date"] = pd.bdate_range("2015-01-01", periods=n_per)
        df["base_close"] = 100.0
        frames.append(df)
    cols = ["ticker", "end_date", *FEATURES, *Y_COLS, "base_close"]
    return pd.concat(frames, ignore_index=True)[cols]


def test_monthly_anchors_thins_to_month_ends():
    panel = _panel(n_per=200, tickers=("AAA",))
    anchors = monthly_anchors(panel)
    assert len(anchors) < len(panel)
    assert len(anchors) >= 6  # ~9 months of business days


def test_run_training_smoke_no_chronos(tmp_path):
    panel = _panel()
    metrics = run_training(
        panel, series_by_ticker={}, n_folds=2, use_chronos=False, out_dir=tmp_path
    )
    assert "gbm" in metrics and "baseline" in metrics
    for model in ("gbm", "baseline"):
        for h in Y_COLS:
            assert np.isfinite(metrics[model][h]["pinball"])
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "feature_reference.parquet").exists()
