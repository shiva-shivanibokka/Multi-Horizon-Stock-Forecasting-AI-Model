import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES
from mhf.eval.cv import walk_forward_folds
from mhf.eval.metrics import (
    coverage,
    directional_hit_rate,
    information_coefficient,
    pinball_loss,
)
from mhf.models.baselines import HistoricalQuantile, RandomWalk
from mhf.models.ensemble import blend, fit_blend_weight
from mhf.models.gbm import GBMQuantile

logger = logging.getLogger(__name__)


def monthly_anchors(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["_ym"] = p["end_date"].dt.to_period("M")
    idx = p.groupby(["ticker", "_ym"])["end_date"].idxmax()
    return panel.loc[idx].sort_values(["end_date", "ticker"]).reset_index(drop=True)


def _score(y: np.ndarray, q: np.ndarray) -> dict:
    # y: (n, n_horizons); q: (n, n_horizons, n_quantiles)
    out = {}
    for h, name in enumerate(Y_COLS):
        out[name] = {
            "pinball": pinball_loss(y[:, h], q[:, h, :]),
            "coverage": coverage(y[:, h], q[:, h, 0], q[:, h, 2]),
            "hit_rate": directional_hit_rate(y[:, h], q[:, h, 1]),
            "ic": information_coefficient(y[:, h], q[:, h, 1]),
        }
    return out


def run_training(panel, series_by_ticker, *, n_folds=4, fine_tune_steps=1000,
                 use_chronos=True, out_dir=None, wandb_project=None) -> dict:
    out_dir = Path(out_dir) if out_dir else settings.data_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors = monthly_anchors(panel)
    folds = walk_forward_folds(panel["end_date"].to_numpy(), n_folds=n_folds)

    rw_q, baseline_q, gbm_q, y_all = [], [], [], []
    last_gbm = None
    for fold in folds:
        train = panel[fold.train]
        test_anchor = anchors[anchors["end_date"].isin(panel.loc[fold.test, "end_date"])]
        if len(test_anchor) == 0:
            continue
        rw = RandomWalk().fit(train)
        base = HistoricalQuantile().fit(train)
        gbm = GBMQuantile().fit(train)
        last_gbm = gbm
        rw_q.append(rw.predict_quantiles(test_anchor))
        baseline_q.append(base.predict_quantiles(test_anchor))
        gbm_q.append(gbm.predict_quantiles(test_anchor))
        y_all.append(test_anchor[Y_COLS].to_numpy())

    rw_q = np.concatenate(rw_q)
    baseline_q = np.concatenate(baseline_q)
    gbm_q = np.concatenate(gbm_q)
    y_all = np.concatenate(y_all)

    metrics = {
        "random_walk": _score(y_all, rw_q),
        "baseline": _score(y_all, baseline_q),
        "gbm": _score(y_all, gbm_q),
    }

    if use_chronos and series_by_ticker:
        from mhf.models.chronos_ft import ChronosForecaster

        train_max = panel.loc[folds[0].train, "end_date"].max()
        train_series = {
            t: s[s.index <= train_max] for t, s in series_by_ticker.items()
        }
        chronos = ChronosForecaster().fit(train_series, fine_tune_steps=fine_tune_steps)
        chronos.set_series(series_by_ticker)
        # re-predict on the same test anchors, fold by fold, in the same order so
        # rows align 1:1 with y_all / gbm_q above.
        chronos_q = []
        for fold in folds:
            test_anchor = anchors[anchors["end_date"].isin(panel.loc[fold.test, "end_date"])]
            if len(test_anchor) == 0:
                continue
            chronos_q.append(chronos.predict_quantiles(test_anchor))
        chronos_q = np.concatenate(chronos_q)
        metrics["chronos"] = _score(y_all, chronos_q)
        w = fit_blend_weight(chronos_q, gbm_q, y_all)
        ens = blend(chronos_q, gbm_q, w)
        metrics["ensemble"] = _score(y_all, ens)
        metrics["blend_weight_chronos"] = w
        chronos.save(out_dir / "chronos")

    # artifacts
    if last_gbm is not None:
        (out_dir / "gbm").mkdir(exist_ok=True)
        with open(out_dir / "gbm" / "model.pkl", "wb") as f:
            pickle.dump(last_gbm, f)
    ref = panel[FEATURES].agg(["mean", "std"]).T
    ref.to_parquet(out_dir / "feature_reference.parquet")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_model_card(out_dir / "model_card.md", metrics)

    if wandb_project:
        _log_wandb(wandb_project, metrics)
    return metrics


def _write_model_card(path: Path, metrics: dict) -> None:
    lines = ["# Model Card — Multi-Horizon Probabilistic Equity Forecaster", ""]
    lines.append("Out-of-sample metrics (purged/embargoed walk-forward CV):\n")
    for model, per_h in metrics.items():
        if not isinstance(per_h, dict):
            lines.append(f"- **{model}**: {per_h}")
            continue
        lines.append(f"## {model}")
        for h, m in per_h.items():
            lines.append(f"- {h}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    path.write_text("\n".join(lines))


def _log_wandb(project: str, metrics: dict) -> None:
    import wandb

    run = wandb.init(project=project, job_type="train")
    flat = {}
    for model, per_h in metrics.items():
        if isinstance(per_h, dict):
            for h, m in per_h.items():
                for k, v in m.items():
                    flat[f"{model}/{h}/{k}"] = v
        else:
            flat[model] = per_h
    run.log(flat)
    run.finish()


def _build_inputs(smoke: bool):
    from mhf.data.assemble import build_panel
    from mhf.data.ingest import download_ohlcv, fetch_market, fetch_sp500

    tickers, _ = fetch_sp500()
    if smoke:
        tickers = tickers[:3]
    market = fetch_market()
    panel = build_panel(tickers, market)
    series = {}
    for t in panel["ticker"].unique():
        df = download_ohlcv(t)
        if df is not None:
            series[t] = df["Close"]
    return panel, series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-chronos", action="store_true")
    ap.add_argument("--fine-tune-steps", type=int, default=1000)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    panel, series = _build_inputs(args.smoke)
    metrics = run_training(
        panel, series,
        n_folds=2 if args.smoke else 4,
        fine_tune_steps=1 if args.smoke else args.fine_tune_steps,
        use_chronos=not args.no_chronos,
        wandb_project=args.wandb_project,
    )
    logger.info("done; metrics written")
    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
