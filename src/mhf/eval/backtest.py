"""Economic backtest of the model's cross-sectional signal.

Turns the statistical forecast into a *strategy*: each month, rank the universe by
predicted 1-month return, go long the top decile (and short the bottom for a
market-neutral version), hold one month, repeat — using only out-of-sample,
walk-forward predictions (same purged/embargoed CV as training, no lookahead).
Reports the equity curve, Sharpe, CAGR, max drawdown, turnover, and how the edge
survives transaction costs.

Uses the GBM (the feature-driven cross-sectional model) for the signal — it needs
no Chronos, so the whole backtest runs in a couple of minutes.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.data.assemble import Y_COLS
from mhf.eval.cv import walk_forward_folds
from mhf.models.gbm import GBMQuantile
from mhf.train import monthly_anchors

logger = logging.getLogger(__name__)


def oos_signal(panel: pd.DataFrame, n_folds: int = 4, horizon: str = "y_1m") -> pd.DataFrame:
    """Per-(month, ticker) out-of-sample: predicted median return + realized return."""
    h = Y_COLS.index(horizon)
    anchors = monthly_anchors(panel)
    folds = walk_forward_folds(panel["end_date"].to_numpy(), n_folds=n_folds)
    frames = []
    for fold in folds:
        train = panel[fold.train]
        test_anchor = anchors[anchors["end_date"].isin(panel.loc[fold.test, "end_date"])]
        if len(test_anchor) == 0:
            continue
        q = GBMQuantile().fit(train).predict_quantiles(test_anchor)
        frames.append(pd.DataFrame({
            "end_date": pd.to_datetime(test_anchor["end_date"].to_numpy()),
            "ticker": test_anchor["ticker"].to_numpy(),
            "pred": q[:, h, 1],                              # predicted median
            "realized": test_anchor[horizon].to_numpy(),     # realized 1m return
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stats(monthly: pd.Series) -> dict:
    m = monthly.dropna()
    if len(m) < 2:
        return {k: None for k in ("ann_return", "ann_vol", "sharpe", "cagr", "max_drawdown", "hit_rate")}
    eq = (1 + m).cumprod()
    ann_ret = float(m.mean() * 12)
    ann_vol = float(m.std(ddof=1) * np.sqrt(12))
    n_years = len(m) / 12.0
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 and eq.iloc[-1] > 0 else None
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(ann_ret / ann_vol, 3) if ann_vol > 0 else None,
        "cagr": round(cagr, 4) if cagr is not None else None,
        "max_drawdown": round(dd, 4),
        "hit_rate": round(float((m > 0).mean()), 3),
    }


def run_backtest(panel: pd.DataFrame, n_folds: int = 4, horizon: str = "y_1m",
                 decile: float = 0.1, min_names: int = 30) -> dict:
    sig = oos_signal(panel, n_folds, horizon)
    if sig.empty:
        raise RuntimeError("no out-of-sample signal produced")
    return build_from_signal(sig, horizon=horizon, decile=decile, min_names=min_names)


def build_from_signal(sig: pd.DataFrame, horizon: str = "y_1m",
                      decile: float = 0.1, min_names: int = 30) -> dict:
    """Pure portfolio construction from a [end_date, ticker, pred, realized] table."""
    dates, ls, lo, bench, turnover = [], [], [], [], []
    prev_long: set = set()
    for d, g in sig.groupby("end_date", sort=True):
        if len(g) < min_names:
            continue
        g = g.sort_values("pred")
        k = max(1, int(round(len(g) * decile)))
        longs, shorts = g.tail(k), g.head(k)
        long_set = set(longs["ticker"])
        turn = 1.0 if not prev_long else len(long_set - prev_long) / len(long_set)
        prev_long = long_set
        dates.append(pd.Timestamp(d))
        lo.append(float(longs["realized"].mean()))
        ls.append(float(longs["realized"].mean() - shorts["realized"].mean()))
        bench.append(float(g["realized"].mean()))
        turnover.append(turn)

    idx = pd.DatetimeIndex(dates)
    ls_s, lo_s, b_s = pd.Series(ls, index=idx), pd.Series(lo, index=idx), pd.Series(bench, index=idx)
    to = np.array(turnover)
    avg_turnover = float(to.mean()) if len(to) else 0.0

    # Cost model: `bps` per side; monthly drag = 2 x turnover x cost (buy + sell of the
    # names rotated each month). Applied to the long-short book.
    cost_rows = []
    for bps in (0, 5, 10, 20, 30):
        net = ls_s - to * 2 * (bps / 1e4)
        s = _stats(net)
        cost_rows.append({"bps": bps, "sharpe": s["sharpe"], "cagr": s["cagr"], "ann_return": s["ann_return"]})

    return {
        "horizon": horizon.replace("y_", ""),
        "signal": "GBM cross-sectional (out-of-sample, walk-forward)",
        "rebalance": "monthly",
        "decile": decile,
        "n_months": len(idx),
        "period": [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")] if len(idx) else None,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "equity": {
            "long_short": [round(v, 4) for v in (1 + ls_s).cumprod().tolist()],
            "long_only": [round(v, 4) for v in (1 + lo_s).cumprod().tolist()],
            "benchmark": [round(v, 4) for v in (1 + b_s).cumprod().tolist()],
        },
        "stats": {
            "long_short": {**_stats(ls_s), "avg_turnover": round(avg_turnover, 3)},
            "long_only": {**_stats(lo_s), "avg_turnover": round(avg_turnover, 3)},
            "benchmark": _stats(b_s),
        },
        "cost_sensitivity": cost_rows,
    }


def _build_panel() -> pd.DataFrame:
    from mhf.data.ingest import fetch_market, fetch_sp500
    tickers = sorted(p.stem for p in settings.raw_dir.glob("*.parquet"))
    if not tickers:
        tickers, _ = fetch_sp500()
    from mhf.data.assemble import build_panel
    return build_panel(tickers, fetch_market())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/data")
    ap.add_argument("--n-folds", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    panel = _build_panel()
    result = run_backtest(panel, n_folds=args.n_folds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    logger.info("backtest: %d months, long-short Sharpe %s -> wrote %s/backtest.json",
                result["n_months"], result["stats"]["long_short"]["sharpe"], out)


if __name__ == "__main__":
    main()
