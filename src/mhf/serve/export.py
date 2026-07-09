"""Export static JSON the frontend consumes: the final ENSEMBLE forecast for every
ticker at its latest anchor, recent price history, plus the training metrics.

No server runs at request time — this batch step (re-run after each training run)
writes three files the React app fetches directly. It reuses the exact inference
path from train.py: GBM + zero-shot Chronos, per-horizon blend, conformal band."""
import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.data.ingest import download_ohlcv, fetch_market, fetch_sp500
from mhf.models.ensemble import apply_conformal, blend

logger = logging.getLogger(__name__)

PRICE_HISTORY_DAYS = 180


def _weights_and_deltas(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull the per-horizon blend weight and conformal delta out of metrics.json,
    ordered to match Y_COLS / settings.horizons."""
    bw = metrics.get("blend_weight_chronos", {})
    cd = metrics.get("conformal_delta", {})
    names = list(settings.horizons)  # e.g. ["1m", "3m", "6m"]
    w = np.array([float(bw.get(n, 1.0)) for n in names])
    delta = np.array([float(cd.get(n, 0.0)) for n in names])
    return w, delta


def _forecast_records(latest: pd.DataFrame, ens_q: np.ndarray,
                      sectors: dict[str, str]) -> list[dict]:
    """Assemble the per-ticker forecast rows. ens_q is (n_rows, n_horizons, 3)
    of RETURNS (q10,q50,q90); anchor price lets the frontend rebuild a price fan."""
    names = list(settings.horizons)
    records = []
    for i, (_, row) in enumerate(latest.reset_index(drop=True).iterrows()):
        t = row["ticker"]
        horizons = {
            name: {
                "q10": round(float(ens_q[i, h, 0]), 6),
                "q50": round(float(ens_q[i, h, 1]), 6),
                "q90": round(float(ens_q[i, h, 2]), 6),
            }
            for h, name in enumerate(names)
        }
        records.append({
            "ticker": t,
            "sector": sectors.get(t, "Unknown"),
            "anchor_date": pd.Timestamp(row["end_date"]).strftime("%Y-%m-%d"),
            "anchor_price": round(float(row["base_close"]), 4),
            "horizons": horizons,
        })
    records.sort(key=lambda r: r["ticker"])
    return records


def _latest_feature_rows(tickers, market: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker at its MOST RECENT date — the real forecast anchor.

    Unlike the training panel, this needs no future target, so it keeps the last
    ~126 days the panel drops. Features are causal (past-only) and compute_features
    strips warmup NaNs at the start, so the final row is a valid forecast input."""
    from mhf.data.features import FEATURES, compute_features
    from mhf.data.quality import DataQualityError, clean_ohlcv

    rows = []
    for t in tickers:
        raw = download_ohlcv(t)
        if raw is None:
            continue
        try:
            clean = clean_ohlcv(raw, t)
        except DataQualityError:
            continue
        feats = compute_features(clean, market)
        if feats.empty:
            continue
        last = feats.iloc[[-1]][FEATURES].copy()
        last.insert(0, "end_date", pd.Timestamp(feats.index[-1]))
        last.insert(0, "ticker", t)
        last["base_close"] = float(feats["Close"].iloc[-1])
        rows.append(last)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _price_history(tickers, days: int = PRICE_HISTORY_DAYS) -> dict:
    """Recent daily closes per ticker (compact parallel arrays) for the fan chart."""
    out = {}
    for t in tickers:
        df = download_ohlcv(t)
        if df is None or df.empty:
            continue
        tail = df["Close"].dropna().tail(days)
        out[t] = {
            "dates": [d.strftime("%Y-%m-%d") for d in pd.to_datetime(tail.index)],
            "close": [round(float(c), 4) for c in tail.to_numpy()],
        }
    return out


def run_export(out_dir: Path, *, use_chronos: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    art = settings.data_dir / "artifacts"
    metrics = json.loads((art / "metrics.json").read_text(encoding="utf-8"))

    # tickers come from what we actually have cached (offline-safe); sectors are a
    # best-effort Wikipedia lookup — the app degrades to "Unknown" if it's down.
    tickers = sorted(p.stem for p in settings.raw_dir.glob("*.parquet"))
    try:
        _, sectors = fetch_sp500()
    except Exception as e:  # network/parse failure is non-fatal for a sector label
        logger.warning("sector lookup failed (%s); labelling all Unknown", e)
        sectors = {}

    market = fetch_market()
    latest = _latest_feature_rows(tickers, market)
    if latest.empty:
        raise RuntimeError("no usable tickers to forecast")

    with open(art / "gbm" / "model.pkl", "rb") as f:
        gbm = pickle.load(f)
    gbm_q = gbm.predict_quantiles(latest)

    if use_chronos:
        from mhf.models.chronos_ft import ChronosForecaster

        series = {}
        for t in latest["ticker"].unique():
            df = download_ohlcv(t)
            if df is not None:
                series[t] = df["Close"]
        chronos = ChronosForecaster.load(art / "chronos", series)
        chronos_q = chronos.predict_quantiles(latest)
        w, delta = _weights_and_deltas(metrics)
        ens_q = apply_conformal(blend(chronos_q, gbm_q, w), delta)
    else:
        logger.warning("use_chronos=False: exporting GBM-only forecasts (dev mode)")
        ens_q = gbm_q

    records = _forecast_records(latest, ens_q, sectors)
    prices = _price_history(latest["ticker"].tolist())

    (out_dir / "forecasts.json").write_text(
        json.dumps({"anchor_horizons": list(settings.horizons), "tickers": records}, indent=1),
        encoding="utf-8",
    )
    (out_dir / "prices.json").write_text(json.dumps(prices), encoding="utf-8")
    shutil.copyfile(art / "metrics.json", out_dir / "metrics.json")
    logger.info("wrote %d forecasts + %d price series to %s",
                len(records), len(prices), out_dir)
    return {"n_forecasts": len(records), "n_price_series": len(prices), "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/data", help="output dir for JSON")
    ap.add_argument("--no-chronos", action="store_true",
                    help="GBM-only export for fast frontend iteration (skips slow Chronos)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    info = run_export(Path(args.out), use_chronos=not args.no_chronos)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
