import logging
from pathlib import Path

import pandas as pd

from mhf.config import settings
from mhf.data.build import build_ticker
from mhf.data.features import FEATURES

logger = logging.getLogger(__name__)

# Target columns are the horizon keys, in horizon order — so they always line up
# with windows.y_ret (built from settings.horizons.values()) and never drift apart.
Y_COLS = [f"y_{name}" for name in settings.horizons]


def build_panel(tickers, market: pd.DataFrame, downloader=build_ticker) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        ws = downloader(ticker, market)
        if ws is None:
            logger.info("skip %s: no windows", ticker)
            continue
        # .copy() is load-bearing: ws.X[:, -1, :] is a VIEW into the ticker's full
        # (n, 252, 35) window array, and pd.DataFrame keeps that view alive — without
        # the copy, every ticker's whole X array is retained (500 tickers x ~160 MB
        # ≈ 70 GB, which pages to death). The copy lets ws.X be freed each iteration.
        feats = pd.DataFrame(ws.X[:, -1, :].copy(), columns=FEATURES)
        feats.insert(0, "end_date", pd.to_datetime(ws.end_dates))
        feats.insert(0, "ticker", ticker)
        for i, col in enumerate(Y_COLS):
            feats[col] = ws.y_ret[:, i]
        feats["base_close"] = ws.base_close
        frames.append(feats)
    if not frames:
        return pd.DataFrame(columns=["ticker", "end_date", *FEATURES, *Y_COLS, "base_close"])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=Y_COLS).reset_index(drop=True)
    return panel


def write_panel(panel: pd.DataFrame, path: Path | None = None) -> Path:
    if path is None:
        path = settings.data_dir / "processed" / "panel.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path
