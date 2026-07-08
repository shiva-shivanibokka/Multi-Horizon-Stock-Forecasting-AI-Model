import logging

import pandas as pd

from mhf.config import settings
from mhf.data.features import compute_features
from mhf.data.ingest import download_ohlcv
from mhf.data.quality import DataQualityError, clean_ohlcv
from mhf.data.windows import WindowSet, build_windows

logger = logging.getLogger(__name__)


def build_ticker(ticker: str, market: pd.DataFrame) -> WindowSet | None:
    raw = download_ohlcv(ticker)
    if raw is None:
        return None
    try:
        clean = clean_ohlcv(raw, ticker)
    except DataQualityError as e:
        logger.warning("skip %s: %s", ticker, e)
        return None
    feats = compute_features(clean, market)
    if len(feats) < settings.window_short + settings.max_horizon:
        return None
    return build_windows(feats, settings.window_short, settings.horizons)


if __name__ == "__main__":
    from mhf.data.ingest import fetch_sp500

    logging.basicConfig(level=logging.INFO)
    tickers, _sectors = fetch_sp500()
    # ponytail: no market-series fetch here yet -- fetch_market() lands in the
    # training plan alongside its first real consumer (YAGNI); this smoke-run
    # CLI is a manual network check, not a unit-tested code path.
    market = pd.DataFrame(columns=["vix_close", "sp500_ret_21d", "sp500_ret_63d"])
    ok = 0
    for t in tickers:
        ws = build_ticker(t, market)
        if ws is not None:
            ok += 1
    logger.info("built %d/%d tickers", ok, len(tickers))
