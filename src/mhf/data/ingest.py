import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from mhf.config import settings

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "mhf research project (github)"}
_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _parse_sp500_html(html: str) -> tuple[list[str], dict[str, str]]:
    df = pd.read_html(io.StringIO(html))[0]
    sector_col = "GICS Sector" if "GICS Sector" in df.columns else df.columns[3]
    tickers = [str(s) for s in df["Symbol"].tolist()]
    sectors = {str(r["Symbol"]): str(r[sector_col]) for _, r in df.iterrows()}
    return tickers, sectors


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def fetch_sp500() -> tuple[list[str], dict[str, str]]:
    resp = requests.get(_WIKI_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return _parse_sp500_html(resp.text)


def cache_path(ticker: str) -> Path:
    return settings.raw_dir / f"{ticker}.parquet"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=15))
def _download_one(ticker: str) -> pd.DataFrame | None:
    sym = ticker.replace(".", "-").upper()
    df = yf.download(sym, period=settings.history_period, interval="1d", progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[_COLS].dropna().sort_index()
    return df if not df.empty else None


def download_ohlcv(ticker: str, refresh: bool = False) -> pd.DataFrame | None:
    path = cache_path(ticker)
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        return cached
    df = _download_one(ticker)
    if df is None:
        return None
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _download_close(symbol: str) -> pd.Series:
    df = yf.download(symbol, period=settings.history_period, interval="1d", progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna().sort_index()


def market_cache_path() -> Path:
    return settings.data_dir / "market.parquet"


def fetch_market(refresh: bool = False) -> pd.DataFrame:
    path = market_cache_path()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    vix = _download_close("^VIX")
    gspc = _download_close("^GSPC")
    idx = gspc.index
    out = pd.DataFrame(
        {
            "vix_close": vix.reindex(idx).ffill(),
            "sp500_ret_21d": gspc.pct_change(21, fill_method=None),
            "sp500_ret_63d": gspc.pct_change(63, fill_method=None),
        },
        index=idx,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out
