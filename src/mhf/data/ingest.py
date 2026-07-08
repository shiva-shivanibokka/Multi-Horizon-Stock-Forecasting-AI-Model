import io

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "mhf research project (github)"}


def _parse_sp500_html(html: str) -> tuple[list[str], dict[str, str]]:
    df = pd.read_html(io.StringIO(html))[0]
    sector_col = "GICS Sector" if "GICS Sector" in df.columns else df.columns[3]
    tickers = [str(s) for s in df["Symbol"].tolist()]
    sectors = {str(r["Symbol"]): str(r[sector_col]) for _, r in df.iterrows()}
    return tickers, sectors


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def fetch_sp500() -> tuple[list[str], dict[str, str]]:
    html = requests.get(_WIKI_URL, headers=_HEADERS, timeout=15).text
    return _parse_sp500_html(html)
