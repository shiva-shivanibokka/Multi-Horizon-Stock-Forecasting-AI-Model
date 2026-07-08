import pandas as pd

from mhf.data import ingest
from mhf.data.ingest import _parse_sp500_html

SAMPLE_HTML = """
<table class="wikitable"><thead><tr><th>Symbol</th><th>Security</th><th>x</th>
<th>GICS Sector</th></tr></thead><tbody>
<tr><td>AAPL</td><td>Apple</td><td>-</td><td>Information Technology</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>-</td><td>Financials</td></tr>
</tbody></table>
"""


def test_parse_sp500_html():
    tickers, sectors = _parse_sp500_html(SAMPLE_HTML)
    assert tickers == ["AAPL", "BRK.B"]
    assert sectors["AAPL"] == "Information Technology"
    assert sectors["BRK.B"] == "Financials"


def test_download_ohlcv_uses_cache(tmp_path, monkeypatch, ohlcv):
    monkeypatch.setattr(ingest.settings, "raw_dir", tmp_path)
    ohlcv.to_parquet(tmp_path / "AAPL.parquet")

    def _boom(*a, **k):
        raise AssertionError("network must not be hit when cache exists")

    monkeypatch.setattr(ingest, "_download_one", _boom)
    out = ingest.download_ohlcv("AAPL")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(out, ohlcv)
