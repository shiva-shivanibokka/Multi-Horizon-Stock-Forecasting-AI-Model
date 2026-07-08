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
