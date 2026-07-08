import numpy as np
import pandas as pd

import mhf.data.ingest as ingest


def _fake_download(symbol, **kwargs):
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    close = 100 + np.cumsum(rng.normal(0, 1, size=300))
    return pd.DataFrame({"Close": np.clip(close, 5, None)}, index=idx)


def test_fetch_market_columns_and_causality(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.settings, "data_dir", tmp_path)
    monkeypatch.setattr(ingest.yf, "download", _fake_download)

    m = ingest.fetch_market(refresh=True)
    assert list(m.columns) == ["vix_close", "sp500_ret_21d", "sp500_ret_63d"]
    # Trailing returns: row t must equal close[t]/close[t-21]-1 -> depends only on the past.
    assert m["sp500_ret_21d"].isna().sum() == 21  # exactly the warmup, no bfill
    assert (ingest.settings.data_dir / "market.parquet").exists()


def test_fetch_market_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.settings, "data_dir", tmp_path)
    calls = {"n": 0}

    def counting_download(symbol, **kwargs):
        calls["n"] += 1
        return _fake_download(symbol, **kwargs)

    monkeypatch.setattr(ingest.yf, "download", counting_download)
    ingest.fetch_market(refresh=True)
    n_after_first = calls["n"]
    ingest.fetch_market()  # cache hit -> no new downloads
    assert calls["n"] == n_after_first
