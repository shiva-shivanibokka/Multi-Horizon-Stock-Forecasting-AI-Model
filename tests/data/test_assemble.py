import numpy as np
import pandas as pd

from mhf.data.assemble import Y_COLS, build_panel, write_panel
from mhf.data.features import FEATURES
from mhf.data.windows import WindowSet


def _fake_windowset(n=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 252, len(FEATURES))).astype(np.float32)
    y = rng.normal(0, 0.05, size=(n, 3)).astype(np.float32)
    dates = pd.bdate_range("2021-01-01", periods=n).to_numpy()
    base = rng.uniform(50, 150, size=n).astype(np.float32)
    return WindowSet(X=X, y_ret=y, end_dates=dates, base_close=base)


def test_build_panel_shape_and_columns():
    def fake_downloader(ticker, market):
        return None if ticker == "BAD" else _fake_windowset(seed=abs(hash(ticker)) % 1000)

    panel = build_panel(["AAA", "BAD", "BBB"], market=pd.DataFrame(), downloader=fake_downloader)

    assert list(panel.columns) == ["ticker", "end_date", *FEATURES, *Y_COLS, "base_close"]
    assert set(panel["ticker"]) == {"AAA", "BBB"}  # BAD skipped
    assert len(panel) == 20  # 10 rows each
    # Last-timestep feature copied verbatim.
    assert not panel[FEATURES].isna().any().any()


def test_build_panel_last_timestep_is_used():
    ws = _fake_windowset(n=3, seed=42)

    def one(ticker, market):
        return ws

    panel = build_panel(["AAA"], market=pd.DataFrame(), downloader=one)
    np.testing.assert_allclose(
        panel[FEATURES].to_numpy(), ws.X[:, -1, :], rtol=1e-6
    )
    np.testing.assert_allclose(panel[Y_COLS].to_numpy(), ws.y_ret, rtol=1e-6)


def test_write_panel_roundtrip(tmp_path):
    panel = build_panel(["AAA"], market=pd.DataFrame(),
                        downloader=lambda t, m: _fake_windowset(n=4))
    path = write_panel(panel, tmp_path / "p.parquet")
    assert path.exists()
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, panel)
