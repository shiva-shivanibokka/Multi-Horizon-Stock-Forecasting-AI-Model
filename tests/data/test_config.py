from mhf.config import Settings


def test_defaults_and_max_horizon():
    s = Settings()
    assert s.horizons == {"1m": 21, "3m": 63, "6m": 126}
    assert s.max_horizon == 126
    assert s.window_long == 756 and s.window_short == 252


def test_env_override(monkeypatch):
    monkeypatch.setenv("MHF_HISTORY_PERIOD", "10y")
    assert Settings().history_period == "10y"
