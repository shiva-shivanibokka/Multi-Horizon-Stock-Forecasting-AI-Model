from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MHF_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    horizons: dict[str, int] = Field(default_factory=lambda: {"1w": 5, "1m": 21, "6m": 126})
    window_long: int = 756
    window_short: int = 252
    history_period: str = "20y"

    @computed_field
    @property
    def max_horizon(self) -> int:
        return max(self.horizons.values())


settings = Settings()
