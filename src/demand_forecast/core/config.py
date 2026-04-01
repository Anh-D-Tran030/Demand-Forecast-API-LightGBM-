from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: Path = Path("model") / "model.pkl"
    model_config = SettingsConfigDict(env_prefix="DEMAND_FORECAST_", env_file=".env")


settings = Settings()
