from functools import lru_cache
from pathlib import Path

import joblib

from demand_forecast.core.config import settings


@lru_cache(maxsize=1)
def get_model_bundle() -> dict:
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
