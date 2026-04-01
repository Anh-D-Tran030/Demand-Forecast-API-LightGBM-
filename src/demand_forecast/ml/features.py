import pandas as pd

from demand_forecast.schemas.forecast import ForecastRecord

REQUIRED_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def records_to_frame(records: list[ForecastRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record.model_dump() for record in records])
    return frame[REQUIRED_FEATURES]
