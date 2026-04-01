from pydantic import BaseModel


class ForecastRecord(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class ForecastRequest(BaseModel):
    records: list[ForecastRecord]


class ForecastResponse(BaseModel):
    predictions: list[float]
