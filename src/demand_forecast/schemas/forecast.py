from datetime import date as dt_date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ForecastRequest(BaseModel):
    store: int = Field(..., ge=1, le=50)
    item: int = Field(..., ge=1, le=50)
    date: dt_date
    historical_sales: list[float] = Field(..., min_length=28, max_length=28)

    @field_validator("date")
    @classmethod
    def date_must_be_future(cls, v: dt_date) -> dt_date:
        if v <= dt_date.today():
            raise ValueError("date must be in the future relative to today")
        return v


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    forecast: float
    confidence_interval: tuple[float, float]
    model_version: str
    inference_time_ms: float
