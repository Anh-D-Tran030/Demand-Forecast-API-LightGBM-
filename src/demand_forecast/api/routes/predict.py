from fastapi import APIRouter, HTTPException

from demand_forecast.core.model import get_model_bundle
from demand_forecast.ml.features import records_to_frame
from demand_forecast.schemas.forecast import ForecastRequest, ForecastResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest) -> ForecastResponse:
    try:
        frame = records_to_frame(payload.records)
        model_bundle = get_model_bundle()
        predictions = model_bundle["model"].predict(frame)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ForecastResponse(predictions=[float(value) for value in predictions])
