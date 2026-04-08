from time import perf_counter

import numpy as np
from fastapi import APIRouter, HTTPException

from demand_forecast.core.model import get_model_bundle
from demand_forecast.ml.features import build_inference_features
from demand_forecast.schemas.forecast import ForecastRequest, ForecastResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest) -> ForecastResponse:
    try:
        model_bundle = get_model_bundle()
        start = perf_counter()
        frame = build_inference_features(
            store=payload.store,
            item=payload.item,
            date=payload.date,
            historical_sales=payload.historical_sales,
        )
        model = model_bundle["model"]
        forecast = float(model.predict(frame)[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid model bundle: {exc}") from exc

    sales_std = float(np.std(payload.historical_sales, ddof=1))
    if np.isnan(sales_std):
        sales_std = 0.0
    interval_half_width = max(1.0, 1.96 * sales_std)

    lower = max(0.0, forecast - interval_half_width)
    upper = forecast + interval_half_width
    inference_time_ms = (perf_counter() - start) * 1000.0

    return ForecastResponse(
        forecast=forecast,
        confidence_interval=(float(lower), float(upper)),
        model_version=str(model_bundle.get("version", "unknown")),
        inference_time_ms=float(inference_time_ms),
    )
