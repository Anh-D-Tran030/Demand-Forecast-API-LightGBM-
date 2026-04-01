from fastapi import FastAPI

from demand_forecast.api.routes.health import router as health_router
from demand_forecast.api.routes.predict import router as predict_router


def create_app() -> FastAPI:
    app = FastAPI(title="Demand Forecast API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(predict_router)
    return app


app = create_app()
