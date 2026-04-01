# Demand Forecast API

LightGBM-backed FastAPI service for demand forecasting.

## Quickstart

1. Install dependencies:

```bash
pip install -e .[dev]
```

2. Download dataset:

```bash
python scripts/download_data.py
```

3. Train model:

```bash
PYTHONPATH=src python -m demand_forecast.ml.train
```

4. Run API:

```bash
PYTHONPATH=src uvicorn demand_forecast.api.main:app --reload
```
