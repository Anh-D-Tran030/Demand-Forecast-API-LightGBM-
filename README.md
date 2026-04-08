# Demand Forecast API

LightGBM-backed REST API for store-item demand forecasting with sub-3ms inference.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│                                                         │
│   Client ──► FastAPI ──► LightGBM Model ──► Response   │
│               (port 8000)   (lru_cache)                 │
│                                │                        │
│                         volume mount                    │
└─────────────────────────────────────────────────────────┘
                                 │
                          ./model/ (host)
                          └── model.pkl
```

## Quickstart

```bash
git clone https://github.com/Anh-D-Tran030/Demand-Forecast-API-LightGBM-.git
cd Demand-Forecast-API-LightGBM-

# Train model (generates synthetic data automatically)
python src/demand_forecast/ml/train.py

# Start API
docker compose up
```

**Predict request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store": 1,
    "item": 1,
    "date": "2026-04-10",
    "historical_sales": [10,12,11,13,14,12,10,11,13,12,14,15,11,10,
                         12,13,14,12,11,10,13,14,12,11,10,12,13,14]
  }'
```

**Response:**

```json
{
  "forecast": 11.02,
  "confidence_interval": [8.13, 13.91],
  "model_version": "1.0.0",
  "inference_time_ms": 1.77
}
```

## API Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| POST | `/predict` | Forecast demand for a store-item pair | Live |
| GET | `/health` | Liveness check | Live |
| GET | `/metrics` | Prometheus-style metrics | Planned |

## Model Performance

| Metric | Value |
|--------|-------|
| MAE | 2.94 |
| Inference time | <3ms (warm) |
| Training time | <30s |
| Image size | ~169MB |

## Engineering Decisions

- **LightGBM over neural networks for tabular demand data** — tree-based gradient boosting consistently outperforms deep learning on structured tabular data with engineered lag/rolling features, trains in seconds rather than minutes, and produces interpretable feature importances. Neural nets require far more data and tuning to match LightGBM's out-of-the-box accuracy on this problem shape.

- **Multi-stage Docker build** — a builder stage installs all build-time dependencies (compilers, headers) and a lean runtime stage copies only the installed packages, cutting the final image size from ~800MB to ~169MB and reducing the attack surface by excluding dev tooling from production.

- **Time-based train/test split over random split** — demand forecasting is an inherently temporal problem: the model must predict future sales from past data. A random split allows the model to learn from future observations to predict the past, producing optimistic evaluation metrics that don't reflect real deployment performance. Holding out the last 30 days as the test set enforces the correct causal ordering.

## Project Structure

```
src/
└── demand_forecast/
    ├── api/
    │   ├── main.py              # FastAPI app factory
    │   └── routes/
    │       ├── health.py        # GET /health
    │       └── predict.py       # POST /predict
    ├── core/
    │   ├── config.py            # Pydantic settings (env vars)
    │   └── model.py             # Cached model loader
    ├── ml/
    │   ├── features.py          # Feature engineering + inference features
    │   └── train.py             # Training pipeline (generates synthetic data)
    └── schemas/
        └── forecast.py          # ForecastRequest / ForecastResponse
```
