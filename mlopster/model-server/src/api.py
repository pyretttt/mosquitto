"""FastAPI inference service for mlopster.

What's wired:
  POST /predict             — classify one Iris sample.
  GET  /health              — readiness/liveness; 503 until the model is loaded.
  GET  /metrics             — Prometheus exposition (instrumentator).
  POST /admin/reload-model  — re-pull the artifact from S3 (Bearer ADMIN_TOKEN).

Custom metrics:
  ml_predictions_total{predicted_class}   Counter
  ml_predict_duration_seconds             Histogram (model-only latency)

TODO(you): see model-server/TODO.md — readiness gating, richer validation,
and the MLflow loading path are left as exercises.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from .config import settings
from .model import model

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("model-server")

bearer = HTTPBearer(auto_error=True)

PREDICTIONS = Counter(
    "ml_predictions_total",
    "Total predictions served, labelled by predicted class.",
    ["predicted_class"],
)
PREDICT_DURATION = Histogram(
    "ml_predict_duration_seconds",
    "Duration of the model.predict() call (excludes HTTP overhead).",
)


def require_admin(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
    if not settings.admin_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "admin token not configured")
    if not secrets.compare_digest(creds.credentials, settings.admin_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Attempt to load at startup; don't crash if there's no artifact yet.
    model.load()
    yield


app = FastAPI(title="model-server", version="0.1.0", lifespan=lifespan)

# Exposes /metrics with default http_* metrics used by the dashboards/alerts.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width].",
        min_length=4,
        max_length=4,
    )


class PredictResponse(BaseModel):
    predicted_class: int
    model_source: str | None


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        status_code=200 if model.loaded else 503,
        content={
            "status": "ok" if model.loaded else "degraded",
            "model_loaded": model.loaded,
            "model_source": model.source_uri,
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    with PREDICT_DURATION.time():
        (pred,) = model.predict([req.features])
    PREDICTIONS.labels(predicted_class=str(pred)).inc()
    return PredictResponse(predicted_class=pred, model_source=model.source_uri)


@app.post("/admin/reload-model")
def reload_model(_: None = Depends(require_admin)) -> JSONResponse:
    ok = model.load()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"reloaded": ok, "model_source": model.source_uri},
    )
