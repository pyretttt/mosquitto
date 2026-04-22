"""FastAPI inference service.

Endpoints:
  GET  /health     — liveness/readiness, reports model status.
  POST /predict    — classify one sample.
  GET  /metrics    — Prometheus scrape target (auto-added by the instrumentator).

The Prometheus instrumentator gives you http_request_duration_seconds histograms
and request counters out of the box. We also register a tiny custom counter to
show you *how* to add domain-specific metrics.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from .model import model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


PREDICTIONS = Counter(
    "ml_predictions_total",
    "Total predictions served, labelled by predicted class.",
    ["predicted_class"],
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        model.load()
    except Exception as e:
        # Don't crash the API; /health will show "model not loaded" so you can
        # debug in the browser. TODO(you): add readiness gate so k8s doesn't
        # send traffic until the model is loaded.
        log.warning("model load failed at startup: %s", e)
    yield


app = FastAPI(title="ml-app", version="0.1.0", lifespan=lifespan)

# Mounts /metrics and wires http_* metrics automatically.
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
def health() -> dict:
    return {
        "status": "ok" if model.loaded else "degraded",
        "model_loaded": model.loaded,
        "model_source": model.source_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    (pred,) = model.predict([req.features])
    PREDICTIONS.labels(predicted_class=str(pred)).inc()
    return PredictResponse(predicted_class=pred, model_source=model.source_uri)
