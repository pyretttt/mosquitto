"""FastAPI inference service.

What's wired here:
  POST /predict  — classify one sample, returns the predicted class.

What's NOT wired (your job — see ml-app/TODO.md):
  GET  /health   — liveness/readiness, must report whether the model is loaded.
  GET  /metrics  — Prometheus scrape target. Without this, the prometheus
                   scrape job for ml-app will go DOWN, the alert rule
                   `MlAppDown` will fire, and the Grafana dashboard will be
                   empty. That's the feedback loop you're learning to close.
  Custom metrics — e.g. a Counter for predictions-by-class, a Histogram for
                   model-only latency.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator


from .model import model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        model.load()
    except Exception as e:
        # Don't crash the API. Once /health exists, it will report degraded.
        log.warning("model load failed at startup: %s", e)
    yield


app = FastAPI(title="ml-app", version="0.1.0", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

COUNTER = Counter(
    "ml_predictions_total",
    "Total predictions served, labelled by predicted class.",
    ["predicted_class"],
)


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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    (pred,) = model.predict([req.features])
    COUNTER.labels(predicted_class=str(pred)).inc()
    return PredictResponse(predicted_class=pred, model_source=model.source_uri)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        status_code=200 if model.loaded else 503,
        content=dict(
            status="ok" if model.loaded else "degraded",
            model_loaded=model.loaded,
            model_source=model.source_uri,
        ),
    )
