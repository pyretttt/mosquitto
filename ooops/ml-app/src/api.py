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

import os
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator


from .model import model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
bearer = HTTPBearer(auto_error=True)


def require_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin token not configured")
    if not secrets.compare_digest(credentials.credentials, ADMIN_TOKEN):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


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
ML_PREDICT_DURATION_SECONDS = Histogram(
    "ml_predict_duration_seconds_hist",
    "Duration of model predictions.",
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width].",
        min_length=4,
        max_length=4,
    )

    @field_validator("features", mode="after")
    @classmethod
    def ensure_list(cls, values: list[float]) -> list[float]:
        if all((val < 10.0 and val > 0.0 for val in values)):
            return values
        else:
            raise ValueError(f"{values} contains element gt 10 or lt 0")


class PredictResponse(BaseModel):
    predicted_class: int
    model_source: str | None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    message = "Validation errors:"
    for error in exc.errors():
        message += f"\nField: {error['loc']}, Error: {error['msg']}"
    return PlainTextResponse(message, status_code=400)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    with ML_PREDICT_DURATION_SECONDS.time():
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


@app.get("/admin/reload_model")
def reload_model(_: None = Depends(require_admin)):
    model.load()
    return JSONResponse(status_code=200, content={"status": "ok", "model_source": model.source_uri})
