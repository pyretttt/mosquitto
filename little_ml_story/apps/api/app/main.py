from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from apps.api.app.cache import RedisCache
from apps.api.app.events import PredictionEventProducer
from apps.api.app.logging import configure_logging, get_logger
from apps.api.app.routes import health, predict
from apps.api.app.settings import get_settings
from apps.api.app.triton_client import TritonConfig, TritonInferenceClient

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.api_log_level)
    log.info("api.starting", model=settings.triton_model_name, triton=settings.triton_url)

    app.state.cache = RedisCache(
        host=settings.redis_host,
        port=settings.redis_port,
        ttl_seconds=settings.redis_cache_ttl_seconds,
    )
    app.state.triton = TritonInferenceClient(
        TritonConfig(
            url=settings.triton_url,
            model_name=settings.triton_model_name,
            model_version=settings.triton_model_version,
        )
    )
    app.state.kafka = PredictionEventProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.kafka_prediction_topic,
    )
    await app.state.kafka.start()

    try:
        yield
    finally:
        await app.state.kafka.stop()
        await app.state.cache.close()
        log.info("api.stopped")


app = FastAPI(
    title="little_ml_story",
    version="0.1.0",
    summary="FastAPI gateway in front of Triton serving ResNet50 ONNX.",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(predict.router, tags=["inference"])

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
