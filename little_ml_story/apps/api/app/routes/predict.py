from __future__ import annotations

import hashlib
import time
import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from apps.api.app.cache import RedisCache
from apps.api.app.db.models import Prediction
from apps.api.app.db.repositories import PredictionRepository
from apps.api.app.db.session import AsyncSessionLocal
from apps.api.app.events import PredictionEventProducer
from apps.api.app.imagenet_labels import label_for
from apps.api.app.logging import get_logger
from apps.api.app.schemas import PredictionResponse, TopK
from apps.api.app.settings import Settings, get_settings
from apps.api.app.triton_client import TritonInferenceClient, preprocess_image, softmax

log = get_logger(__name__)

router = APIRouter()

MAX_BYTES = 8 * 1024 * 1024  # 8 MiB upload cap


def _get_cache(request: Request) -> RedisCache:
    return request.app.state.cache


def _get_triton(request: Request) -> TritonInferenceClient:
    return request.app.state.triton


def _get_producer(request: Request) -> PredictionEventProducer:
    return request.app.state.kafka


async def _rate_limit(request: Request, settings: Settings) -> None:
    cache: RedisCache = request.app.state.cache
    client = request.client.host if request.client else "anonymous"
    key = f"rl:{client}:{int(time.time() // 60)}"
    count = await cache.incr_window(key, window_seconds=60)
    if count > settings.api_rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Image file to classify")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictionResponse:
    await _rate_limit(request, settings)

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    image_sha = hashlib.sha256(body).hexdigest()
    cache = _get_cache(request)
    cache_key = f"pred:{settings.triton_model_name}:{settings.triton_model_version}:{image_sha}"

    started = time.perf_counter()
    cached = await cache.get_json(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        cached["latency_ms"] = (time.perf_counter() - started) * 1000.0
        log.info("predict.cache_hit", image_sha=image_sha, request_id=cached["request_id"])
        return PredictionResponse(**cached)

    triton = _get_triton(request)
    try:
        batch = preprocess_image(body)
        logits = await triton.infer(batch)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("predict.triton_failure", error=str(exc))
        raise HTTPException(status_code=502, detail="inference backend failed") from exc

    probs = softmax(logits.squeeze())
    top_indices = np.argsort(probs)[::-1][:5]
    top_k = [
        TopK(class_id=int(i), label=label_for(int(i)), score=float(probs[i]))
        for i in top_indices
    ]

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    request_id = str(uuid.uuid4())
    response = PredictionResponse(
        request_id=request_id,
        model_name=settings.triton_model_name,
        model_version=settings.triton_model_version,
        cache_hit=False,
        latency_ms=elapsed_ms,
        top_class=top_k[0],
        top_k=top_k,
    )

    response_dict = response.model_dump()
    await cache.set_json(cache_key, response_dict)

    async with AsyncSessionLocal() as session:
        await PredictionRepository(session).add(
            Prediction(
                id=request_id,
                image_sha256=image_sha,
                model_name=settings.triton_model_name,
                model_version=settings.triton_model_version,
                top_class_id=top_k[0].class_id,
                top_class_label=top_k[0].label,
                top_class_score=top_k[0].score,
                top_k=[t.model_dump() for t in top_k],
                latency_ms=elapsed_ms,
                cache_hit=False,
            )
        )

    producer = _get_producer(request)
    try:
        await producer.publish(
            {
                "request_id": request_id,
                "image_sha256": image_sha,
                "model_name": settings.triton_model_name,
                "model_version": settings.triton_model_version,
                "top_class_id": top_k[0].class_id,
                "top_class_label": top_k[0].label,
                "top_class_score": top_k[0].score,
                "latency_ms": elapsed_ms,
            },
            key=image_sha,
        )
    except Exception as exc:
        log.warning("predict.kafka_publish_failed", error=str(exc))

    log.info(
        "predict.ok",
        request_id=request_id,
        image_sha=image_sha,
        top_class=top_k[0].label,
        score=top_k[0].score,
        latency_ms=elapsed_ms,
    )
    return response
