"""FastAPI /predict endpoint that:
  1. Runs a fake model.
  2. Publishes a 'prediction_made' event to Kafka topic `predictions`.

The producer runs in its own thread (managed by confluent-kafka). Calling
.produce() only enqueues — the network IO happens off-thread, so we don't
block the FastAPI event loop. We flush on shutdown to avoid losing messages.
"""

import json
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager

from confluent_kafka import Producer
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

TOPIC = "predictions"
BOOTSTRAP = "localhost:9092"

producer = Producer({
    "bootstrap.servers": BOOTSTRAP,
    "client.id": "predictions-api",
    # acks=all is the safe default; leader will wait for in-sync replicas.
    "acks": "all",
    # The producer batches messages internally for throughput; these are tuned
    # for low-latency dev. In prod set linger.ms higher (e.g. 20) for throughput.
    "linger.ms": 5,
    "enable.idempotence": True,  # avoids duplicates on retry
})


def _delivery(err, msg):
    if err:
        log.error("kafka delivery failed: %s", err)
    else:
        log.info(
            "kafka delivered partition=%d offset=%d key=%s",
            msg.partition(), msg.offset(), msg.key(),
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Block until all in-flight messages are sent or fail.
    log.info("flushing producer ...")
    producer.flush(10)


app = FastAPI(title="Prediction API + Kafka", lifespan=lifespan)


class PredictIn(BaseModel):
    user_id: int
    features: list[float]


class PredictOut(BaseModel):
    request_id: str
    prediction: float


@app.post("/predict", response_model=PredictOut)
async def predict(payload: PredictIn) -> PredictOut:
    # Fake "model": deterministic-ish on the features.
    prediction = round(sum(payload.features) / max(len(payload.features), 1)
                       + random.random() * 0.01, 4)
    request_id = uuid.uuid4().hex

    event = {
        "request_id": request_id,
        "user_id": payload.user_id,
        "features": payload.features,
        "prediction": prediction,
        "ts": time.time(),
    }
    # KEY = user_id  -> events for same user go to the SAME partition,
    # which guarantees per-user ordering downstream.
    producer.produce(
        TOPIC,
        key=str(payload.user_id).encode(),
        value=json.dumps(event).encode(),
        on_delivery=_delivery,
    )
    # poll(0) services the delivery callbacks without blocking.
    producer.poll(0)

    return PredictOut(request_id=request_id, prediction=prediction)
