from __future__ import annotations

import orjson
from aiokafka import AIOKafkaProducer

from apps.api.app.logging import get_logger

log = get_logger(__name__)


class PredictionEventProducer:
    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            value_serializer=lambda v: orjson.dumps(v),
            enable_idempotence=True,
            acks="all",
            compression_type="lz4",
        )
        await self._producer.start()
        log.info("kafka_producer.started", topic=self._topic)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            log.info("kafka_producer.stopped")

    async def publish(self, event: dict, key: str | None = None) -> None:
        if self._producer is None:
            raise RuntimeError("Producer not started")
        await self._producer.send_and_wait(
            self._topic,
            value=event,
            key=key.encode() if key else None,
        )
