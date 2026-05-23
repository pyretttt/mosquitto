"""Kafka -> ClickHouse sink.

- Subscribes to `predictions`.
- Buffers a small window of messages, then bulk-inserts into ClickHouse.
- Commits offsets only after the insert succeeds (at-least-once delivery).
"""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from datetime import datetime, timezone

import clickhouse_connect
import orjson
import structlog
from aiokafka import AIOKafkaConsumer

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger("consumer")


BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
TOPIC = os.getenv("KAFKA_PREDICTION_TOPIC", "predictions")
GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "predictions-clickhouse-sink")
CH_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CH_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "mlops")
BATCH_SIZE = int(os.getenv("CONSUMER_BATCH_SIZE", "100"))
FLUSH_INTERVAL_S = float(os.getenv("CONSUMER_FLUSH_INTERVAL_S", "1.0"))

COLUMNS = [
    "event_time",
    "request_id",
    "image_sha256",
    "model_name",
    "model_version",
    "top_class_id",
    "top_class_label",
    "top_class_score",
    "latency_ms",
]


def event_to_row(event: dict) -> list:
    return [
        datetime.now(tz=timezone.utc),
        event["request_id"],
        event["image_sha256"],
        event["model_name"],
        str(event["model_version"]),
        int(event["top_class_id"]),
        event["top_class_label"],
        float(event["top_class_score"]),
        float(event["latency_ms"]),
    ]


async def run() -> None:
    ch = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE,
    )
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: orjson.loads(v),
    )
    await consumer.start()
    log.info("consumer.started", topic=TOPIC, group=GROUP_ID, bootstrap=BOOTSTRAP)

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)

    buffer: list[list] = []
    last_flush = loop.time()

    async def flush() -> None:
        if not buffer:
            return
        rows = list(buffer)
        buffer.clear()
        await asyncio.to_thread(
            ch.insert, "prediction_events", rows, column_names=COLUMNS
        )
        await consumer.commit()
        log.info("consumer.flushed", rows=len(rows))

    try:
        while not stopping.is_set():
            try:
                msg_pack = await asyncio.wait_for(consumer.getmany(timeout_ms=500), timeout=1.0)
            except asyncio.TimeoutError:
                msg_pack = {}

            for _tp, msgs in msg_pack.items():
                for msg in msgs:
                    try:
                        buffer.append(event_to_row(msg.value))
                    except Exception as exc:
                        log.warning("consumer.bad_message", error=str(exc), value=msg.value)

            now = loop.time()
            if len(buffer) >= BATCH_SIZE or (buffer and now - last_flush >= FLUSH_INTERVAL_S):
                await flush()
                last_flush = now
    finally:
        with suppress(Exception):
            await flush()
        await consumer.stop()
        ch.close()
        log.info("consumer.stopped")


if __name__ == "__main__":
    asyncio.run(run())
