"""Kafka consumer worker.

Run with: python -m app.consumer

Start two of these to see partitions get rebalanced between them.
"""

import json
import logging
import signal
import sys

from confluent_kafka import Consumer, KafkaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("consumer")

TOPIC = "predictions"
GROUP_ID = "replayer"  # change to start a fresh replay (with earliest)
BOOTSTRAP = "localhost:9092"


def main() -> None:
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "security.protocol": "PLAINTEXT",
        "group.id": GROUP_ID,
        # earliest = if no committed offset for this group, replay from start
        "auto.offset.reset": "earliest",
        # We commit MANUALLY after processing -> at-least-once.
        "enable.auto.commit": False,
    })
    consumer.subscribe([TOPIC])
    log.info("subscribed group=%s topic=%s", GROUP_ID, TOPIC)

    running = True

    def _shutdown(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.error("kafka error: %s", msg.error())
                continue

            try:
                event = json.loads(msg.value())
            except json.JSONDecodeError:
                log.error("bad json at offset %d", msg.offset())
                consumer.commit(msg)  # poison pill -> skip
                continue

            log.info(
                "partition=%d offset=%d key=%s user_id=%s pred=%.4f",
                msg.partition(), msg.offset(), msg.key(),
                event.get("user_id"), event.get("prediction"),
            )

            # ===== "process" the event here =====
            # e.g. batch into ClickHouse, increment a metric, etc.

            # COMMIT only after successful processing -> at-least-once.
            consumer.commit(msg, asynchronous=False)
    finally:
        consumer.close()
        log.info("consumer closed.")


if __name__ == "__main__":
    sys.exit(main())
