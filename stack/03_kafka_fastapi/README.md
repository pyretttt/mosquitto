# 03 — Kafka + FastAPI (≈ 60 min)

You will build a tiny **event-driven inference logging pipeline**:

```
client → FastAPI /predict (produces "prediction_made" event) → Kafka → consumer worker → stdout
```

In a real system the consumer would write to ClickHouse (see task 2) or trigger a retraining DAG. This decoupling is the #1 reason Kafka exists in MLOps: producers don't care who consumes, and consumers can be slow / down without blocking the request.

## Concepts you must be able to explain

1. **Topic, partition, offset.** A topic is split into N partitions. Each partition is an append-only log. Each message has an offset (its position in the log). Order is guaranteed **within a partition**, not across partitions.
2. **Key → partition routing.** When you produce with `key=user_id`, all events for the same user go to the same partition → ordered. If you produce with `key=None`, Kafka round-robins.
3. **Consumer group.** A group ID is the unit of "I want a copy of the stream". Within a group, partitions are split between consumers — each partition is read by exactly one consumer in the group. To scale, add more consumers; max useful = number of partitions.
4. **Offset commits.** The consumer tells the broker "I processed up to offset X". On restart, it resumes from there. **`auto.offset.reset`** decides what to do if no committed offset exists: `earliest` (replay all) or `latest` (only new messages).
5. **At-least-once vs exactly-once.** Default is at-least-once (you might process the same message twice if you crash after processing but before committing). Idempotent consumers handle this.
6. **Why not just HTTP?** Backpressure (Kafka stores), replay (Kafka keeps history), fan-out (many consumer groups, one producer), decoupling (deploy times don't need to align).
7. **Sync vs async producer in FastAPI.** The `confluent-kafka` producer is non-blocking by design — `produce()` queues, a background thread sends. You must call `flush()` on shutdown, otherwise messages are lost.

## Task

### Step 1 — Start Kafka (KRaft mode, no Zookeeper)

```bash
docker compose up -d
docker compose logs -f kafka   # wait for "Kafka Server started"
```

`docker-compose.yml` uses **PLAINTEXT** listeners (`PLAINTEXT_HOST` on `0.0.0.0:9092`, advertised as `localhost:9092`) so FastAPI on your Mac can connect through the published port. If you see `Disconnected while requesting ApiVersion` / “SSL listener” from `confluent-kafka`, the broker was almost certainly bound to `localhost` *inside* the container only — recreate with `docker compose up -d` after any listener change.

### Step 2 — Create a topic with 3 partitions

The `apache/kafka` image ships admin scripts under `/opt/kafka/bin/` (they are not on `PATH` like the old Bitnami image).

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic predictions --partitions 3 --replication-factor 1
```

List it:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --describe --topic predictions
```

Notice: **3 partitions, 1 replica** (1 because we have 1 broker; in prod you'd want 3 replicas across 3 brokers).

### Step 3 — Install deps and start the API

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

### Step 4 — Start the consumer in another terminal

```bash
source .venv/bin/activate
python -m app.consumer
```

### Step 5 — Generate traffic

```bash
# in a third terminal
for i in $(seq 1 30); do
  curl -s -X POST localhost:8000/predict \
    -H 'content-type: application/json' \
    -d "{\"user_id\": $((RANDOM % 5)), \"features\": [0.1, 0.2, 0.3]}"
done
```

Watch the consumer log. You'll see:
- Events for `user_id=0` always come from the **same partition** (because we use `user_id` as the key).
- Within a partition, offsets monotonically increase.

### Step 6 — Scale by starting a second consumer

```bash
# fourth terminal
python -m app.consumer
```

Watch: Kafka does a "rebalance" — partitions are redistributed. Each consumer now owns ~half of them. Start a 4th consumer → one consumer sits idle (only 3 partitions exist). This is the rule "max parallelism = partitions".

### Step 7 — Replay everything (different consumer group)

Edit `GROUP_ID` in `app/consumer.py` to `"replayer"` and restart. With a new group + `auto.offset.reset=earliest`, the consumer replays the entire topic from offset 0. This is how a new "log-into-ClickHouse" consumer would bootstrap from history.

## Interview questions to rehearse

- "How many partitions should I choose for a topic?" → enough for your peak parallelism (you cannot increase order-preserving parallelism beyond it). 6 / 12 / 24 are common starting points. Repartitioning later is annoying.
- "Producer ack settings?" → `acks=0` (fire-and-forget, lose data on broker crash), `acks=1` (leader confirms), `acks=all` + `min.insync.replicas=2` (durable).
- "What happens if my consumer is slower than the producer?" → lag grows. You monitor `kafka_consumergroup_lag`. Add consumers up to partition count, or repartition with more partitions.
- "Kafka vs RabbitMQ vs Redis Streams?" → Kafka = high-throughput durable log, replay, big data. RabbitMQ = traditional queue, per-message ack, routing. Redis Streams = simple, low-latency, smaller scale.
- "How would you log model predictions to ClickHouse using Kafka?" → producer in the API → topic `predictions` → consumer batches 10k rows → `client.insert("inferences", ...)` to CH.

## References

- Apache Kafka — quickstart: <https://kafka.apache.org/quickstart>
- "A practical introduction to Kafka" (Confluent): <https://developer.confluent.io/courses/apache-kafka/>
- `confluent-kafka-python` docs: <https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html>
- "Kafka the definitive guide" — chapter 4 (consumers): free PDF on Confluent.
- KRaft mode (no Zookeeper) — <https://developer.confluent.io/learn/kraft/>
