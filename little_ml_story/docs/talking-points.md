# Talking points

One page per tool. For each: the *60-second pitch* you'd say if asked "tell me what you did with X", plus *three follow-up questions* the interviewer is likely to ask. Use this to rehearse out loud.

## FastAPI

**60s pitch.** The gateway in `apps/api` is an async FastAPI app. Long-lived resources — Redis client, Kafka producer, Triton client — are wired up in `lifespan` so they share a connection pool across requests. The `/predict` route accepts a multipart upload, hashes the bytes, checks Redis for a cached result, otherwise preprocesses the image (PIL → NCHW float32) and calls Triton over gRPC. I offload the sync gRPC client to a thread executor so it doesn't block the event loop. The whole thing exposes `/metrics` via `prometheus-fastapi-instrumentator` and emits structlog JSON to stdout.

**Be ready for…**
- "What's the difference between `Depends`, app state, and a module-level singleton?"
- "How would you keep this snappy under 1000 RPS?"
- "Why async if Triton is the bottleneck anyway?"

## Triton Inference Server

**60s pitch.** Triton serves a ResNet50 v1.7 ONNX model from a model_repository on disk. `config.pbtxt` declares input/output tensors, FP32 NCHW, max batch size 8, dynamic batching with a 1ms queue delay window. The server polls the repo every 30s so I can hot-swap weights without a restart. The FastAPI gateway dials the gRPC endpoint at `triton:8001`. Triton exposes Prometheus metrics on `:8002`, and Grafana visualises `nv_inference_request_duration_us`, `queue_duration_us`, and request count per model.

**Be ready for…**
- "What does the ONNX backend buy us vs ONNX Runtime in-process in FastAPI?" (model-server isolation, batching, multi-model, GPU pooling.)
- "How does dynamic batching work, and what's the trade-off?"
- "How would you A/B two versions of the model?"

## PostgreSQL

**60s pitch.** Postgres 16 stores per-request metadata: `predictions(id, image_sha256, top_class, top_k, latency_ms, ...)`. The API uses SQLAlchemy 2.0's async API with asyncpg under the hood. Migrations live in `apps/api/alembic/versions/`. `pg_stat_statements` is enabled out of the box so I can grep for slow queries. The Spark batch job's daily rollup lands in `daily_class_counts` via a stage-and-merge using `ON CONFLICT … DO UPDATE`.

**Be ready for…**
- "How would you scale writes to 5k inserts/sec?" (batching, COPY, partitioning, separate analytics DB.)
- "How does asyncpg differ from psycopg2 in production?"
- "When would you choose Postgres over ClickHouse here?"

## Redis

**60s pitch.** Redis 7 plays two roles: a 5-minute response cache keyed by `pred:<model>:<version>:<image_sha>`, and a sliding-window rate limiter using `INCR + EXPIRE NX` per IP per minute. Connection is async via `redis.asyncio`. The hit rate is exposed via a Prometheus counter that I added on Day 1.

**Be ready for…**
- "What about cache stampedes?" (advanced track: per-key lock via `SET NX PX`.)
- "How do you invalidate when the model changes?" (cache key includes `model_version`.)
- "Persistence story?" (AOF on, snapshot off; pretend it's ephemeral.)

## Kafka (Redpanda)

**60s pitch.** Redpanda is a drop-in Kafka API broker — kept here for fast boot, but the consumer is plain `aiokafka` so the lesson transfers. Each prediction emits a JSON event keyed by `image_sha256` (same image = same partition = ordering). The producer is `acks=all`, idempotent, lz4-compressed. The consumer in `apps/consumer/consumer.py` reads the `predictions` topic with `enable_auto_commit=False`, batches 100 or 1s, inserts into ClickHouse, **then** commits — at-least-once delivery.

**Be ready for…**
- "Why not exactly-once?" (transactional outbox is in `advanced.md`.)
- "How does partition count affect throughput and ordering guarantees?"
- "What's the failure mode if ClickHouse is down for 10 minutes?"

## ClickHouse

**60s pitch.** ClickHouse 24.8 stores raw `prediction_events` in a MergeTree partitioned by day, ordered by `(model_name, model_version, event_time)`, with a 30-day TTL. A materialized view rolls hourly counts into a SummingMergeTree. Labels use `LowCardinality(String)` to compress well. Grafana queries ClickHouse directly with the official datasource plugin.

**Be ready for…**
- "ORDER BY vs PRIMARY KEY?" (ORDER BY = primary key by default; tied to the sparse primary index for skipping.)
- "When would you reach for `AggregatingMergeTree` over `SummingMergeTree`?"
- "How do projections compare to MVs?"

## Prometheus + Grafana

**60s pitch.** Prometheus scrapes the API, Triton, Postgres exporter, and Redis exporter every 10s, with two alerting rules: 5xx ratio > 5% and p95 latency > 1s. Grafana is provisioned from files (`infra/grafana/provisioning/`) so dashboards live in git as JSON. I implemented an API RED dashboard (rate, errors, latency p50/p95/p99), a Triton inference panel set, and a ClickHouse-backed analytics dashboard.

**Be ready for…**
- "RED vs USE — when do you reach for each?"
- "Histogram vs summary?"
- "How would you do SLOs and burn-rate alerts?"

## OpenSearch / ELK

**60s pitch.** Vector tails Docker container logs, parses the JSON line emitted by structlog, enriches it with the container's service label, and bulk-indexes into `mlops-logs-YYYY.MM.DD` on OpenSearch. Dashboards is the UI; I have a saved search filtered on `service: api AND event: "predict.ok"`. ILM and ingest pipelines are in the advanced track.

**Be ready for…**
- "ELK vs OpenSearch licensing context?"
- "What goes in a log message vs a metric vs a trace?"
- "How would you rotate indices in production?"

## Spark

**60s pitch.** A nightly batch job in `apps/spark_jobs/daily_aggregations.py` reads yesterday's rows from ClickHouse via JDBC, groups by `top_class_label`, aggregates count and average score, stages into a temporary Postgres table, then upserts into `daily_class_counts` with `ON CONFLICT … DO UPDATE`. Runs as `spark-submit` in a Docker container with both JDBC drivers baked in.

**Be ready for…**
- "When would you turn this into Structured Streaming?"
- "What's a `partitionColumn` and when does it matter?"
- "What's `broadcast` join good for?"

## Kubernetes (kind + Helm)

**60s pitch.** A local kind cluster runs the API behind a Helm chart with NodePort `30080` mapped to host `8000`. Stateful deps (Postgres, Redis, a minimal Triton) come in as raw YAML manifests because they're not what I'm learning. The chart supports replicaCount, resource requests/limits, readiness/liveness probes, and a toggleable HPA on CPU.

**Be ready for…**
- "Deployment vs StatefulSet — why and when?"
- "Readiness vs liveness probe in 30 seconds."
- "Helm vs Kustomize — pros and cons?"

---

## Cross-cutting "tell me about the project" pitch (90 seconds)

> I built a small but complete MLOps lab: a FastAPI gateway in front of Triton serving a ResNet50 ONNX classifier. The gateway caches in Redis, persists metadata to Postgres, and emits each prediction as a Kafka event. A small Python consumer drains the topic into ClickHouse, and a nightly Spark batch rolls those events into a Postgres summary table. Prometheus scrapes everything, Grafana visualises both Prometheus metrics and ClickHouse analytics, and Vector ships structured logs to OpenSearch. All of it runs from a single `docker compose up`, and I have a thin Helm chart deploying the gateway on a local kind cluster. The repo doubles as a study workbook with day-by-day tasks for the parts I want to deepen — model ensembles in Triton, materialized views in ClickHouse, exactly-once on the Kafka leg, and an OpenTelemetry trace tree across the whole thing.
