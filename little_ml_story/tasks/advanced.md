# Advanced track

Once the fast track is in your hands, work these blocks in whatever order maps to your interview pipeline. They build *on top of* the existing repo — no greenfield rewrites.

Each exercise has a difficulty tag (easy / medium / hard) and a rough time estimate.

---

## Triton Inference Server

- **[easy, 30m] Multi-version traffic split.** Drop a second version of ResNet50 under `infra/triton/model_repository/resnet50/2/` and a `version_policy: { specific: { versions: [1, 2] } }` clause. Add a small router in FastAPI that sends 10% of traffic to v2.
- **[medium, 1h] Ensemble model with a Python preprocessing backend.** Move image decode + resize + normalise into a [Python backend](https://github.com/triton-inference-server/python_backend) model named `preprocess`, then declare an ensemble that pipes raw bytes → `preprocess` → `resnet50`. FastAPI then just uploads raw bytes.
- **[medium, 45m] Dynamic batching tuning.** Bench with `perf_analyzer` (`docker run --rm nvcr.io/nvidia/tritonserver:24.08-py3-sdk perf_analyzer -m resnet50 -u <host>:8001 -i grpc -b 4 --concurrency-range 1:16`). Tune `preferred_batch_size` and `max_queue_delay_microseconds` for both latency and throughput.
- **[medium, 45m] Model warmup.** Add `model_warmup` to `config.pbtxt` so cold starts don't spike p99.
- **[hard, 2h] BLS (Business Logic Scripting).** Use the Python backend's BLS to route between two model versions based on input shape inside Triton itself.

## Kafka

- **[easy, 30m] Consumer lag dashboard.** Add a `kminion` or `redpanda metrics` exporter, scrape it in Prometheus, plot `kafka_consumergroup_lag` per partition in Grafana.
- **[medium, 1h] Avro + Schema Registry.** Switch the predictions topic to Avro using Redpanda's built-in schema registry (already running on `:8081`). Update producer and consumer; show that a backward-incompatible schema is rejected at publish time.
- **[medium, 45m] Dead letter topic.** When the consumer hits a bad payload, publish it to `predictions.dlq` instead of crashing. Build a tiny replay CLI.
- **[hard, 1.5h] Exactly-once.** Wrap the consumer's "insert into ClickHouse + commit offset" into a transactional outbox stored in Postgres, or research Redpanda's transactional API.

## ClickHouse

- **[easy, 30m] Projections.** Add a projection on `(model_version, top_class_label)` and watch `system.parts` populate. Compare query plans with and without.
- **[medium, 45m] Materialized view to aggregate state.** Replace `hourly_class_counts` with an AggregatingMergeTree-backed MV using `quantileState`/`avgState` so you can compute exact p99 per hour after the fact.
- **[medium, 1h] S3 cold storage.** Configure `storage_configuration` with an S3 disk (use MinIO locally), add a TTL `MOVE` policy on `prediction_events` for parts older than 7 days.
- **[hard, 1.5h] Kafka engine table.** Skip the Python consumer entirely and use a ClickHouse `Engine = Kafka` source table with an MV that materialises into `prediction_events`. Compare reliability vs the custom consumer.

## Postgres

- **[easy, 30m] `pg_stat_statements`.** Already enabled via `init.sql`. Run a load test, then sort by `mean_exec_time` and explain what you'd index.
- **[medium, 45m] Range partitioning by month.** Partition `predictions` by `created_at`. Write a migration that creates partitions for the last 3 months. Use `pg_partman` if you want to go further.
- **[medium, 1h] Logical replication.** Spin up a second Postgres in compose as a `replica`, set up a publication / subscription for `predictions`, and switch reads in the API to a `replica_url`.
- **[hard, 1.5h] Connection pooling.** Add PgBouncer in front, mark the pool as `transaction` mode, observe what breaks with prepared statements (asyncpg uses them by default).

## Redis

- **[easy, 30m] Streams.** Replace the JSON cache with a Redis Stream of recent predictions and consume it from a small dashboard.
- **[medium, 45m] Distributed lock for cache stampede.** Use `SET key value NX PX 5000` to ensure only one Triton call per unseen image even under burst load.
- **[medium, 1h] RedisBloom for dedupe.** Use a Bloom filter to short-circuit "have I ever seen this image hash?" before hitting Postgres.

## Spark

- **[medium, 1h] Partitioned JDBC read.** Re-do the daily rollup with `partitionColumn=event_time`, four partitions, observe the parallelism in the Spark UI.
- **[medium, 1h] Structured Streaming.** Read directly from Kafka `predictions`, micro-batch into ClickHouse, checkpoint to a local directory.
- **[hard, 2h] Spark on k8s.** Run the daily job as a Spark on Kubernetes submission against the kind cluster.

## Prometheus + Grafana

- **[easy, 30m] Recording rules.** Pre-compute the API RED metrics into 5m/1h recording rules so dashboards become cheaper.
- **[medium, 45m] Alertmanager.** Wire Alertmanager into compose, route warnings to a webhook (use `requestbin.com` or similar).
- **[medium, 1h] SLO dashboards.** Implement a multi-window multi-burn-rate alert per Google SRE workbook for an availability SLO of 99.5%.

## OpenSearch / ELK

- **[easy, 30m] Ingest pipeline.** Parse `event` strings into structured fields with an `ingest pipeline`, set it as the default for `mlops-logs-*`.
- **[medium, 45m] ILM (Index Lifecycle Management).** Set up a rollover policy (hot 1d → warm 7d → delete).
- **[medium, 1h] Log-based alert.** Create an alert in OpenSearch on `event: "predict.kafka_publish_failed"` > 10 per minute.

## Kubernetes

- **[easy, 30m] ConfigMap + Secret split.** Move the API env in `values.yaml` into a separate `ConfigMap`/`Secret` template.
- **[medium, 45m] ServiceMonitor.** Install kube-prometheus-stack and have it scrape the API via a `ServiceMonitor` instead of static config.
- **[medium, 1h] NetworkPolicy.** Lock down Postgres/Redis so only `api` and `consumer` can talk to them.
- **[hard, 2h] Argo CD.** Reverse the flow: a tiny Argo CD installation that watches `infra/k8s/`, with the `api` chart as an `Application`.

## Cross-cutting

- **[medium, 45m] Auth.** Replace the rate limit with a stub JWT verifier (HS256, secret in env). Add tests.
- **[medium, 1h] OpenTelemetry tracing.** Instrument the API with OTel, send traces to Tempo (or Jaeger), add a Grafana datasource, link logs ↔ traces via `trace_id`.
- **[hard, 1.5h] mTLS FastAPI → Triton.** Generate a tiny CA, give Triton a server cert, configure `tritonclient[grpc]` with mTLS.
- **[hard, 2h] CI.** Add GitHub Actions: ruff + pytest on PR, image build + push on `main`, ephemeral PR env on a single workflow.
