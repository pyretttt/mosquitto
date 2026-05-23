# Runbook

Short, opinionated "what do I do when X breaks" notes. Treat as a starter
the user fills in once they've actually broken each service themselves.

## API returns 502 on `/predict`

Likely Triton is not ready or the model failed to load.

1. `docker compose ps triton` — running?
2. `curl http://localhost:8800/v2/health/ready` — model server healthy?
3. `curl http://localhost:8800/v2/models/resnet50` — model loaded?
4. `docker compose logs triton --tail 100` — look for ONNX load errors. Most common cause: `infra/triton/model_repository/resnet50/1/model.onnx` is missing. Fix: `make fetch-model`.

## `/predict` is slow

1. Grafana → API RED dashboard. Is p95 spiking on `/predict` only, or everything?
2. Grafana → Triton dashboard. `nv_inference_queue_duration_us` rising means the model is the bottleneck.
3. Look at cache hit rate in Prometheus: `rate(cache_hits_total[1m]) / rate(http_requests_total{handler="/predict"}[1m])`. Low? Verify TTL.
4. Last resort: `EXPLAIN ANALYZE` the slowest Postgres query from `pg_stat_statements`.

## Consumer is lagging

1. Redpanda Console → group `predictions-clickhouse-sink`. Lag per partition.
2. Look at consumer logs for "consumer.flushed" events. Frequency dropped?
3. Is ClickHouse healthy? `docker compose exec clickhouse clickhouse-client --query "SELECT 1"`.
4. Bump `CONSUMER_BATCH_SIZE` if the bottleneck is many small inserts.

## OpenSearch isn't getting logs

1. `docker compose logs vector --tail 50` — any sink errors?
2. `curl http://localhost:9200/_cat/indices?v` — `mlops-logs-*` present?
3. Containers need the label `logs: include` (api and consumer have it).

## Grafana panel is empty

1. Datasource health: Grafana → Connections → Data sources. Test connection.
2. For Prometheus panels, check the metric exists: `curl localhost:9090/api/v1/label/__name__/values | jq | grep <metric>`.
3. For ClickHouse panels, run the SQL directly in `http://localhost:8123/play`.

## kind cluster won't start

1. Docker has enough RAM (4 GiB+) and CPUs?
2. `kind delete cluster --name little-ml-story && make kind-up`.
3. After cluster is up: `kind load docker-image little-ml-story-api:latest --name little-ml-story`.
