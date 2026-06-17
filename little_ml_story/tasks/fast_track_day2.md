# Day 2 — Streaming + observability

**Goal:** Predictions stream through **Kafka** into **ClickHouse**. **Prometheus** scrapes everything, **Grafana** has live dashboards, container logs land in **OpenSearch**.

Time budget: 3-4 hours.

## Background (~25 min)

- Kafka semantics primer (delivery guarantees, partitions, consumer groups): <https://kafka.apache.org/documentation/#intro_concepts>
- aiokafka quickstart: <https://aiokafka.readthedocs.io/en/stable/index.html>
- ClickHouse for events 101 (MergeTree, partitioning, TTL, materialized views): <https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree>
- Prometheus RED method: <https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/>
- Vector + OpenSearch sink: <https://vector.dev/docs/reference/configuration/sinks/elasticsearch/>

## Exercises

### 1. Boot the full stack (15 min)
```bash
make up-all
docker compose ps  # everything healthy
```
Open Grafana (admin/admin), Prometheus, Redpanda Console, OpenSearch Dashboards, ClickHouse Play.

### 2. Drive traffic, watch it land (15 min)
Hit `/predict` a dozen times with different images (force cache misses by tweaking the filename — content hash differs).

```bash
docker compose exec clickhouse clickhouse-client --query \
  "SELECT count(), avg(latency_ms), max(latency_ms) FROM mlops.prediction_events"
```

**Accept:** the count grows in real time and matches Postgres rows for new requests.

### 3. Read the consumer (15 min)
Open [apps/consumer/consumer.py](../apps/consumer/consumer.py). Walk through:
- `enable_auto_commit=False` and explicit `commit()` after a successful insert — what failure modes does that buy us?
- `getmany(timeout_ms=500)` + buffer + size-or-time flush — classic micro-batch pattern.

Then bump `CONSUMER_BATCH_SIZE` to `500` and confirm fewer, larger ClickHouse inserts via `SELECT * FROM system.query_log ORDER BY event_time DESC LIMIT 5`.

### 4. Add a ClickHouse aggregate (20 min)
On top of the existing `hourly_class_counts_mv`, write a query that gives you predictions per minute for the last hour. Pin it as a saved query in ClickHouse Play. Reference: <https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions#tostartofminute>

**Accept:** you can compute p95 latency per model per 5-minute bucket with one query.

### 5. Prometheus + Grafana RED (25 min)
Open [infra/grafana/dashboards/api-red.json](../infra/grafana/dashboards/api-red.json). In Grafana, edit the third panel and add a row for **error budget burn**:
```
1 - (sum(rate(http_requests_total{service="api",status=~"5.."}[5m])) / sum(rate(http_requests_total{service="api"}[5m])))
```
Save it as a copy. Look at the **Triton** dashboard and explain `nv_inference_queue_duration_us` to yourself.

**Accept:** dashboards populate when you run `k6 run infra/load/k6.js` for 30 seconds (`make load`).

### 6. Alert rules (15 min)
[infra/prometheus/rules/api.yml](../infra/prometheus/rules/api.yml) already has two rules. Curl the Prometheus alerts API:
```bash
curl -s http://localhost:9090/api/v1/rules | jq
```
Trigger `ApiHighErrorRate` by stopping Triton and hammering `/predict` for a few minutes.

**Accept:** the alert flips to `FIRING` in the Prometheus UI.

### 7. OpenSearch logs (20 min)
After traffic, open Dashboards (5601), Management -> Data Views -> create `mlops-logs-*` with `timestamp` as the time field. Save a search filtered on `service: api AND event: "predict.ok"`. Notice the structured fields (`top_class`, `latency_ms`, `image_sha`) — that's why we emit JSON from `structlog`.

**Accept:** you can filter to the last 5 minutes of successful predictions and see them as a table.

### 8. Stretch: idempotent producer + key choice (20 min, optional)
We already set `enable_idempotence=True`. Reason about what would change if we used `request_id` as the Kafka key instead of `image_sha256`. Hint: ordering vs. parallelism.

## Talking points

- **Kafka**: at-least-once vs exactly-once, why partitioning by `image_sha256` keeps hot images on the same consumer, idempotent producer guarantees.
- **ClickHouse**: MergeTree partition pruning, `LowCardinality(String)` for labels, materialized views for pre-aggregation, why we keep TTL low in the lab.
- **Prometheus**: pull model, scrape jobs vs service discovery, RED vs USE methods, why histograms over averages.
- **Grafana**: provisioning datasources/dashboards via files (no clicks), alerting vs Prometheus rules.
- **OpenSearch/ELK**: log shipping topology (Vector vs Filebeat vs Fluent Bit), index naming conventions, ILM teaser.

## Definition of done

- [x] `make up-all` brings up 12+ healthy containers.
- [x] `make load` populates the API RED and Triton dashboards.
- [ ] At least one Prometheus alert has gone `FIRING` and recovered.
- [ ] You can describe the path of a single prediction event from FastAPI to a Grafana panel and to an OpenSearch search hit.
