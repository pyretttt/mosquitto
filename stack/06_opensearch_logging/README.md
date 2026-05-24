# 06 — OpenSearch for structured logs (≈ 45 min)

You will run OpenSearch + OpenSearch Dashboards (the AWS-maintained fork of Elasticsearch + Kibana), ship structured JSON logs to it from a Python service, and answer the typical interview question *"a user reports prediction errors — show me what happened"*.

> **OpenSearch vs Elasticsearch**: same query DSL, same concepts (index, shard, mapping, analyzer). OpenSearch is an Apache-2 fork after Elastic re-licensed. Everything you learn here applies to ELK 1-to-1.

## Concepts you must be able to explain

1. **Document, index, shard, replica.** Each log line = a JSON *document*. Documents live in an *index*. Each index is split into *primary shards* (immutable Lucene segments per shard) and optional *replica shards* for HA & read scaling.
2. **Mappings.** The schema. Fields are typed (`keyword`, `text`, `long`, `date`, `ip`, `nested` …). **`keyword` vs `text`** is the #1 confusion: `text` is analysed (tokenised, lower-cased) for full-text search; `keyword` is the raw string for exact match / aggregations.
3. **Inverted index.** For text fields, OS builds a map `token → list of doc IDs`. Why search is fast.
4. **Index template + ILM/ISM.** "Logs of the day" pattern: write to `logs-myapp-2026.05.24`, controlled by a template (consistent mappings) and ISM policy (rollover daily, delete after 14 days, hot/warm/cold tiers).
5. **Logging pipeline.** App → JSON to stdout → Filebeat/Fluent Bit → OpenSearch. For local learning we send directly from Python.
6. **Why structured logs.** `"latency_ms": 472` is filterable, sortable, aggregatable. `"served in 472 ms"` is a string and useless for analytics.
7. **What you actually search for as an MLOps engineer.** `level:ERROR AND model:"churn_v2" AND ts:[now-1h TO now]`. Group by `request_id` to reconstruct a full trace.

## Task

### Step 1 — Start OpenSearch + Dashboards

```bash
docker compose up -d
docker compose logs -f opensearch | head -50    # wait for "started"
```

Open Dashboards: <http://localhost:5601>
Open the API: <http://localhost:9200>

If you get a memory error, give Docker Desktop 4+ GB RAM.

### Step 2 — Apply the index template

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_setup_template.py
```

Read the script — note that we set `request_id` as `keyword` (we want exact match + group-by) and `message` as `text` (we want full-text search).

### Step 3 — Run the fake service that emits logs

```bash
python scripts/02_emit_logs.py    # produces ~500 fake inference events
```

This script imitates a real FastAPI handler: each "request" emits 2–3 structured log lines (`request_started`, `inference_done`, sometimes `error`) all sharing a `request_id`.

### Step 4 — Search like at 3 AM during an incident

```bash
python scripts/03_search.py
```

It runs three queries you must be able to write:

1. **All errors in the last 24 h** (`bool / must / range`).
2. **Slowest 5 inferences** (`sort` on `latency_ms desc`).
3. **Error rate by model_version** (aggregation, equivalent of `GROUP BY`).

### Step 5 — Build a dashboard in the UI

In Dashboards (<http://localhost:5601>):

1. *Stack Management → Index Patterns → Create* `logs-mlops-*` with `@timestamp` as the time field.
2. *Discover* → filter `level:ERROR` → see the same errors as in step 4.
3. *Visualize* → vertical bar → count split by `model_version.keyword`.
4. *Dashboard* → combine: errors-over-time line chart + p95 latency by model.

You don't need to keep the dashboard, you need to be able to say *"we built dashboards in OpenSearch that show error rate and p95 latency per model_version, sliced by region"* in an interview.

## Interview questions to rehearse

- "Difference between `keyword` and `text`." (The classic.)
- "What is an inverted index?"
- "Why daily indices instead of one big `logs` index?" → cheaper retention (drop a whole index = O(1)), better mapping evolution, smaller shards.
- "How many shards should a daily index have?" → rule of thumb: ~20–50 GB per shard, aim for ≤ 1 primary if your daily volume is small; scale up otherwise. Over-sharding kills the cluster.
- "Logs vs metrics vs traces — which goes where?" → metrics in Prometheus/VictoriaMetrics, logs in OpenSearch/Loki, traces in Tempo/Jaeger. They cross-link via `request_id` / `trace_id`.
- "How would you stop logs from filling the disk?" → ISM rollover by size/time + delete after N days + monitor disk watermark settings.

## References

- OpenSearch quickstart: <https://opensearch.org/docs/latest/quickstart/>
- Query DSL: <https://opensearch.org/docs/latest/query-dsl/>
- Mappings: <https://opensearch.org/docs/latest/field-types/>
- "Elasticsearch the definitive guide" — still the best intro despite age: <https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html>
- Index State Management (ISM): <https://opensearch.org/docs/latest/im-plugin/ism/index/>
- Python client: <https://opensearch.org/docs/latest/clients/python-low-level/>
