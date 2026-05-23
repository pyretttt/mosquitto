# Day 3 — Spark batch + k8s + polish

**Goal:** A nightly **Spark** rollup from ClickHouse into Postgres, the API running on a local **kind** cluster via a **Helm** chart, and a polished README you'd be happy to send a recruiter.

Time budget: 3-4 hours.

## Background (~20 min)

- Spark JDBC source/sink intro: <https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html>
- ClickHouse JDBC driver notes: <https://clickhouse.com/docs/en/integrations/java#jdbc>
- kind quickstart: <https://kind.sigs.k8s.io/docs/user/quick-start/>
- Helm chart basics: <https://helm.sh/docs/chart_template_guide/getting_started/>

## Exercises

### 1. Run the Spark job once (25 min)
```bash
make up-all              # ensure ClickHouse has data
make load                # populate the day with traffic
TARGET_DAY=$(date -u +%F) docker compose run --rm spark spark-submit \
  --master local[2] /app/apps/spark_jobs/daily_aggregations.py
```

**Accept:** `SELECT * FROM daily_class_counts ORDER BY day DESC LIMIT 5;` in Postgres shows yesterday or today's rollup with a non-zero `n`.

### 2. Tweak the rollup (20 min)
Edit [apps/spark_jobs/daily_aggregations.py](../apps/spark_jobs/daily_aggregations.py) to also store `min_score` and `max_score`. You'll need a new alembic migration to extend the `daily_class_counts` table. Restart the job.

**Accept:** new columns appear in Postgres and your job succeeds with no stack trace.

### 3. Reason about partitions (10 min)
With this lab dataset Spark uses a single JDBC partition. Look up `partitionColumn`, `lowerBound`, `upperBound`, `numPartitions` and write down (in a comment in the file) what you'd do at 10M rows/day. Don't implement it now — note it for the advanced track.

### 4. Build the API image and load it into kind (20 min)
```bash
docker build -t little-ml-story-api:latest -f apps/api/Dockerfile .
make kind-up
kind load docker-image little-ml-story-api:latest --name little-ml-story
```

**Accept:** `kubectl --context kind-little-ml-story get nodes` shows control-plane + worker `Ready`.

### 5. Helm install the API (25 min)
```bash
make helm-install
kubectl -n mlops get pods -w
```

The kind config maps node port `30080` to host port `8000`:

```bash
curl -fsS http://localhost:8000/livez
```

Now `helm upgrade` with `--set replicaCount=4 --reuse-values` and watch new pods schedule.

**Accept:** the chart installs cleanly, `/livez` returns 200, scaling replicas up/down works.

### 6. Toggle the HPA (15 min)
```bash
helm upgrade api infra/k8s/helm/api -n mlops --reuse-values \
  --set autoscaling.enabled=true
kubectl -n mlops get hpa -w
```

Generate CPU pressure with a quick loop calling `/livez` and watch the HPA react (it won't necessarily scale because there's no real CPU load — the point is recognising what it's doing).

### 7. Light load test from outside kind (15 min)
With the kind port mapping in place, point k6 at `localhost:8000`. Compare RPS and p95 against the Compose run from Day 2. Note: Triton in the kind cluster has an empty model_repository, so `/predict` will fail — that's expected. The point is to compare the framework-level latency for `/livez` / `/healthz`.

**Accept:** you have two p95 numbers and an opinion on why they differ.

### 8. Polish (30 min)
- Update [README.md](../README.md) with a "results" section: p95, RPS, screenshot links.
- Add a short [docs/runbook.md](../docs/runbook.md) covering "what do I do when X breaks" for each service.
- Skim [docs/talking-points.md](../docs/talking-points.md) and adjust anything that doesn't sound like you.

## Talking points

- **Spark**: when batch vs streaming, JDBC partitioning tactics, predicate pushdown, why we stage-and-merge instead of writing the table directly.
- **Kubernetes**: deployment vs statefulset, readiness vs liveness probe semantics, HPA on CPU vs custom metrics, when to reach for Helm.
- **Helm**: `_helpers.tpl` patterns, `--reuse-values`, why dependencies usually live in their own charts.

## Definition of done

- [ ] `daily_class_counts` has at least one rolled-up day.
- [ ] kind cluster exists; `helm list -n mlops` shows the `api` release.
- [ ] You can rattle off the trade-offs between Compose and k8s in 30 seconds.
- [ ] README has a results section.

## You shipped a story

Stand back, look at the whole stack: a user uploads an image, the gateway calls Triton, caches in Redis, logs to Postgres, emits to Kafka, sinks to ClickHouse, gets rolled up nightly by Spark, is monitored by Prometheus and visualised in Grafana, with logs queryable in OpenSearch, all running on either Compose or kind. That **is** the JD. Now pivot to [advanced.md](advanced.md) and keep grinding until the call.
