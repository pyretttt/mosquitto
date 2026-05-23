# Day 1 — Serving spine

**Goal:** A `POST /predict` endpoint that takes an image, calls **Triton** (ResNet50 ONNX) over gRPC, caches in **Redis**, and writes a row to **Postgres**. Everything boots from `make up && make migrate`.

Time budget: 3-4 hours.

## Background (~20 min)

- Triton Inference Server overview — what model_repository / config.pbtxt do: <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md>
- ONNX Runtime backend in Triton: <https://github.com/triton-inference-server/onnxruntime_backend>
- `tritonclient` Python API: <https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_grpc_infer_client.py>
- SQLAlchemy 2.0 async patterns: <https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html>
- Redis with `redis.asyncio`: <https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html>

## Exercises

### 1. Boot the spine (20 min)
```bash
make env && make fetch-model
docker compose -f docker-compose.yml up -d --build postgres redis triton
make migrate
docker compose -f docker-compose.yml up -d api
```
**Accept:** `curl localhost:8000/healthz` returns `{"api": true, "triton": true, "redis": true}`.

### 2. Read the request path (15 min)
Open [apps/api/app/routes/predict.py](../apps/api/app/routes/predict.py) and trace one call end-to-end. You should be able to explain:
- where preprocessing happens and why we pin to 224x224 NCHW,
- where the cache key is built,
- when the DB row is committed relative to publishing the Kafka event.

### 3. Hit it from `curl` (10 min)
Grab a cat photo, save as `tests/fixtures/cat.jpg`, then:
```bash
make predict
make predict   # second call
```
**Accept:** the second call returns `"cache_hit": true` with sub-5ms `latency_ms`.

### 4. Inspect Postgres (15 min)
```bash
docker compose exec postgres psql -U mlops -c \
  "SELECT id, top_class_label, top_class_score, latency_ms, cache_hit FROM predictions ORDER BY created_at DESC LIMIT 5;"
```
Then add an index on `(model_name, created_at DESC)` via a new alembic migration. Re-run `make migrate`.

**Accept:** `make migration m="api index"` autogens a non-empty revision; `\d predictions` shows your new index.

### 5. Stress the cache (20 min)
Change `REDIS_CACHE_TTL_SECONDS` to `10` in `.env`, restart `api`, and confirm the second-after-ten-seconds call misses the cache. Add a Prometheus-style counter for cache hits/misses in [cache.py](../apps/api/app/cache.py) (look up `prometheus_client.Counter`).

**Accept:** `curl localhost:8000/metrics | grep cache_` shows your counter rising.

### 6. Break Triton, observe the error (15 min)
Stop the triton container (`docker compose stop triton`). Hit `/predict` and notice the 502. Re-read the exception path in `predict.py`. Bring Triton back up.

**Accept:** you can explain the difference between failure modes (Triton down, model not loaded, bad input shape).

### 7. Stretch: wire a model warmup (30 min, optional)
Add `model_warmup` to [config.pbtxt](../infra/triton/model_repository/resnet50/config.pbtxt) with a zero-tensor sample so the first inference after restart isn't slow. Reference: <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-warmup>

## Talking points to lock in tonight

- **FastAPI**: lifespan-managed resources (Kafka producer, Redis pool), why we keep one Triton client per process, async + thread offload for the gRPC sync client.
- **Triton**: model_repository layout, polling vs explicit control mode, dynamic batching basics, why FP32 NCHW for ONNX ResNet50.
- **Postgres**: async SQLAlchemy 2.0 sessions, alembic for migrations, when to commit, indexing strategy for write-heavy tables.
- **Redis**: cache TTL choice, sliding-window rate limit via `INCR + EXPIRE NX`, when *not* to cache (write-after-read consistency).

## Definition of done

- [ ] `make up && make migrate` succeeds from a clean checkout.
- [ ] `make predict` returns a sensible top-1 label.
- [ ] Repeated `make predict` shows `cache_hit: true`.
- [ ] `predictions` table has rows; you've added at least one custom index.
- [ ] You can describe the request lifecycle to yourself out loud in 90 seconds.
