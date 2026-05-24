# 04 — Redis for caching + rate limiting (≈ 30 min)

You will add Redis in front of a deliberately slow FastAPI endpoint and watch the response time collapse from ~500 ms to ~1 ms. You will also implement a sliding-window rate limiter — a classic interview task.

## Concepts you must be able to explain

1. **Redis is single-threaded** for command execution. That's why each command is atomic by definition (no race between `GET` and the next op *unless* you use multi-step logic — for that there's `MULTI/EXEC`, Lua, or `WATCH`).
2. **TTL & eviction.** Use `EX` with `SET`. The eviction policy (`maxmemory-policy`) decides what to drop when memory is full: `allkeys-lru` is the common choice for a cache.
3. **Cache-aside (lazy)** pattern: app checks Redis → miss → compute → write back. *Write-through* and *write-behind* exist too.
4. **Stampede / dogpile.** When a hot key expires, hundreds of requests miss simultaneously and all recompute. Mitigations: probabilistic early expiration (XFetch), single-flight in the app, short jittered TTLs, or a "lock + compute + set" pattern.
5. **Why MLOps uses Redis.** Online feature store (Feast's online store is Redis-backed), prediction cache (idempotent inputs), rate limiting per API key, lightweight job queues (RQ / Arq), session/token storage.
6. **Serialization.** Redis stores bytes. Pick one of: JSON (debuggable), msgpack (faster, smaller), pickle (DON'T — security + version coupling), or protobuf.
7. **Connection pooling** — same as Postgres. `redis.asyncio` returns a singleton pool by default per client.

## Task

### Step 1 — Start Redis

```bash
docker compose up -d
docker compose exec redis redis-cli PING   # -> PONG
```

### Step 2 — Install deps and run the API

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Step 3 — Hit the cached endpoint

```bash
# first call is slow (~500 ms)
time curl -s 'localhost:8000/score?user_id=42'

# second call is fast (<10 ms)
time curl -s 'localhost:8000/score?user_id=42'

# different key is slow again
time curl -s 'localhost:8000/score?user_id=43'
```

Look at the API logs: you'll see `MISS` then `HIT`. In `redis-cli`:

```bash
docker compose exec redis redis-cli
KEYS score:*
TTL score:42      # should be < 60
GET score:42
```

### Step 4 — Hit the rate-limited endpoint

```bash
for i in $(seq 1 15); do
  curl -s -o /dev/null -w '%{http_code}\n' \
    'localhost:8000/predict?api_key=alice'
done
```

You should see ten `200`s followed by `429`s — the sliding-window rate limiter allows 10 req / 10 s per API key. Try `api_key=bob` immediately after — it gets its own counter.

### Step 5 — (Optional, +10 min) Break the cache, then fix the stampede

In `app/main.py` lower the TTL to `1` second and run a tiny load test:

```bash
hey -z 5s -c 50 'http://localhost:8000/score?user_id=42'   # if you have `hey`
# or:
for i in $(seq 1 500); do curl -s 'localhost:8000/score?user_id=42' & done; wait
```

You will see the slow function `_expensive_score` get called many times each second — that's the **thundering herd**. Implement the `_get_with_lock` variant noted in the code: try `SET ... NX EX 5`, if you "won the lock" you compute, otherwise you wait + retry the GET. This is the single-flight pattern in 10 lines.

## Interview questions to rehearse

- "How would you cache predictions for a model that takes 200ms per call?" → key = hash(features), value = prediction, TTL based on how often the model gets updated, invalidate on model deploy by changing the key prefix (`v17:`).
- "Cache stampede / dogpile — what's that and how do you solve it?"
- "How do you make sure stale entries are eventually evicted?" → TTL + `maxmemory-policy=allkeys-lru`.
- "Redis vs Memcached?" → Redis has data structures (lists, sorted sets, streams), persistence, replication, Lua, modules. Memcached is dumber, slightly faster for KV-only, simpler ops.
- "Redis vs in-memory dict in the app?" → in-process dict is per-pod (not shared, lost on restart). Redis is shared and durable.
- "What's the difference between SET and SETNX?" → `SETNX` only sets if key is absent — building block for distributed locks (with a TTL, and ideally with RedLock for multi-node).

## References

- redis.io docs — data types: <https://redis.io/docs/latest/develop/data-types/>
- `redis-py` (async): <https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html>
- "Cache strategies" (AWS): <https://docs.aws.amazon.com/AmazonElastiCache/latest/mem-ug/Strategies.html>
- Rate limiting algorithms: <https://blog.cloudflare.com/counting-things-a-lot-of-different-things/>
- Single-flight in distributed systems: <https://redis.io/docs/latest/develop/use/patterns/distributed-locks/>
