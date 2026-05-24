"""FastAPI demo of two Redis patterns:

1. /score  : cache-aside in front of an artificially slow "model".
2. /predict: per-API-key sliding-window rate limiter using a sorted set.

Run:  uvicorn app.main:app --reload --port 8000
"""

import asyncio
import json
import logging
import time

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Redis cache + rate limit demo")

# A single connection pool is shared across all requests (default behaviour).
rds: redis.Redis = redis.from_url("redis://localhost:6379/0", decode_responses=True)


# ---------- 1. cache-aside ----------

CACHE_TTL = 60  # seconds


async def _expensive_score(user_id: int) -> dict:
    """Pretend a heavy model takes ~500 ms."""
    await asyncio.sleep(0.5)
    return {"user_id": user_id, "score": (user_id * 1_103_515_245 + 12345) % 1000 / 1000}


@app.get("/score")
async def score(user_id: int) -> dict:
    key = f"score:{user_id}"

    cached = await rds.get(key)
    if cached is not None:
        log.info("HIT  %s", key)
        return json.loads(cached)

    log.info("MISS %s", key)
    result = await _expensive_score(user_id)
    # SET with TTL — atomic in one round-trip.
    await rds.set(key, json.dumps(result), ex=CACHE_TTL)
    return result


# ---------- 2. sliding-window rate limiter ----------
#
# Algorithm (per API key):
#   ZADD limit:<key> <now_ms> <uuid>      # add this request
#   ZREMRANGEBYSCORE limit:<key> -inf now-window_ms   # drop old
#   ZCARD limit:<key>                     # how many in window?
#   EXPIRE limit:<key> window_s           # garbage-collect inactive keys
#
# We do it in a MULTI/EXEC pipeline so the four commands are atomic
# (Redis sees them as one block; no other client can interleave).

WINDOW_SEC = 10
MAX_REQUESTS = 10


async def _allow(api_key: str) -> tuple[bool, int]:
    now_ms = int(time.time() * 1000)
    window_ms = WINDOW_SEC * 1000
    z_key = f"limit:{api_key}"

    async with rds.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(z_key, 0, now_ms - window_ms)
        pipe.zadd(z_key, {f"{now_ms}-{id(object())}": now_ms})
        pipe.zcard(z_key)
        pipe.expire(z_key, WINDOW_SEC)
        _, _, count, _ = await pipe.execute()

    return count <= MAX_REQUESTS, count


@app.get("/predict")
async def predict(api_key: str = Query(..., min_length=1)) -> dict:
    allowed, count = await _allow(api_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit: max {MAX_REQUESTS} req / {WINDOW_SEC}s (current: {count})",
        )
    return {"api_key": api_key, "prediction": 0.42, "used_in_window": count}


@app.on_event("shutdown")
async def _close() -> None:
    await rds.aclose()
