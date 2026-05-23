import orjson
import redis.asyncio as aioredis


class RedisCache:
    """Tiny JSON cache + sliding-window rate limiter on top of Redis."""

    def __init__(self, host: str, port: int, ttl_seconds: int) -> None:
        self._client: aioredis.Redis = aioredis.Redis(
            host=host, port=port, decode_responses=False
        )
        self._ttl = ttl_seconds

    async def close(self) -> None:
        await self._client.aclose()

    async def get_json(self, key: str) -> dict | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        return orjson.loads(raw)

    async def set_json(self, key: str, value: dict) -> None:
        await self._client.set(key, orjson.dumps(value), ex=self._ttl)

    async def incr_window(self, key: str, window_seconds: int) -> int:
        """Returns the new counter value. Sets expiry on first increment."""
        pipe = self._client.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, window_seconds, nx=True)
        new_value, _ = await pipe.execute()
        return int(new_value)

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False
