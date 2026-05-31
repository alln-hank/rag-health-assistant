import json
import time
from collections import defaultdict, deque
from typing import Any

from backend.app.config import settings


class CacheService:
    def __init__(self) -> None:
        self.redis = None
        self.redis_status = "disabled"
        self._memory_cache: dict[str, tuple[float, Any]] = {}
        self._memory_rate: dict[str, deque[float]] = defaultdict(deque)

    async def connect(self) -> None:
        if not settings.redis_url:
            self.redis_status = "memory-fallback"
            return

        try:
            import redis.asyncio as redis

            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            await self.redis.ping()
            self.redis_status = "connected"
        except Exception:
            self.redis = None
            self.redis_status = "memory-fallback"

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()

    async def get_json(self, key: str) -> Any | None:
        if self.redis is not None:
            value = await self.redis.get(key)
            return json.loads(value) if value else None

        item = self._memory_cache.get(key)
        if not item:
            return None

        expire_at, value = item
        if expire_at and expire_at < time.time():
            self._memory_cache.pop(key, None)
            return None
        return value

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or settings.cache_ttl_seconds
        if self.redis is not None:
            await self.redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
            return

        self._memory_cache[key] = (time.time() + ttl, value)

    async def allow_request(self, client_id: str, scope: str = "global") -> bool:
        limit = settings.rate_limit
        window = settings.rate_window_seconds
        key = f"rate:{scope}:{client_id}"

        if self.redis is not None:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, window)
            return count <= limit

        now = time.time()
        bucket = self._memory_rate[key]
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


cache_service = CacheService()
