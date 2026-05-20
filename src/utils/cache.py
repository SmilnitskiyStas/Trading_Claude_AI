from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from src.utils.config import REDIS_URL, REDIS_DB
from src.utils.logger import logger


class CacheManager:
    def __init__(self, url: str = REDIS_URL, db: int = REDIS_DB):
        self._url = url
        self._db = db
        self.client: aioredis.Redis | None = None
        self.available: bool = False

    async def connect(self) -> bool:
        try:
            self.client = aioredis.from_url(
                self._url,
                db=self._db,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self.client.ping()
            self.available = True
            logger.info(f"Redis connection OK: {self._url}")
        except Exception as e:
            self.available = False
            logger.warning(f"Redis unavailable — running in degraded mode (no cache): {e}")
        return self.available

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self.available or self.client is None:
            return
        try:
            encoded = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, encoded)
            else:
                await self.client.set(key, encoded)
        except Exception as e:
            logger.warning(f"Redis SET failed for key={key}: {e}")

    async def _get(self, key: str) -> Any | None:
        if not self.available or self.client is None:
            return None
        try:
            data = await self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Redis GET failed for key={key}: {e}")
            return None

    # ── Tickers (TTL: 10s) ─────────────────────────────────────────────────

    async def set_ticker(self, exchange: str, symbol: str, data: dict) -> None:
        await self._set(f"ticker:{exchange}:{symbol}", data, ttl=10)

    async def get_ticker(self, exchange: str, symbol: str) -> Optional[dict]:
        return await self._get(f"ticker:{exchange}:{symbol}")

    # ── ML signals (TTL: 5min) ─────────────────────────────────────────────

    async def set_signal(self, symbol: str, signal: dict) -> None:
        await self._set(f"signal:{symbol}", signal, ttl=300)

    async def get_signal(self, symbol: str) -> Optional[dict]:
        return await self._get(f"signal:{symbol}")

    # ── AI Agent response (TTL: 1h) ────────────────────────────────────────

    async def set_agent_analysis(self, analysis: dict) -> None:
        await self._set("agent:analysis", analysis, ttl=3600)

    async def get_agent_analysis(self) -> Optional[dict]:
        return await self._get("agent:analysis")

    # ── News sentiment (TTL: 30min) ────────────────────────────────────────

    async def set_sentiment(self, symbol: str, score: float) -> None:
        if not self.available or self.client is None:
            return
        try:
            await self.client.setex(f"sentiment:{symbol}", 1800, str(score))
        except Exception as e:
            logger.warning(f"Redis set_sentiment failed: {e}")

    async def get_sentiment(self, symbol: str) -> Optional[float]:
        if not self.available or self.client is None:
            return None
        try:
            data = await self.client.get(f"sentiment:{symbol}")
            return float(data) if data else None
        except Exception as e:
            logger.warning(f"Redis get_sentiment failed: {e}")
            return None

    # ── Portfolio state (no TTL) ───────────────────────────────────────────

    async def set_portfolio_state(self, state: dict) -> None:
        await self._set("portfolio:state", state)

    async def get_portfolio_state(self) -> Optional[dict]:
        return await self._get("portfolio:state")

    # ── News headlines (TTL: 15min) ────────────────────────────────────────

    async def set_news_headlines(self, headlines: list) -> None:
        await self._set("news:headlines", headlines, ttl=900)

    async def get_news_headlines(self) -> Optional[list]:
        return await self._get("news:headlines")

    # ── Pub/Sub ────────────────────────────────────────────────────────────

    async def publish_trade(self, trade: dict) -> None:
        if not self.available or self.client is None:
            return
        try:
            await self.client.publish("channel:trades", json.dumps(trade))
        except Exception as e:
            logger.warning(f"Redis publish_trade failed: {e}")

    async def publish_alert(self, alert: dict) -> None:
        if not self.available or self.client is None:
            return
        try:
            await self.client.publish("channel:alerts", json.dumps(alert))
        except Exception as e:
            logger.warning(f"Redis publish_alert failed: {e}")

    async def publish_signal(self, symbol: str, signal: dict) -> None:
        if not self.available or self.client is None:
            return
        try:
            await self.client.publish(f"channel:signals:{symbol}", json.dumps(signal))
        except Exception as e:
            logger.warning(f"Redis publish_signal failed: {e}")

    # ── Rate limit counters (TTL: 60s) ─────────────────────────────────────

    async def increment_rate_limit(self, exchange: str) -> int:
        if not self.available or self.client is None:
            return 0
        try:
            key = f"ratelimit:{exchange}"
            pipe = self.client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"Redis rate_limit increment failed: {e}")
            return 0

    async def check_rate_limit(self, exchange: str, max_per_minute: int) -> bool:
        if not self.available or self.client is None:
            return True
        try:
            count = await self.client.get(f"ratelimit:{exchange}")
            return (int(count) if count else 0) < max_per_minute
        except Exception as e:
            logger.warning(f"Redis check_rate_limit failed: {e}")
            return True

    # ── Exchange status (TTL: 30s) ─────────────────────────────────────────

    async def set_exchange_status(self, exchange: str, is_online: bool) -> None:
        if not self.available or self.client is None:
            return
        try:
            await self.client.setex(f"exchange:status:{exchange}", 30, "1" if is_online else "0")
        except Exception as e:
            logger.warning(f"Redis set_exchange_status failed: {e}")

    async def get_exchange_status(self, exchange: str) -> Optional[bool]:
        if not self.available or self.client is None:
            return None
        try:
            data = await self.client.get(f"exchange:status:{exchange}")
            return bool(int(data)) if data else None
        except Exception as e:
            logger.warning(f"Redis get_exchange_status failed: {e}")
            return None


cache = CacheManager()
