from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import ccxt.async_support as ccxt

from src.utils.logger import logger


class BaseExchange(ABC):
    """Unified async interface for all exchanges."""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0      # seconds, doubled on each retry

    def __init__(self, ccxt_exchange: ccxt.Exchange) -> None:
        self._exchange = ccxt_exchange

    # ── Abstract properties ────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def rate_limit(self) -> int:
        """Max requests per minute."""
        ...

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._exchange.close()

    # ── Retry wrapper ──────────────────────────────────────────────────────

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call ccxt method with exponential-backoff retry."""
        fn = getattr(self._exchange, method)
        delay = self.RETRY_BASE_DELAY

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                wait = delay * 2
                logger.warning(f"[{self.name}] Rate limit hit, waiting {wait:.1f}s — {e}")
                await asyncio.sleep(wait)
                delay *= 2
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if attempt == self.MAX_RETRIES:
                    logger.error(f"[{self.name}] {method} failed after {self.MAX_RETRIES} retries: {e}")
                    raise
                logger.warning(f"[{self.name}] {method} attempt {attempt} failed: {e}. Retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay *= 2
            except ccxt.ExchangeError as e:
                logger.error(f"[{self.name}] Exchange error in {method}: {e}")
                raise

    # ── Public API ─────────────────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list]:
        """Return list of [timestamp, open, high, low, close, volume]."""
        return await self._call("fetch_ohlcv", symbol, timeframe, since, limit)

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._call("fetch_ticker", symbol)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._call("fetch_order_book", symbol, limit)

    async def fetch_funding_rate(self, symbol: str) -> dict | None:
        """Funding rate for futures (not available on all exchanges)."""
        if not self._exchange.has.get("fetchFundingRate"):
            return None
        try:
            return await self._call("fetch_funding_rate", symbol)
        except Exception:
            return None

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        raise NotImplementedError(
            "Live trading is not enabled. Set PAPER_TRADING=false and implement live_trader.py"
        )

    # ── Diagnostics ────────────────────────────────────────────────────────

    async def test_connection(self) -> tuple[bool, int]:
        """
        Ping exchange using a public endpoint (1 OHLCV candle).
        Returns (is_online, latency_ms). Works without API keys.
        """
        start = time.monotonic()
        try:
            # fetch_ohlcv is always a public endpoint — no auth needed
            result = await self._exchange.fetch_ohlcv("BTC/USDT", "1h", limit=1)
            latency = int((time.monotonic() - start) * 1000)
            if result:
                logger.info(f"[{self.name}] Connected. Latency: {latency}ms")
                return True, latency
            raise ValueError("Empty response from exchange")
        except Exception as e:
            latency = int((time.monotonic() - start) * 1000)
            logger.error(f"[{self.name}] Connection FAILED: {e}")
            return False, latency

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
