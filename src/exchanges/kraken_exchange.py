from __future__ import annotations

import asyncio
from typing import Any

import ccxt.async_support as ccxt

from src.exchanges.base_exchange import BaseExchange
from src.utils.logger import logger

_KRAKEN_MIN_DELAY = 1.0  # seconds between requests — 60 req/min hard limit


class KrakenExchange(BaseExchange):
    """
    Kraken exchange. Oldest history (BTC/ETH from 2013).
    IMPORTANT: 60 req/min limit — mandatory 1s delay between calls.
    """

    def __init__(self, api_key: str = "", secret: str = "") -> None:
        exchange = ccxt.kraken({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })
        super().__init__(exchange)
        self._last_call: float = 0.0

    @property
    def name(self) -> str:
        return "kraken"

    @property
    def rate_limit(self) -> int:
        return 60

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Enforce minimum delay between Kraken requests to avoid bans."""
        import time
        elapsed = time.monotonic() - self._last_call
        if elapsed < _KRAKEN_MIN_DELAY:
            await asyncio.sleep(_KRAKEN_MIN_DELAY - elapsed)
        result = await super()._call(method, *args, **kwargs)
        self._last_call = asyncio.get_event_loop().time()
        return result
