from __future__ import annotations

import ccxt.async_support as ccxt

from src.exchanges.base_exchange import BaseExchange


class BybitExchange(BaseExchange):
    """Bybit spot + futures exchange. Good for funding rate data."""

    def __init__(self, api_key: str = "", secret: str = "") -> None:
        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })
        super().__init__(exchange)

    @property
    def name(self) -> str:
        return "bybit"

    @property
    def rate_limit(self) -> int:
        return 600
