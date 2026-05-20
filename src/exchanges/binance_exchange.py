from __future__ import annotations

import ccxt.async_support as ccxt

from src.exchanges.base_exchange import BaseExchange


class BinanceExchange(BaseExchange):
    """Binance spot exchange. Primary source — highest liquidity."""

    def __init__(self, api_key: str = "", secret: str = "") -> None:
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
                "fetchCurrencies": False,   # skip authenticated SAPI call during loadMarkets
            },
        })
        super().__init__(exchange)

    @property
    def name(self) -> str:
        return "binance"

    @property
    def rate_limit(self) -> int:
        return 1200
