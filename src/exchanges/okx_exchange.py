from __future__ import annotations

import ccxt.async_support as ccxt

from src.exchanges.base_exchange import BaseExchange


class OkxExchange(BaseExchange):
    """OKX spot exchange. Good API, 300 req/min."""

    def __init__(self, api_key: str = "", secret: str = "", passphrase: str = "") -> None:
        exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })
        super().__init__(exchange)

    @property
    def name(self) -> str:
        return "okx"

    @property
    def rate_limit(self) -> int:
        return 300
