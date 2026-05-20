from __future__ import annotations

from src.exchanges.base_exchange import BaseExchange
from src.exchanges.binance_exchange import BinanceExchange
from src.exchanges.bybit_exchange import BybitExchange
from src.exchanges.kraken_exchange import KrakenExchange
from src.exchanges.okx_exchange import OkxExchange
from src.utils.logger import logger


class ExchangeFactory:

    @staticmethod
    def create(name: str, api_key: str = "", secret: str = "", passphrase: str = "") -> BaseExchange:
        match name.lower():
            case "binance":
                return BinanceExchange(api_key, secret)
            case "bybit":
                return BybitExchange(api_key, secret)
            case "kraken":
                return KrakenExchange(api_key, secret)
            case "okx":
                return OkxExchange(api_key, secret, passphrase)
            case _:
                raise ValueError(f"Unknown exchange: '{name}'. Supported: binance, bybit, kraken, okx")

    @staticmethod
    async def create_all(
        credentials: dict[str, dict],
        active_exchanges: list[str],
    ) -> dict[str, BaseExchange]:
        """Create and verify connections for all active exchanges."""
        exchanges: dict[str, BaseExchange] = {}

        for name in active_exchanges:
            creds = credentials.get(name, {})
            try:
                exchange = ExchangeFactory.create(name, **creds)
                exchanges[name] = exchange
                logger.info(f"Exchange '{name}' initialized")
            except Exception as e:
                logger.error(f"Failed to initialize exchange '{name}': {e}")

        return exchanges

    @staticmethod
    async def close_all(exchanges: dict[str, BaseExchange]) -> None:
        for name, exchange in exchanges.items():
            try:
                await exchange.close()
                logger.debug(f"Exchange '{name}' closed")
            except Exception as e:
                logger.warning(f"Error closing exchange '{name}': {e}")
