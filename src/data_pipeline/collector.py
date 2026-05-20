from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import asyncpg
from tqdm import tqdm

from src.exchanges.base_exchange import BaseExchange
from src.utils.config import DATABASE_URL, SYMBOLS, TIMEFRAMES
from src.utils.logger import logger

# Default history start: 3 years back from 2022-01-01
HISTORY_START_MS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
CANDLES_PER_REQUEST = 500

_TF_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def tf_to_ms(timeframe: str) -> int:
    return int(timeframe[:-1]) * _TF_MS[timeframe[-1]]


def _asyncpg_url(url: str) -> str:
    """Strip SQLAlchemy driver prefix for asyncpg."""
    return url.replace("postgresql+asyncpg://", "postgresql://")


class OHLCVCollector:

    def __init__(self, database_url: str = DATABASE_URL):
        self._db_url = _asyncpg_url(database_url)

    # ── DB helpers ─────────────────────────────────────────────────────────

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._db_url)

    async def _get_latest_ts(
        self, conn: asyncpg.Connection, exchange: str, symbol: str, timeframe: str
    ) -> int:
        """Return timestamp of the next missing candle (or HISTORY_START_MS)."""
        ts = await conn.fetchval(
            "SELECT MAX(timestamp) FROM ohlcv_data "
            "WHERE exchange=$1 AND symbol=$2 AND timeframe=$3",
            exchange, symbol, timeframe,
        )
        return (ts + tf_to_ms(timeframe)) if ts else HISTORY_START_MS

    async def _save_candles(
        self,
        conn: asyncpg.Connection,
        candles: list,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> int:
        rows = [
            (exchange, symbol, timeframe,
             int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]))
            for c in candles
            if len(c) >= 6 and all(v is not None for v in c[:6])
        ]
        if not rows:
            return 0
        await conn.executemany(
            """
            INSERT INTO ohlcv_data
                (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (exchange, symbol, timeframe, timestamp) DO NOTHING
            """,
            rows,
        )
        return len(rows)

    # ── Core download logic ────────────────────────────────────────────────

    async def _fetch_symbol(
        self,
        exchange: BaseExchange,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Download and store all missing candles for one series. Returns candles saved."""
        conn = await self._connect()
        try:
            since = await self._get_latest_ts(conn, exchange.name, symbol, timeframe)
        finally:
            await conn.close()

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        total_saved = 0

        while since < now_ms:
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=CANDLES_PER_REQUEST
                )
            except Exception as e:
                logger.error(f"[{exchange.name}] {symbol}/{timeframe} fetch failed: {e}")
                break

            if not candles:
                break

            conn = await self._connect()
            try:
                saved = await self._save_candles(conn, candles, exchange.name, symbol, timeframe)
            finally:
                await conn.close()

            total_saved += saved
            last_ts = candles[-1][0]

            if last_ts <= since or len(candles) < CANDLES_PER_REQUEST:
                break

            since = last_ts + tf_to_ms(timeframe)

        return total_saved

    # ── Public API ─────────────────────────────────────────────────────────

    async def collect_exchange(
        self,
        exchange: BaseExchange,
        symbols: list[str] = SYMBOLS,
        timeframes: list[str] = TIMEFRAMES,
    ) -> dict:
        """Collect all symbols/timeframes for one exchange sequentially."""
        total = 0
        combos = [(s, tf) for tf in timeframes for s in symbols]

        for symbol, timeframe in tqdm(combos, desc=f"[{exchange.name}]", unit="series"):
            saved = await self._fetch_symbol(exchange, symbol, timeframe)
            total += saved
            if saved:
                logger.info(f"[{exchange.name}] {symbol}/{timeframe}: +{saved} candles")

        logger.info(f"[{exchange.name}] Done. Total new candles: {total}")
        return {"exchange": exchange.name, "candles_saved": total}

    async def collect_all(
        self,
        exchanges: dict[str, BaseExchange],
        symbols: list[str] = SYMBOLS,
        timeframes: list[str] = TIMEFRAMES,
    ) -> list[dict]:
        """
        Collect from all exchanges.
        Fast exchanges (binance, bybit, okx) run concurrently.
        Kraken runs separately due to strict rate limits.
        """
        fast = {k: v for k, v in exchanges.items() if k != "kraken"}
        kraken = exchanges.get("kraken")

        tasks = [
            self.collect_exchange(ex, symbols, timeframes)
            for ex in fast.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for name, r in zip(fast.keys(), results):
            if isinstance(r, Exception):
                logger.error(f"[{name}] collection error: {r}")
            else:
                all_results.append(r)

        if kraken:
            logger.info("[kraken] Starting (sequential, 1 req/s)...")
            r = await self.collect_exchange(kraken, symbols, timeframes)
            all_results.append(r)

        total = sum(r["candles_saved"] for r in all_results)
        logger.info(f"Collection complete. Total new candles across all exchanges: {total}")
        return all_results

    async def update(
        self,
        exchanges: dict[str, BaseExchange],
        symbols: list[str] = SYMBOLS,
        timeframes: list[str] = TIMEFRAMES,
    ) -> list[dict]:
        """Incremental update — only fetches candles newer than what's stored."""
        return await self.collect_all(exchanges, symbols, timeframes)
