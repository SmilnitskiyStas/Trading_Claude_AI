from __future__ import annotations

import pandas as pd
import numpy as np

import asyncpg

from src.utils.config import DATABASE_URL, ACTIVE_EXCHANGES
from src.utils.logger import logger


def _asyncpg_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


class DataAggregator:

    def __init__(self, database_url: str = DATABASE_URL):
        self._db_url = _asyncpg_url(database_url)

    # ── DB load ────────────────────────────────────────────────────────────

    async def load_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV from DB into a datetime-indexed DataFrame."""
        conn = await asyncpg.connect(self._db_url)
        try:
            query = (
                "SELECT timestamp, open, high, low, close, volume "
                "FROM ohlcv_data "
                "WHERE exchange=$1 AND symbol=$2 AND timeframe=$3 "
                "ORDER BY timestamp"
            )
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = await conn.fetch(query, exchange, symbol, timeframe)
        finally:
            await conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        df = df.astype(float)
        return df

    # ── Multi-exchange merge ───────────────────────────────────────────────

    async def get_aggregated(
        self,
        symbol: str,
        timeframe: str,
        exchanges: list[str] = ACTIVE_EXCHANGES,
        primary: str = "binance",
    ) -> pd.DataFrame:
        """
        Merge OHLCV from multiple exchanges.
        Returns primary exchange data enriched with cross-exchange features:
          - close_vwap    : volume-weighted average close across exchanges
          - binance_bybit_spread : relative price spread (if both available)
        """
        dfs: dict[str, pd.DataFrame] = {}
        for ex in exchanges:
            df = await self.load_ohlcv(ex, symbol, timeframe)
            if not df.empty:
                dfs[ex] = df
                logger.debug(f"Loaded {len(df)} rows from {ex}/{symbol}/{timeframe}")

        if not dfs:
            logger.warning(f"No data found for {symbol}/{timeframe}")
            return pd.DataFrame()

        base_ex = primary if primary in dfs else next(iter(dfs))
        result = dfs[base_ex].copy()

        if len(dfs) > 1:
            # VWAP across exchanges
            closes = pd.concat([dfs[ex]["close"].rename(ex) for ex in dfs], axis=1)
            volumes = pd.concat([dfs[ex]["volume"].rename(ex) for ex in dfs], axis=1)
            total_vol = volumes.sum(axis=1).replace(0, np.nan)
            result["close_vwap"] = (closes * volumes).sum(axis=1) / total_vol

            # Binance–Bybit spread
            if "binance" in dfs and "bybit" in dfs:
                ref = dfs["binance"]["close"].reindex(result.index)
                other = dfs["bybit"]["close"].reindex(result.index)
                result["binance_bybit_spread"] = ((ref - other) / ref).fillna(0)
        else:
            result["close_vwap"] = result["close"]
            result["binance_bybit_spread"] = 0.0

        return result

    # ── Convenience ───────────────────────────────────────────────────────

    async def get_candle_counts(self) -> pd.DataFrame:
        """Return a summary table of how many candles are stored per series."""
        conn = await asyncpg.connect(self._db_url)
        try:
            rows = await conn.fetch(
                """
                SELECT exchange, symbol, timeframe,
                       COUNT(*) AS candles,
                       MIN(timestamp) AS first_ts,
                       MAX(timestamp) AS last_ts
                FROM ohlcv_data
                GROUP BY exchange, symbol, timeframe
                ORDER BY exchange, symbol, timeframe
                """
            )
        finally:
            await conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["exchange", "symbol", "timeframe", "candles", "first_ts", "last_ts"])
        df["from"] = pd.to_datetime(df["first_ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        df["to"]   = pd.to_datetime(df["last_ts"],  unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        return df[["exchange", "symbol", "timeframe", "candles", "from", "to"]]
