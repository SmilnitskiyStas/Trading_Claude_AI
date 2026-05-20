from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from src.data_pipeline.aggregator import DataAggregator
from src.data_pipeline.processor import FeatureProcessor
from src.models.signals import SignalGenerator
from src.monitoring.metrics import PerformanceMetrics
from src.strategy.risk_manager import RiskManager
from src.trading.paper_trader import PaperTrader
from src.utils.config import INITIAL_BALANCE, PRIMARY_EXCHANGE, SYMBOLS as _ALL_SYMBOLS
from src.utils.logger import logger

_SYMBOL_ID_MAP = {s: i for i, s in enumerate(_ALL_SYMBOLS)}


async def run_backtest(
    from_date: str,
    to_date: str,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    exchange: str = PRIMARY_EXCHANGE,
    model_path: str | None = None,
) -> PerformanceMetrics:
    """
    Load OHLCV data, compute signals with a trained model, and run PaperTrader.
    """
    active_symbols = symbols or _ALL_SYMBOLS
    logger.info(f"Backtest {from_date} -> {to_date} | {exchange} | {active_symbols}")

    sg = SignalGenerator.from_file(model_path)
    processor = FeatureProcessor()
    aggregator = DataAggregator()

    from_dt = pd.Timestamp(from_date, tz="UTC")
    to_dt   = pd.Timestamp(to_date,   tz="UTC")

    data: dict[str, pd.DataFrame] = {}

    for sym in active_symbols:
        df = await aggregator.load_ohlcv(exchange, sym, timeframe)
        if df.empty:
            logger.warning(f"No data for {sym}, skipping")
            continue

        # Filter date range using DatetimeIndex directly
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[(df.index >= from_dt) & (df.index <= to_dt)]
        if len(df) < 200:
            logger.warning(f"{sym}: only {len(df)} bars in range, skipping")
            continue

        # Apply feature engineering
        processed = processor.process(df)
        processed.dropna(inplace=True)
        if len(processed) < 50:
            logger.warning(f"{sym}: insufficient rows after processing, skipping")
            continue

        # Add symbol_id to match training features
        processed["symbol_id"] = _SYMBOL_ID_MAP.get(sym, 0)

        # Generate signals for the full slice
        signal_df = sg.generate_batch(processed, symbol=sym, exchange=exchange)
        data[sym] = signal_df
        logger.info(f"{sym}: {len(signal_df)} bars, signals generated")

    if not data:
        raise RuntimeError("No data available for backtest. Run --mode download first.")

    rm = RiskManager(initial_balance=INITIAL_BALANCE)
    trader = PaperTrader(sg, rm, initial_balance=INITIAL_BALANCE, exchange=exchange)
    metrics = trader.run_backtest(data, symbols=list(data.keys()))
    return metrics
