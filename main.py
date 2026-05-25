"""
Trading system entry point.

Usage:
    python main.py --mode check          # verify config, DB, Redis
    python main.py --mode download       # download OHLCV data
    python main.py --mode train          # train ML model (local only)
    python main.py --mode backtest       # backtest on saved data
    python main.py --mode paper_trade    # live paper trading loop
    python main.py --mode dashboard      # start web dashboard only
    python main.py --mode all            # paper_trade + dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from src.utils.logger import logger
from src.utils.config import PAPER_TRADING, DATABASE_URL, REDIS_URL


async def mode_check() -> None:
    """Verify all connections and configuration."""
    from src.utils.database import check_connection, init_db
    from src.utils.cache import cache
    from src.exchanges.exchange_factory import ExchangeFactory
    from src.utils.config import ACTIVE_EXCHANGES, EXCHANGE_CREDENTIALS

    logger.info("=== System check ===")
    logger.info(f"PAPER_TRADING     : {PAPER_TRADING}")
    logger.info(f"DATABASE_URL      : {DATABASE_URL.split('?')[0]}")
    logger.info(f"REDIS_URL         : {REDIS_URL}")
    logger.info(f"ACTIVE_EXCHANGES  : {ACTIVE_EXCHANGES}")

    db_ok = await check_connection()
    redis_ok = await cache.connect()

    if db_ok:
        await init_db()

    # Test exchange connections
    exchanges = await ExchangeFactory.create_all(EXCHANGE_CREDENTIALS, ACTIVE_EXCHANGES)
    exchange_results: dict[str, str] = {}
    for name, exchange in exchanges.items():
        is_online, latency = await exchange.test_connection()
        exchange_results[name] = f"OK ({latency}ms)" if is_online else "FAILED"
    await ExchangeFactory.close_all(exchanges)

    logger.info("-" * 40)
    logger.info(f"Database : {'OK' if db_ok    else 'FAILED'}")
    logger.info(f"Redis    : {'OK' if redis_ok else 'UNAVAILABLE (degraded mode)'}")
    for name, status in exchange_results.items():
        logger.info(f"{name:<10}: {status}")
    logger.info("-" * 40)

    if not db_ok:
        logger.error("Database is required. Fix the connection before proceeding.")
        sys.exit(1)

    logger.info("System check complete.")
    await cache.close()


async def mode_download(exchanges: list[str]) -> None:
    from src.exchanges.exchange_factory import ExchangeFactory
    from src.data_pipeline.collector import OHLCVCollector
    from src.data_pipeline.aggregator import DataAggregator
    from src.utils.config import EXCHANGE_CREDENTIALS, SYMBOLS, TIMEFRAMES

    logger.info(f"Download mode — exchanges: {exchanges}")

    # Init exchanges
    active_creds = {k: v for k, v in EXCHANGE_CREDENTIALS.items() if k in exchanges}
    ex_map = await ExchangeFactory.create_all(active_creds, exchanges)

    if not ex_map:
        logger.error("No exchanges available. Check your .env credentials.")
        return

    collector = OHLCVCollector()
    results = await collector.collect_all(ex_map, SYMBOLS, TIMEFRAMES)

    # Print summary
    aggregator = DataAggregator()
    counts = await aggregator.get_candle_counts()
    if not counts.empty:
        logger.info("\n" + counts.to_string(index=False))

    await ExchangeFactory.close_all(ex_map)


async def mode_train() -> None:
    from src.models.trainer import WalkForwardTrainer
    from src.utils.config import SYMBOLS

    logger.info("=== ML Training (walk-forward validation) ===")
    logger.info("Loading data from DB...")

    trainer = WalkForwardTrainer()
    df = await trainer.load_all_symbols(symbols=SYMBOLS, timeframe="1h", exchange="binance")

    logger.info("Running walk-forward validation...")
    result = trainer.run(df)
    logger.info(result.summary())

    logger.info("Training final model on full dataset...")
    trainer.train_final(df, model_name="lgbm_final")
    logger.info("Training complete. Model saved to data/models/lgbm_final.pkl")


async def mode_backtest(from_date: str, to_date: str) -> None:
    from src.trading.backtest import run_backtest
    logger.info(f"Backtest mode: {from_date} -> {to_date}")
    metrics = await run_backtest(from_date=from_date, to_date=to_date)
    print(metrics)


async def mode_paper_trade() -> None:
    from src.ai_agent.agent import TradingAgent
    from src.models.signals import SignalGenerator
    from src.monitoring.signal_log import SignalLog
    from src.monitoring.telegram_bot import TelegramBot
    from src.trading.paper_trader import PaperTrader
    from src.utils.database import check_connection, init_db
    from src.utils.cache import cache
    from src.utils.config import TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY

    logger.info("Paper trading mode starting...")
    db_ok = await check_connection()
    if not db_ok:
        logger.error("Database required for paper trading.")
        return
    await init_db()
    await cache.connect()

    sg = SignalGenerator.from_file()
    trader = PaperTrader(sg)
    sig_log = SignalLog()

    bot: TelegramBot | None = None
    if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "your_token":
        bot = TelegramBot(trader)
        await bot.start()

    agent: TradingAgent | None = None
    _has_ai_key = (ANTHROPIC_API_KEY not in ("", "your_key", "your_anthropic_key") or
                   OPENAI_API_KEY not in ("", "your_openai_key"))
    if _has_ai_key:
        try:
            agent = TradingAgent(trader)
            logger.info(f"AI agent enabled (provider: {agent._provider})")
        except ValueError as e:
            logger.warning(f"AI agent disabled: {e}")

    try:
        await trader.run_live(bot=bot, agent=agent, sig_log=sig_log)
    finally:
        if bot:
            await bot.stop()
        await cache.close()


async def mode_dashboard() -> None:
    from src.monitoring.dashboard import run_dashboard
    logger.info("Dashboard mode starting...")
    await run_dashboard(trader=None, agent=None)


async def mode_all() -> None:
    from src.ai_agent.agent import TradingAgent
    from src.models.signals import SignalGenerator
    from src.monitoring.dashboard import run_dashboard
    from src.monitoring.signal_log import SignalLog
    from src.monitoring.telegram_bot import TelegramBot
    from src.trading.paper_trader import PaperTrader
    from src.utils.database import check_connection, init_db
    from src.utils.cache import cache
    from src.utils.config import TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY

    db_ok = await check_connection()
    if not db_ok:
        logger.error("Database required.")
        return
    await init_db()
    await cache.connect()

    sg = SignalGenerator.from_file()
    trader = PaperTrader(sg)
    sig_log = SignalLog()

    bot: TelegramBot | None = None
    if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "your_token":
        bot = TelegramBot(trader)
        await bot.start()

    agent: TradingAgent | None = None
    _has_ai_key = (ANTHROPIC_API_KEY not in ("", "your_key", "your_anthropic_key") or
                   OPENAI_API_KEY not in ("", "your_openai_key"))
    if _has_ai_key:
        try:
            agent = TradingAgent(trader)
            logger.info(f"AI agent enabled (provider: {agent._provider})")
        except ValueError as e:
            logger.warning(f"AI agent disabled: {e}")

    try:
        await asyncio.gather(
            trader.run_live(bot=bot, agent=agent, sig_log=sig_log),
            run_dashboard(trader=trader, agent=agent, signal_log=sig_log),
        )
    finally:
        if bot:
            await bot.stop()
        await cache.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto Trading System")
    parser.add_argument(
        "--mode",
        choices=["check", "download", "train", "backtest", "paper_trade", "dashboard", "all"],
        default="check",
    )
    parser.add_argument("--exchanges", default="binance", help="Comma-separated exchange list")
    parser.add_argument("--from", dest="from_date", default="2023-01-01")
    parser.add_argument("--to",   dest="to_date",   default="2023-12-31")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    match args.mode:
        case "check":
            await mode_check()
        case "download":
            await mode_download(args.exchanges.split(","))
        case "train":
            await mode_train()
        case "backtest":
            await mode_backtest(args.from_date, args.to_date)
        case "paper_trade":
            await mode_paper_trade()
        case "dashboard":
            await mode_dashboard()
        case "all":
            await mode_all()


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Handle SIGTERM (Docker stop) and SIGINT (Ctrl+C) gracefully."""
    def _shutdown():
        logger.info("Shutdown signal received — cancelling tasks...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, OSError):
            # Windows doesn't support add_signal_handler for SIGTERM
            pass


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)
    try:
        loop.run_until_complete(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Trading system stopped.")
    finally:
        loop.close()
