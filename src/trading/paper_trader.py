from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.models.signals import SignalGenerator
from src.monitoring.metrics import PerformanceMetrics, calculate_metrics
from src.strategy.risk_manager import RiskManager
from src.utils.config import INITIAL_BALANCE, PAPER_TRADING, PRIMARY_EXCHANGE, SYMBOLS
from src.utils.database import get_session, PortfolioSnapshot
from src.utils.logger import logger

if False:  # TYPE_CHECKING
    from src.monitoring.telegram_bot import TelegramBot
    from src.ai_agent.agent import TradingAgent

assert PAPER_TRADING, "PaperTrader must only run with PAPER_TRADING=true"

TRADING_FEE = 0.001   # 0.1% taker fee per side


@dataclass
class Position:
    symbol: str
    direction: int             # 1=long, -1=short
    entry_price: float
    quantity: float            # in base asset
    notional_usd: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    entry_fee: float
    best_price: float = 0.0     # trailing SL: high-watermark (long) or low-watermark (short)


@dataclass
class ClosedTrade:
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    quantity: float
    notional_usd: float
    pnl_usd: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str            # "signal"|"stop_loss"|"take_profit"
    holding_hours: float


class PaperTrader:
    """
    Simulates trading using historical OHLCV data and ML signals.

    - Processes candles chronologically, one bar at a time
    - Checks stop-loss and take-profit against high/low of each candle
    - Opens new positions on valid signals at close price + slippage
    - Tracks equity curve and all closed trades
    """

    SLIPPAGE = 0.0005   # 0.05% fill slippage

    def __init__(
        self,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager | None = None,
        initial_balance: float = INITIAL_BALANCE,
        exchange: str = PRIMARY_EXCHANGE,
    ) -> None:
        self.sg = signal_generator
        self.rm = risk_manager or RiskManager(initial_balance=initial_balance)
        self.exchange = exchange
        self._cash = initial_balance
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._equity_curve: list[float] = [initial_balance]
        self._equity_timestamps: list[datetime] = []

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._cash + sum(
            p.notional_usd for p in self._positions.values()
        )

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._closed_trades)

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    # ── Core simulation ────────────────────────────────────────────────────

    def run_backtest(
        self,
        data: dict[str, pd.DataFrame],
        symbols: list[str] | None = None,
    ) -> PerformanceMetrics:
        """
        Run a vectorised backtest over prepared DataFrames.

        data: { symbol: df } where df has columns [open, high, low, close, volume,
               signal, confidence] and a DatetimeIndex (already processed by SignalGenerator)
        """
        target_symbols = symbols or list(data.keys())
        logger.info(f"Backtest starting: {len(target_symbols)} symbols")

        # Merge all symbol bars into one chronological timeline
        events: list[tuple[datetime, str, pd.Series]] = []
        for sym in target_symbols:
            if sym not in data or data[sym].empty:
                continue
            df = data[sym]
            for ts, row in df.iterrows():
                events.append((ts, sym, row))

        events.sort(key=lambda x: x[0])

        prev_day = None
        for ts, sym, row in events:
            self._process_bar(sym, row, ts)

            cur_day = ts.date()
            if cur_day != prev_day:
                self._equity_curve.append(self.equity)
                self._equity_timestamps.append(ts)
                self.rm.update_equity(self.equity, ts)
                prev_day = cur_day

        # Force-close all remaining positions at last known price
        for sym, pos in list(self._positions.items()):
            last_close = data[sym]["close"].iloc[-1]
            last_ts = data[sym].index[-1]
            self._close_position(sym, last_close, last_ts, "end_of_backtest")

        metrics = self._compute_metrics()
        logger.info("\n" + str(metrics))
        return metrics

    async def run_live(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        poll_interval_seconds: int = 60,
        bot: "TelegramBot | None" = None,
        agent: "TradingAgent | None" = None,
    ) -> None:
        """
        Live paper-trading loop. Runs indefinitely until cancelled.
        Polls for new candles and generates signals.
        """
        active = symbols or SYMBOLS
        logger.info(f"Live paper-trade starting: {active} [{timeframe}]")
        _prev_halted = False
        _agent_signals: dict[str, dict] = {}  # last agent override per symbol

        # Background OHLCV refresh — keeps DB current without manual uploads
        asyncio.create_task(self._refresh_ohlcv_loop(active, timeframe))

        while True:
            try:
                # Check if Telegram sent /stop
                if bot and bot.stop_requested:
                    logger.info("Stop requested via Telegram — shutting down")
                    break

                # Notify on new halt
                if self.rm.is_halted and not _prev_halted:
                    _prev_halted = True
                    if bot:
                        await bot.notify_risk_halt("Drawdown or daily loss limit reached")

                if not self.rm.is_halted:
                    _prev_halted = False

                # Collect ML signals for all symbols first
                ml_signals: dict[str, dict] = {}
                for sym in active:
                    sig = await self.sg.latest_signal(sym, timeframe=timeframe, exchange=self.exchange)
                    ml_signals[sym] = sig
                    logger.debug(f"{sym}: ml={sig['action']} conf={sig['confidence']:.3f}")

                # Ask agent (uses Redis cache — actual Claude call happens once/hour)
                if agent and agent.enabled:
                    try:
                        _agent_signals = await agent.analyze(ml_signals, symbols=active)
                    except Exception as exc:
                        logger.warning(f"Agent error: {exc}")
                elif agent and not agent.enabled:
                    _agent_signals = {}   # clear overrides when disabled

                for sym in active:
                    # Always check SL/TP on existing positions (even when halted)
                    if sym in self._positions:
                        await self._check_live_sl_tp(sym, bot=bot)
                        continue
                    if self.rm.is_halted:
                        logger.warning("Risk manager halted — skipping new signals")
                        break

                    # Merge: agent overrides ML only when it disagrees with "hold"
                    final_sig = dict(ml_signals[sym])
                    agent_dec = _agent_signals.get(sym, {})
                    if agent_dec.get("action") in ("buy", "sell"):
                        # Only act if both ML and agent agree on direction
                        if agent_dec["action"] == final_sig.get("action"):
                            # Boost confidence when both agree
                            final_sig["confidence"] = min(
                                1.0,
                                final_sig.get("confidence", 0) * 1.1,
                            )
                        else:
                            # Disagree → stay conservative (hold)
                            final_sig["action"] = "hold"
                            final_sig["signal"] = 0

                    if final_sig.get("action") in ("buy", "sell"):
                        await self._open_live_position(sym, final_sig, bot=bot)

                await self._save_snapshot()
                await asyncio.sleep(poll_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Paper trader stopped.")
                break
            except Exception as exc:
                logger.error(f"Paper trader error: {exc}")
                await asyncio.sleep(10)

    # ── Bar processing ─────────────────────────────────────────────────────

    def _process_bar(self, sym: str, row: pd.Series, ts: datetime) -> None:
        # First: check SL/TP on existing position
        if sym in self._positions:
            pos = self._positions[sym]
            closed = self._check_sl_tp(sym, pos, row, ts)
            if closed:
                return

        # Second: open new position on signal
        signal = int(row.get("signal", 0))
        confidence = float(row.get("confidence", 0.0))
        if signal == 0 or sym in self._positions:
            return

        direction = signal  # 1 or -1
        entry_price = float(row["close"]) * (1 + self.SLIPPAGE * direction)

        # Simple win_rate/win_loss estimates from last 50 closed trades
        wr, avg_w, avg_l = self._rolling_stats(sym)

        sizing = self.rm.size_position(sym, direction, entry_price, wr, avg_w, avg_l)
        if sizing is None:
            return

        quantity = sizing.position_size_usd / entry_price
        fee = sizing.position_size_usd * TRADING_FEE

        if self._cash < sizing.position_size_usd + fee:
            return

        self._cash -= sizing.position_size_usd + fee
        pos = Position(
            symbol=sym,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            notional_usd=sizing.position_size_usd,
            stop_loss=sizing.stop_loss_price,
            take_profit=sizing.take_profit_price,
            entry_time=ts,
            entry_fee=fee,
            best_price=entry_price,
        )
        self._positions[sym] = pos
        self.rm.register_open(sym, sizing.position_size_usd)

    def _check_sl_tp(
        self, sym: str, pos: Position, row: pd.Series, ts: datetime,
        bot: "TelegramBot | None" = None,
    ) -> bool:
        high = float(row["high"])
        low  = float(row["low"])

        # Safety: initialise best_price if position was opened before trailing-SL feature
        if pos.best_price == 0.0:
            pos.best_price = pos.entry_price

        hit_sl = hit_tp = False
        if pos.direction > 0:  # long
            hit_sl = low  <= pos.stop_loss
            hit_tp = high >= pos.take_profit
        else:                  # short
            hit_sl = high >= pos.stop_loss
            hit_tp = low  <= pos.take_profit

        if hit_tp:
            self._close_position(sym, pos.take_profit, ts, "take_profit", bot=bot)
            return True
        if hit_sl:
            self._close_position(sym, pos.stop_loss, ts, "stop_loss", bot=bot)
            return True

        # Position still open — ratchet trailing stop for next bar
        self._update_trailing_stop(sym, pos, high, low)
        return False

    def _update_trailing_stop(
        self, sym: str, pos: Position, high: float, low: float
    ) -> None:
        """Move stop-loss toward current price as the trade moves in our favour."""
        trail_pct = self.rm.stop_loss
        if pos.direction > 0:          # long: trail below high watermark
            if high > pos.best_price:
                pos.best_price = high
                new_sl = round(pos.best_price * (1 - trail_pct), 8)
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
                    logger.debug(f"Trailing SL {sym} → {pos.stop_loss:.4f}")
        else:                          # short: trail above low watermark
            if low < pos.best_price:
                pos.best_price = low
                new_sl = round(pos.best_price * (1 + trail_pct), 8)
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl
                    logger.debug(f"Trailing SL {sym} → {pos.stop_loss:.4f}")

    def _close_position(
        self,
        sym: str,
        exit_price: float,
        ts: datetime,
        reason: str,
        bot: "TelegramBot | None" = None,
    ) -> None:
        pos = self._positions.pop(sym, None)
        if pos is None:
            return

        self.rm.register_close(sym)
        exit_fee = pos.notional_usd * TRADING_FEE
        proceeds = pos.notional_usd + pos.notional_usd * (
            (exit_price - pos.entry_price) / pos.entry_price * pos.direction
        )
        net_proceeds = proceeds - exit_fee
        self._cash += net_proceeds

        pnl_usd = net_proceeds - pos.notional_usd - pos.entry_fee
        pnl_pct = pnl_usd / pos.notional_usd

        holding_h = (ts - pos.entry_time).total_seconds() / 3600

        trade = ClosedTrade(
            symbol=sym,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            notional_usd=pos.notional_usd,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=ts,
            exit_reason=reason,
            holding_hours=holding_h,
        )
        self._closed_trades.append(trade)

        dir_str = "LONG" if pos.direction > 0 else "SHORT"
        logger.debug(
            f"CLOSE {sym} {reason:12s} | "
            f"pnl={pnl_usd:+.2f} ({pnl_pct:+.2%}) | "
            f"hold={holding_h:.1f}h"
        )
        if bot:
            asyncio.create_task(
                bot.notify_trade_closed(sym, dir_str, pnl_usd, pnl_pct, reason)
            )

    # ── Live open ──────────────────────────────────────────────────────────

    async def _open_live_position(
        self, sym: str, sig: dict, bot: "TelegramBot | None" = None
    ) -> None:
        from src.data_pipeline.aggregator import DataAggregator
        agg = DataAggregator()
        df = await agg.load_ohlcv(self.exchange, sym, "1h", limit=1)
        if df.empty:
            return
        entry_price = float(df["close"].iloc[-1]) * (1 + self.SLIPPAGE * sig["signal"])
        direction = sig["signal"]
        wr, avg_w, avg_l = self._rolling_stats(sym)
        sizing = self.rm.size_position(sym, direction, entry_price, wr, avg_w, avg_l)
        if sizing is None:
            return
        quantity = sizing.position_size_usd / entry_price
        fee = sizing.position_size_usd * TRADING_FEE
        if self._cash < sizing.position_size_usd + fee:
            return
        self._cash -= sizing.position_size_usd + fee
        pos = Position(
            symbol=sym, direction=direction,
            entry_price=entry_price, quantity=quantity,
            notional_usd=sizing.position_size_usd,
            stop_loss=sizing.stop_loss_price,
            take_profit=sizing.take_profit_price,
            entry_time=datetime.now(timezone.utc),
            entry_fee=fee,
            best_price=entry_price,
        )
        self._positions[sym] = pos
        self.rm.register_open(sym, sizing.position_size_usd)
        dir_str = "LONG" if direction > 0 else "SHORT"
        logger.info(
            f"OPEN {sym} {dir_str} | "
            f"entry={entry_price:.4f} | size={sizing.position_size_usd:.2f} USD | "
            f"sl={sizing.stop_loss_price:.4f} tp={sizing.take_profit_price:.4f}"
        )
        if bot:
            await bot.notify_trade_opened(
                sym, dir_str, entry_price,
                sizing.position_size_usd,
                sizing.stop_loss_price,
                sizing.take_profit_price,
            )

    # ── Live SL/TP check ──────────────────────────────────────────────────

    async def _check_live_sl_tp(
        self, sym: str, bot: "TelegramBot | None" = None
    ) -> None:
        """
        Load the latest closed candle and check stop-loss / take-profit
        for an open live position. Called every poll cycle.
        """
        if sym not in self._positions:
            return
        try:
            from src.data_pipeline.aggregator import DataAggregator
            agg = DataAggregator()
            df = await agg.load_ohlcv(self.exchange, sym, "1h", limit=2)
            if df.empty:
                return
            row = df.iloc[-1]
            ts  = datetime.now(timezone.utc)
            pos = self._positions[sym]
            self._check_sl_tp(sym, pos, row, ts, bot=bot)
        except Exception as exc:
            logger.warning(f"Live SL/TP check failed for {sym}: {exc}")

    # ── Background OHLCV refresh ───────────────────────────────────────────

    async def _refresh_ohlcv_loop(
        self,
        symbols: list[str],
        timeframe: str,
        interval_hours: int = 4,
    ) -> None:
        """
        Downloads fresh OHLCV candles every N hours so the DB stays current.
        Uses ACTIVE_EXCHANGES from config (set PRIMARY_EXCHANGE=kraken on VPS).
        Runs as a background task — failures are logged but never crash the trader.
        """
        from src.data_pipeline.collector import OHLCVCollector
        from src.exchanges.exchange_factory import ExchangeFactory
        from src.utils.config import ACTIVE_EXCHANGES, EXCHANGE_CREDENTIALS

        # First refresh after 5 min (let system fully start first)
        await asyncio.sleep(300)

        while True:
            try:
                ex_map = await ExchangeFactory.create_all(
                    EXCHANGE_CREDENTIALS, ACTIVE_EXCHANGES
                )
                if ex_map:
                    collector = OHLCVCollector()
                    await collector.collect_all(ex_map, symbols, [timeframe])
                    await ExchangeFactory.close_all(ex_map)
                    logger.info(
                        f"Background OHLCV refresh complete "
                        f"({', '.join(ex_map.keys())}, {timeframe})"
                    )
                else:
                    logger.warning("Background OHLCV refresh: no exchanges available")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Background OHLCV refresh failed: {exc}")

            await asyncio.sleep(interval_hours * 3600)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _rolling_stats(self, symbol: str, lookback: int = 50) -> tuple[float, float, float]:
        """Returns (win_rate, avg_win, avg_loss) from recent trades for this symbol."""
        recent = [t for t in self._closed_trades[-lookback:] if t.symbol == symbol]
        if len(recent) < 5:
            # Default conservative estimate
            return 0.45, 0.025, 0.015
        wins = [t.pnl_pct for t in recent if t.pnl_pct > 0]
        losses = [abs(t.pnl_pct) for t in recent if t.pnl_pct <= 0]
        wr = len(wins) / len(recent)
        avg_w = sum(wins) / len(wins) if wins else 0.025
        avg_l = sum(losses) / len(losses) if losses else 0.015
        return wr, avg_w, avg_l

    def _compute_metrics(self) -> PerformanceMetrics:
        trade_returns = [t.pnl_pct for t in self._closed_trades]
        holding_hours = [t.holding_hours for t in self._closed_trades]
        # equity_curve is daily (one entry per calendar day in backtest)
        return calculate_metrics(
            equity_curve=self._equity_curve,
            trade_returns=trade_returns,
            holding_hours=holding_hours,
            periods_per_year=365,
        )

    async def _save_snapshot(self) -> None:
        try:
            positions_val = sum(p.notional_usd for p in self._positions.values())
            async with get_session() as session:
                snap = PortfolioSnapshot(
                    timestamp=datetime.utcnow(),
                    total_value=self.equity,
                    cash_balance=self._cash,
                    positions_value=positions_val,
                    daily_pnl=self.rm._state.daily_pnl,
                    drawdown=(self.rm._state.peak_equity - self.equity) / max(self.rm._state.peak_equity, 1),
                )
                session.add(snap)
                await session.commit()
        except Exception as exc:
            logger.warning(f"Could not save portfolio snapshot: {exc}")
