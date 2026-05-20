from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.paper_trader import PaperTrader

# Allowed chat — only the owner can control the bot
_ALLOWED_CHAT_ID = int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else 0


def _authorized(update: Update) -> bool:
    return update.effective_chat.id == _ALLOWED_CHAT_ID


def _esc(text: str) -> str:
    """Escape special chars for MarkdownV2."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


class TelegramBot:
    """
    Bot commands:
      /start    — greeting
      /status   — equity, drawdown, open positions count
      /positions — list of open positions with entry price and PnL
      /trades   — last 10 closed trades
      /metrics  — full performance metrics
      /stop     — graceful shutdown signal
    """

    def __init__(self, trader: "PaperTrader | None" = None) -> None:
        if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_token":
            raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env")

        self._trader = trader
        self._stop_event: asyncio.Event = asyncio.Event()
        self._app: Application = (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .build()
        )
        self._register_handlers()

    # ── Handler registration ───────────────────────────────────────────────

    def _register_handlers(self) -> None:
        a = self._app
        a.add_handler(CommandHandler("start",     self._cmd_start))
        a.add_handler(CommandHandler("status",    self._cmd_status))
        a.add_handler(CommandHandler("positions", self._cmd_positions))
        a.add_handler(CommandHandler("trades",    self._cmd_trades))
        a.add_handler(CommandHandler("metrics",   self._cmd_metrics))
        a.add_handler(CommandHandler("stop",      self._cmd_stop))

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize and start polling in background."""
        await self._app.initialize()
        await self._app.bot.set_my_commands([
            BotCommand("start",     "Greeting"),
            BotCommand("status",    "Equity & risk status"),
            BotCommand("positions", "Open positions"),
            BotCommand("trades",    "Last 10 closed trades"),
            BotCommand("metrics",   "Full performance metrics"),
            BotCommand("stop",      "Stop the trading system"),
        ])
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started — polling")
        await self.send_message("*Trading bot started* ✅\nPaper trading mode active\\.")

    async def stop(self) -> None:
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram bot stopped")

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # ── Push notifications ─────────────────────────────────────────────────

    async def send_message(self, text: str) -> None:
        try:
            await self._app.bot.send_message(
                chat_id=_ALLOWED_CHAT_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception as exc:
            logger.warning(f"Telegram send_message failed: {exc}")

    async def notify_trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size_usd: float,
        sl: float,
        tp: float,
    ) -> None:
        emoji = "🟢" if direction == "LONG" else "🔴"
        text = (
            f"{emoji} *OPEN {_esc(direction)}* `{_esc(symbol)}`\n"
            f"Entry: `{entry_price:.4f}`\n"
            f"Size: `{size_usd:.2f} USDT`\n"
            f"SL: `{sl:.4f}` \\| TP: `{tp:.4f}`"
        )
        await self.send_message(text)

    async def notify_trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl_usd: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        emoji = "✅" if pnl_usd >= 0 else "❌"
        sign = "+" if pnl_usd >= 0 else ""
        text = (
            f"{emoji} *CLOSE* `{_esc(symbol)}`\n"
            f"PnL: `{sign}{pnl_usd:.2f} USDT` \\(`{sign}{pnl_pct:.2%}`\\)\n"
            f"Reason: `{_esc(reason)}`"
        )
        await self.send_message(text)

    async def notify_risk_halt(self, reason: str) -> None:
        await self.send_message(
            f"⛔ *RISK HALT*\n{_esc(reason)}\nAll new positions blocked\\."
        )

    # ── Commands ───────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(update):
            return
        await update.message.reply_text(
            "👋 *Trading Bot Online*\n\n"
            "Commands:\n"
            "/status — risk & equity\n"
            "/positions — open positions\n"
            "/trades — last closed trades\n"
            "/metrics — performance metrics\n"
            "/stop — stop system",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(update):
            return
        if self._trader is None:
            await update.message.reply_text("No trader attached\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        rm = self._trader.rm
        s = rm.summary()
        equity = s["equity"]
        peak = s["peak_equity"]
        dd = s["drawdown"]
        halted = s["halted"]
        daily_pnl = s["daily_pnl"]
        n_pos = s["open_positions"]
        exposure = s["total_exposure"]

        status_icon = "⛔" if halted else "✅"
        dd_icon = "🔴" if dd > 0.05 else ("🟡" if dd > 0.02 else "🟢")
        sign = "+" if daily_pnl >= 0 else ""

        text = (
            f"{status_icon} *System Status*\n"
            f"Equity: `{equity:,.2f} USDT`\n"
            f"Peak:   `{peak:,.2f} USDT`\n"
            f"{dd_icon} Drawdown: `{dd:.2%}`\n"
            f"Daily PnL: `{sign}{daily_pnl:.2f} USDT`\n"
            f"Open positions: `{n_pos}`\n"
            f"Total exposure: `{exposure:.2f} USDT`\n"
            f"Halted: `{halted}`"
        )
        await update.message.reply_text(_clean_md(text), parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(update):
            return
        if self._trader is None:
            await update.message.reply_text("No trader attached\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        positions = self._trader._positions
        if not positions:
            await update.message.reply_text("No open positions\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        lines = ["*Open Positions*\n"]
        for sym, pos in positions.items():
            direction = "LONG" if pos.direction > 0 else "SHORT"
            held_h = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
            lines.append(
                f"`{_esc(sym)}` {direction}\n"
                f"  Entry: `{pos.entry_price:.4f}`\n"
                f"  Size: `{pos.notional_usd:.2f} USDT`\n"
                f"  SL: `{pos.stop_loss:.4f}` TP: `{pos.take_profit:.4f}`\n"
                f"  Held: `{held_h:.1f}h`"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(update):
            return
        if self._trader is None:
            await update.message.reply_text("No trader attached\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        trades = self._trader.closed_trades[-10:]
        if not trades:
            await update.message.reply_text("No closed trades yet\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        lines = [f"*Last {len(trades)} Closed Trades*\n"]
        for t in reversed(trades):
            emoji = "✅" if t.pnl_usd >= 0 else "❌"
            sign = "+" if t.pnl_usd >= 0 else ""
            lines.append(
                f"{emoji} `{_esc(t.symbol)}` "
                f"`{sign}{t.pnl_usd:.2f}` \\(`{sign}{t.pnl_pct:.2%}`\\) "
                f"_{_esc(t.exit_reason)}_"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(update):
            return
        if self._trader is None:
            await update.message.reply_text("No trader attached\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        from src.monitoring.metrics import calculate_metrics
        trade_returns = [t.pnl_pct for t in self._trader.closed_trades]
        holding_hours = [t.holding_hours for t in self._trader.closed_trades]

        if not trade_returns:
            await update.message.reply_text("Not enough trades yet\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        m = calculate_metrics(
            self._trader.equity_curve,
            trade_returns,
            holding_hours,
            periods_per_year=365,
        )
        sign = "+" if m.total_return >= 0 else ""
        text = (
            f"*Performance Metrics*\n"
            f"Return: `{sign}{m.total_return:.2%}`\n"
            f"Sharpe: `{m.sharpe_ratio:.3f}`\n"
            f"Sortino: `{m.sortino_ratio:.3f}`\n"
            f"Max DD: `{m.max_drawdown:.2%}`\n"
            f"Win rate: `{m.win_rate:.2%}` \\({m.winning_trades}/{m.total_trades}\\)\n"
            f"Profit factor: `{m.profit_factor:.3f}`\n"
            f"Avg win/loss: `{m.avg_win_pct:+.2%}` / `{m.avg_loss_pct:+.2%}`"
        )
        await update.message.reply_text(_clean_md(text), parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(update):
            return
        logger.warning("Stop command received via Telegram")
        self._stop_event.set()
        await update.message.reply_text(
            "⛔ *Stop signal sent*\\. System will shut down after current operations complete\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


def _clean_md(text: str) -> str:
    """Escape standalone special chars that aren't part of intentional formatting."""
    # Only escape + and = outside of backtick spans (simple pass for non-nested cases)
    result = []
    inside_code = False
    for ch in text:
        if ch == "`":
            inside_code = not inside_code
        if not inside_code and ch in ("+", "=", "|", "-") and ch not in result[-2:]:
            result.append(f"\\{ch}")
        else:
            result.append(ch)
    return "".join(result)
