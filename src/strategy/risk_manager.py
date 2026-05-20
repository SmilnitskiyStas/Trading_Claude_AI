from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.utils.config import (
    INITIAL_BALANCE, MAX_POSITION_SIZE, STOP_LOSS, TAKE_PROFIT,
    MAX_DAILY_LOSS,
)
from src.utils.logger import logger

MAX_DRAWDOWN = 0.15          # halt trading if equity drops >15% from peak
MIN_KELLY_FRACTION = 0.01    # floor Kelly output at 1%
MAX_KELLY_FRACTION = 0.25    # cap Kelly output (never bet more than 25%)
TRADING_FEE = 0.001          # 0.1% per side (taker fee estimate)


@dataclass
class PositionSizeResult:
    symbol: str
    direction: int           # 1=long, -1=short
    position_size_usd: float
    position_size_pct: float # fraction of equity
    stop_loss_price: float
    take_profit_price: float
    max_loss_usd: float
    kelly_fraction: float
    reason: str = ""


@dataclass
class RiskState:
    equity: float = INITIAL_BALANCE
    peak_equity: float = INITIAL_BALANCE
    daily_start_equity: float = INITIAL_BALANCE
    daily_pnl: float = 0.0
    current_date: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    open_positions: dict[str, float] = field(default_factory=dict)  # symbol → notional value
    halted: bool = False


class RiskManager:
    """
    Controls position sizing and risk limits.

    Rules:
    - Position size = Kelly Criterion capped at MAX_POSITION_SIZE
    - Stop-loss = STOP_LOSS % from entry
    - Take-profit = TAKE_PROFIT % from entry
    - Daily loss limit = MAX_DAILY_LOSS % of start-of-day equity
    - Max drawdown = MAX_DRAWDOWN % from peak equity → halts trading
    - One position per symbol at a time
    """

    def __init__(
        self,
        initial_balance: float = INITIAL_BALANCE,
        max_position_size: float = MAX_POSITION_SIZE,
        stop_loss: float = STOP_LOSS,
        take_profit: float = TAKE_PROFIT,
        max_daily_loss: float = MAX_DAILY_LOSS,
        max_drawdown: float = MAX_DRAWDOWN,
        fee: float = TRADING_FEE,
    ) -> None:
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.fee = fee
        self._state = RiskState(
            equity=initial_balance,
            peak_equity=initial_balance,
            daily_start_equity=initial_balance,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._state.equity

    @property
    def is_halted(self) -> bool:
        return self._state.halted

    def update_equity(self, new_equity: float, ts: datetime | None = None) -> None:
        """Call after each closed trade or mark-to-market update."""
        now = (ts or datetime.now(timezone.utc)).date()
        state = self._state

        # Reset daily PnL on new day
        if now != state.current_date:
            state.current_date = now
            state.daily_start_equity = state.equity
            state.daily_pnl = 0.0

        pnl = new_equity - state.equity
        state.equity = new_equity
        state.daily_pnl += pnl

        if new_equity > state.peak_equity:
            state.peak_equity = new_equity

        # Check halt conditions
        drawdown = (state.peak_equity - new_equity) / state.peak_equity
        daily_loss_pct = (state.daily_start_equity - new_equity) / state.daily_start_equity

        if drawdown >= self.max_drawdown:
            if not state.halted:
                logger.warning(
                    f"RISK HALT: drawdown {drawdown:.2%} >= {self.max_drawdown:.2%}. "
                    f"Equity {new_equity:.2f} / peak {state.peak_equity:.2f}"
                )
            state.halted = True
        elif daily_loss_pct >= self.max_daily_loss:
            if not state.halted:
                logger.warning(
                    f"RISK HALT: daily loss {daily_loss_pct:.2%} >= {self.max_daily_loss:.2%}"
                )
            state.halted = True
        else:
            state.halted = False

    def can_open(self, symbol: str) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if self._state.halted:
            return False, "trading halted (drawdown or daily loss limit)"
        if symbol in self._state.open_positions:
            return False, f"already have position in {symbol}"
        total_exposure = sum(self._state.open_positions.values())
        if total_exposure >= self._state.equity * 0.80:
            return False, "total exposure exceeds 80% of equity"
        return True, ""

    def size_position(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> PositionSizeResult | None:
        """
        Calculate position size using Kelly Criterion.
        Returns None if trade is not allowed.
        """
        allowed, reason = self.can_open(symbol)
        if not allowed:
            logger.debug(f"Position rejected for {symbol}: {reason}")
            return None

        kelly = self._kelly(win_rate, avg_win, avg_loss)
        size_pct = min(kelly, self.max_position_size)
        size_usd = self._state.equity * size_pct

        sl_price, tp_price = self._sl_tp(entry_price, direction)
        max_loss_usd = size_usd * self.stop_loss + size_usd * self.fee * 2

        logger.debug(
            f"{symbol} {'+' if direction > 0 else '-'} | "
            f"kelly={kelly:.3f} capped={size_pct:.3f} | "
            f"size={size_usd:.2f} USD | sl={sl_price:.4f} tp={tp_price:.4f}"
        )

        return PositionSizeResult(
            symbol=symbol,
            direction=direction,
            position_size_usd=round(size_usd, 2),
            position_size_pct=round(size_pct, 4),
            stop_loss_price=round(sl_price, 8),
            take_profit_price=round(tp_price, 8),
            max_loss_usd=round(max_loss_usd, 2),
            kelly_fraction=round(kelly, 4),
        )

    def register_open(self, symbol: str, notional_usd: float) -> None:
        self._state.open_positions[symbol] = notional_usd

    def register_close(self, symbol: str) -> None:
        self._state.open_positions.pop(symbol, None)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Full Kelly: f = (p * b - q) / b
        where b = avg_win / avg_loss, p = win_rate, q = 1 - p.
        Returns half-Kelly for conservatism, clamped to [MIN, MAX].
        """
        if avg_loss <= 0 or avg_win <= 0:
            return MIN_KELLY_FRACTION
        b = avg_win / avg_loss
        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p
        full_kelly = (p * b - q) / b
        half_kelly = full_kelly * 0.5
        return max(MIN_KELLY_FRACTION, min(MAX_KELLY_FRACTION, half_kelly))

    def _sl_tp(self, entry: float, direction: int) -> tuple[float, float]:
        if direction > 0:  # long
            sl = entry * (1 - self.stop_loss)
            tp = entry * (1 + self.take_profit)
        else:              # short
            sl = entry * (1 + self.stop_loss)
            tp = entry * (1 - self.take_profit)
        return sl, tp

    # ── Reporting ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        s = self._state
        drawdown = (s.peak_equity - s.equity) / s.peak_equity if s.peak_equity > 0 else 0.0
        return {
            "equity":           round(s.equity, 2),
            "peak_equity":      round(s.peak_equity, 2),
            "drawdown":         round(drawdown, 4),
            "daily_pnl":        round(s.daily_pnl, 2),
            "open_positions":   len(s.open_positions),
            "total_exposure":   round(sum(s.open_positions.values()), 2),
            "halted":           s.halted,
        }
