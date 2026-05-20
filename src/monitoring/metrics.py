from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return: float       # fractional, e.g. 0.25 = +25%
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float       # positive fraction, e.g. 0.12 = 12%
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    avg_trade_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_holding_hours: float
    trading_days: float

    def __str__(self) -> str:
        lines = [
            "=== Performance Metrics ===",
            f"Total return       : {self.total_return:+.2%}",
            f"Annualized return  : {self.annualized_return:+.2%}",
            f"Sharpe ratio       : {self.sharpe_ratio:.3f}",
            f"Sortino ratio      : {self.sortino_ratio:.3f}",
            f"Calmar ratio       : {self.calmar_ratio:.3f}",
            f"Max drawdown       : {self.max_drawdown:.2%}",
            f"Win rate           : {self.win_rate:.2%}  ({self.winning_trades}/{self.total_trades})",
            f"Profit factor      : {self.profit_factor:.3f}",
            f"Avg win / loss     : {self.avg_win_pct:+.2%} / {self.avg_loss_pct:+.2%}",
            f"Best / worst trade : {self.best_trade_pct:+.2%} / {self.worst_trade_pct:+.2%}",
            f"Avg holding time   : {self.avg_holding_hours:.1f}h",
            f"Trading days       : {self.trading_days:.0f}",
        ]
        return "\n".join(lines)


def calculate_metrics(
    equity_curve: Sequence[float],
    trade_returns: Sequence[float],
    holding_hours: Sequence[float] | None = None,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 8760,  # hourly bars
) -> PerformanceMetrics:
    """
    equity_curve   : list of portfolio values (first = initial capital)
    trade_returns  : list of fractional PnL per closed trade, e.g. 0.025 = +2.5%
    holding_hours  : list of hours each trade was held (optional)
    """
    eq = np.array(equity_curve, dtype=float)
    tr = np.array(trade_returns, dtype=float) if len(trade_returns) else np.array([0.0])

    # ── Returns & cumulative performance ──────────────────────────────────
    if len(eq) < 2:
        total_ret = 0.0
        ann_ret = 0.0
    else:
        total_ret = (eq[-1] - eq[0]) / eq[0]
        n_periods = len(eq) - 1
        years = n_periods / periods_per_year
        ann_ret = (1 + total_ret) ** (1 / max(years, 1e-6)) - 1

    # ── Sharpe ratio (hourly equity curve) ────────────────────────────────
    if len(eq) > 1:
        pct_rets = np.diff(eq) / eq[:-1]
        rf_per_period = risk_free_rate / periods_per_year
        excess = pct_rets - rf_per_period
        sharpe = (np.mean(excess) / (np.std(excess) + 1e-10)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # ── Sortino ratio (downside deviation only) ────────────────────────────
    if len(eq) > 1:
        downside = pct_rets[pct_rets < 0]
        down_std = np.std(downside) if len(downside) > 0 else 1e-10
        sortino = (np.mean(pct_rets) / (down_std + 1e-10)) * np.sqrt(periods_per_year)
    else:
        sortino = 0.0

    # ── Maximum drawdown ──────────────────────────────────────────────────
    max_dd = _max_drawdown(eq)

    # ── Calmar ratio ──────────────────────────────────────────────────────
    calmar = ann_ret / max_dd if max_dd > 1e-6 else 0.0

    # ── Trade statistics ──────────────────────────────────────────────────
    total = len(tr)
    wins = tr[tr > 0]
    losses = tr[tr < 0]

    win_rate = len(wins) / total if total > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    avg_trade = float(np.mean(tr))
    best = float(np.max(tr)) if total > 0 else 0.0
    worst = float(np.min(tr)) if total > 0 else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    avg_hold = float(np.mean(holding_hours)) if holding_hours and len(holding_hours) > 0 else 0.0

    trading_days = (len(eq) - 1) if len(eq) > 1 else 0.0

    return PerformanceMetrics(
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 4),
        calmar_ratio=round(calmar, 4),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        total_trades=total,
        winning_trades=len(wins),
        losing_trades=len(losses),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        avg_trade_pct=round(avg_trade, 4),
        best_trade_pct=round(best, 4),
        worst_trade_pct=round(worst, 4),
        avg_holding_hours=round(avg_hold, 1),
        trading_days=round(trading_days, 1),
    )


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-10)
    return float(np.max(dd))
