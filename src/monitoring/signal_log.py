from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SignalEntry:
    ts: datetime
    symbol: str
    ml_action: str       # buy | hold | sell
    ml_confidence: float
    p_buy: float         # raw P(buy) from model
    p_hold: float        # raw P(hold) from model
    p_sell: float        # raw P(sell) from model
    agent_action: str    # buy | hold | sell | disabled
    final_action: str    # buy | hold | sell
    blocked_reason: str  # "" | position_exists | risk_halted | agent_disagrees


class SignalLog:
    """In-memory ring buffer of signal decisions for dashboard display."""

    def __init__(self, maxlen: int = 500) -> None:
        self._entries: deque[SignalEntry] = deque(maxlen=maxlen)

    def append(self, entry: SignalEntry) -> None:
        self._entries.append(entry)

    def recent(self, n: int = 100) -> list[dict]:
        return [
            {
                "ts":             e.ts.isoformat(),
                "symbol":         e.symbol,
                "ml_action":      e.ml_action,
                "ml_confidence":  round(e.ml_confidence * 100, 1),
                "p_buy":          round(e.p_buy * 100, 1),
                "p_hold":         round(e.p_hold * 100, 1),
                "p_sell":         round(e.p_sell * 100, 1),
                "agent_action":   e.agent_action,
                "final_action":   e.final_action,
                "blocked_reason": e.blocked_reason,
            }
            for e in list(self._entries)[-n:]
        ]
