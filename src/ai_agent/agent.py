from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import anthropic

from src.ai_agent.news_analyzer import NewsAnalyzer
from src.ai_agent.news_fetcher import NewsFetcher
from src.utils.cache import cache
from src.utils.config import (
    AGENT_MODEL, ANTHROPIC_API_KEY, SYMBOLS,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.paper_trader import PaperTrader

_CACHE_KEY = "agent:analysis"
_CACHE_TTL = 3600   # 1 hour — same as AGENT_CALL_INTERVAL_SECONDS


_SYSTEM_PROMPT = """\
You are an expert crypto trading analyst.
You receive:
1. Current portfolio status (equity, drawdown, open positions)
2. Recent ML model signals per symbol (buy/hold/sell + confidence)
3. News sentiment scores per symbol (range -1 to +1)

Your task: provide a concise trading decision per symbol.

Rules:
- Output ONLY valid JSON (no markdown, no explanation outside the JSON)
- Format: { "decisions": [ { "symbol": "BTC/USDT", "action": "buy"|"hold"|"sell", "confidence": 0.0-1.0, "reasoning": "one sentence" }, ... ] }
- Be conservative — when in doubt, output "hold"
- Never suggest sizing or leverage, only direction
- If portfolio drawdown > 10%, suggest "hold" for all symbols\
"""


class TradingAgent:
    """
    Calls Claude API once per hour to get high-level trading direction
    per symbol, incorporating ML signals + news sentiment.
    Results are cached in Redis for 1 hour.
    Can be enabled/disabled at runtime via the dashboard toggle.
    """

    def __init__(self, trader: "PaperTrader | None" = None) -> None:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY in ("your_key", ""):
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        self._client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self._fetcher = NewsFetcher(lookback_hours=4)
        self._analyzer = NewsAnalyzer()
        self._trader = trader
        self.enabled: bool = True   # can be toggled via dashboard

    # ── Main entry point ───────────────────────────────────────────────────

    async def analyze(
        self,
        ml_signals: dict[str, dict],
        symbols: list[str] | None = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Returns cached agent analysis if fresh, otherwise calls Claude.

        ml_signals: { "BTC/USDT": {"action": "buy", "confidence": 0.72, ...} }
        Returns:    { "BTC/USDT": {"action": "buy", "confidence": 0.85, "reasoning": "..."} }
        """
        active_symbols = symbols or SYMBOLS

        # Try cache first
        if not force_refresh and cache.available:
            cached = await cache.get(_CACHE_KEY)
            if cached:
                logger.debug("Agent: returning cached analysis")
                return json.loads(cached)

        # Fetch fresh news + sentiment
        news_items = await self._fetcher.fetch_all(active_symbols)
        sentiment_results = self._analyzer.analyze(news_items)
        sentiment_by_symbol = self._analyzer.aggregate_by_symbol(sentiment_results)

        # Build prompt payload
        portfolio = self._portfolio_summary()
        user_content = self._build_user_message(
            active_symbols, ml_signals, sentiment_by_symbol, portfolio, news_items[:10]
        )

        logger.info(f"Agent: calling Claude {AGENT_MODEL} ...")
        response_text = await self._call_claude(user_content)
        decisions = self._parse_response(response_text, active_symbols)

        # Cache for 1 hour
        if cache.available:
            await cache.set(_CACHE_KEY, json.dumps(decisions), ttl=_CACHE_TTL)

        # Persist to DB
        await self._save_predictions(decisions, ml_signals)

        logger.info(f"Agent decisions: {decisions}")
        return decisions

    # ── Claude call ────────────────────────────────────────────────────────

    async def _call_claude(self, user_content: str) -> str:
        try:
            message = await self._client.messages.create(
                model=AGENT_MODEL,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            return message.content[0].text
        except Exception as exc:
            logger.error(f"Claude API error: {exc}")
            return "{}"

    # ── Helpers ────────────────────────────────────────────────────────────

    def _portfolio_summary(self) -> dict:
        if self._trader is None:
            return {"equity": 10000, "drawdown": 0.0, "open_positions": 0}
        s = self._trader.rm.summary()
        return {
            "equity":         s["equity"],
            "drawdown_pct":   round(s["drawdown"] * 100, 2),
            "open_positions": s["open_positions"],
            "total_exposure": s["total_exposure"],
            "halted":         s["halted"],
        }

    def _build_user_message(
        self,
        symbols: list[str],
        ml_signals: dict[str, dict],
        sentiment: dict[str, float],
        portfolio: dict,
        news_items: list,
    ) -> str:
        lines = [
            "=== PORTFOLIO STATUS ===",
            json.dumps(portfolio, indent=2),
            "",
            "=== ML SIGNALS ===",
        ]
        for sym in symbols:
            sig = ml_signals.get(sym, {})
            sent = sentiment.get(sym, 0.0)
            lines.append(
                f"{sym}: action={sig.get('action','hold')} "
                f"confidence={sig.get('confidence', 0):.2f} "
                f"news_sentiment={sent:+.2f}"
            )

        lines += ["", "=== RECENT HEADLINES ==="]
        for item in news_items[:8]:
            lines.append(f"- [{item.source}] {item.title}")

        lines += [
            "",
            f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "Provide your trading decisions JSON now.",
        ]
        return "\n".join(lines)

    def _parse_response(self, text: str, symbols: list[str]) -> dict:
        """Parse Claude JSON response → { symbol: {action, confidence, reasoning} }."""
        default = {sym: {"action": "hold", "confidence": 0.5, "reasoning": "default"} for sym in symbols}
        if not text or text == "{}":
            return default

        # Extract JSON block (Claude sometimes wraps in ```json)
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return default

        try:
            data = json.loads(match.group())
            decisions = {}
            for item in data.get("decisions", []):
                sym = item.get("symbol", "")
                if sym in symbols:
                    decisions[sym] = {
                        "action":     item.get("action", "hold"),
                        "confidence": float(item.get("confidence", 0.5)),
                        "reasoning":  item.get("reasoning", ""),
                    }
            # Fill missing symbols with hold
            for sym in symbols:
                if sym not in decisions:
                    decisions[sym] = {"action": "hold", "confidence": 0.5, "reasoning": "no signal"}
            return decisions
        except Exception as exc:
            logger.warning(f"Agent response parse error: {exc}\nRaw: {text[:200]}")
            return default

    async def _save_predictions(self, decisions: dict, ml_signals: dict) -> None:
        from src.utils.database import get_session, MLPrediction
        signal_map = {"buy": 1, "hold": 0, "sell": -1}
        try:
            async with get_session() as session:
                ts = datetime.now(timezone.utc)
                for sym, dec in decisions.items():
                    ml_sig = ml_signals.get(sym, {})
                    pred = MLPrediction(
                        timestamp=ts,
                        symbol=sym,
                        exchange="binance",
                        signal=signal_map.get(dec["action"], 0),
                        confidence=dec["confidence"],
                        features_json=json.dumps({
                            "ml_action":      ml_sig.get("action"),
                            "ml_confidence":  ml_sig.get("confidence"),
                            "agent_action":   dec["action"],
                            "agent_reasoning": dec["reasoning"],
                        }),
                    )
                    session.add(pred)
                await session.commit()
        except Exception as exc:
            logger.warning(f"Could not save agent predictions: {exc}")
