from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import feedparser

from src.utils.config import CRYPTOPANIC_API_KEY
from src.utils.logger import logger

# RSS feeds — no API key required
_RSS_FEEDS: dict[str, str] = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "decrypt":       "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
}

# CryptoPanic filter → only important/bullish/bearish
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# Map symbol to search keywords
_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "BTC/USDT":  ["bitcoin", "btc"],
    "ETH/USDT":  ["ethereum", "eth"],
    "BNB/USDT":  ["bnb", "binance coin"],
    "SOL/USDT":  ["solana", "sol"],
    "XRP/USDT":  ["xrp", "ripple"],
    "DOGE/USDT": ["dogecoin", "doge"],
    "ADA/USDT":  ["cardano", "ada"],
    "AVAX/USDT": ["avalanche", "avax"],
    "TRX/USDT":  ["tron", "trx"],
    "LINK/USDT": ["chainlink", "link"],
}


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published_at: datetime
    symbols: list[str]     # which symbols this news relates to
    raw_text: str = ""


class NewsFetcher:
    """
    Fetches crypto news from CryptoPanic API and RSS feeds.
    Returns a list of NewsItem objects for the last N hours.
    """

    def __init__(self, lookback_hours: int = 4) -> None:
        self.lookback_hours = lookback_hours

    async def fetch_all(self, symbols: list[str] | None = None) -> list[NewsItem]:
        """Fetch news from all sources concurrently."""
        tasks = [self._fetch_rss()]
        if CRYPTOPANIC_API_KEY and CRYPTOPANIC_API_KEY not in ("your_key", ""):
            tasks.append(self._fetch_cryptopanic())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"News fetch error: {r}")
            else:
                items.extend(r)

        # Tag each item with relevant symbols
        target = symbols or list(_SYMBOL_KEYWORDS.keys())
        for item in items:
            item.symbols = self._tag_symbols(item.title + " " + item.raw_text, target)

        # De-duplicate by URL and sort by recency
        seen: set[str] = set()
        unique: list[NewsItem] = []
        for item in sorted(items, key=lambda x: x.published_at, reverse=True):
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)

        logger.info(f"Fetched {len(unique)} news items ({len(tasks)} sources)")
        return unique

    # ── CryptoPanic ────────────────────────────────────────────────────────

    async def _fetch_cryptopanic(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        params = {
            "auth_token": CRYPTOPANIC_API_KEY,
            "public":     "true",
            "kind":       "news",
            "filter":     "important",
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(_CRYPTOPANIC_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"CryptoPanic HTTP {resp.status}")
                        return items
                    data = await resp.json()

            cutoff = self._cutoff()
            for post in data.get("results", []):
                dt = self._parse_dt(post.get("published_at", ""))
                if dt and dt < cutoff:
                    continue
                items.append(NewsItem(
                    title=post.get("title", ""),
                    source="cryptopanic",
                    url=post.get("url", ""),
                    published_at=dt or datetime.now(timezone.utc),
                    symbols=[],
                ))
        except Exception as exc:
            logger.warning(f"CryptoPanic fetch failed: {exc}")
        return items

    # ── RSS ────────────────────────────────────────────────────────────────

    async def _fetch_rss(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        cutoff = self._cutoff()

        async def _one(name: str, url: str) -> list[NewsItem]:
            result: list[NewsItem] = []
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=8),
                    headers={"User-Agent": "TradingBot/1.0"},
                ) as session:
                    async with session.get(url) as resp:
                        content = await resp.read()
                feed = feedparser.parse(content)
                for entry in feed.entries[:20]:
                    dt = self._parse_dt(
                        entry.get("published", "") or entry.get("updated", "")
                    )
                    if dt and dt < cutoff:
                        continue
                    summary = entry.get("summary", "") or ""
                    # Strip HTML tags (simple)
                    import re
                    text = re.sub(r"<[^>]+>", " ", summary)
                    result.append(NewsItem(
                        title=entry.get("title", ""),
                        source=name,
                        url=entry.get("link", ""),
                        published_at=dt or datetime.now(timezone.utc),
                        symbols=[],
                        raw_text=text[:500],
                    ))
            except Exception as exc:
                logger.debug(f"RSS {name} failed: {exc}")
            return result

        tasks = [_one(name, url) for name, url in _RSS_FEEDS.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if not isinstance(r, Exception):
                items.extend(r)
        return items

    # ── Helpers ────────────────────────────────────────────────────────────

    def _cutoff(self) -> datetime:
        from datetime import timedelta
        return datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)

    @staticmethod
    def _parse_dt(s: str) -> datetime | None:
        if not s:
            return None
        from email.utils import parsedate_to_datetime
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S%z",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        try:
            return parsedate_to_datetime(s)
        except Exception:
            return None

    @staticmethod
    def _tag_symbols(text: str, symbols: list[str]) -> list[str]:
        text_lower = text.lower()
        found = []
        for sym in symbols:
            for kw in _SYMBOL_KEYWORDS.get(sym, []):
                if kw in text_lower:
                    found.append(sym)
                    break
        return found
