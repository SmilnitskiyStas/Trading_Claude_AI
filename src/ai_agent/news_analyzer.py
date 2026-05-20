from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from src.utils.logger import logger

# ProsusAI/finbert is the standard financial sentiment model (3 classes: positive/negative/neutral)
_MODEL_NAME = "ProsusAI/finbert"
_LABELS = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Score threshold — ignore very weak signals
_MIN_CONFIDENCE = 0.60


@dataclass
class SentimentResult:
    title: str
    label: str           # "positive" | "negative" | "neutral"
    score: float         # raw probability of winning label
    sentiment: float     # mapped to [-1, +1]
    symbols: list[str]


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load FinBERT once and cache globally (heavy model)."""
    try:
        from transformers import pipeline
        logger.info(f"Loading FinBERT model: {_MODEL_NAME} ...")
        pipe = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            truncation=True,
            max_length=512,
            device=-1,   # CPU only
        )
        logger.info("FinBERT loaded OK")
        return pipe
    except Exception as exc:
        logger.error(f"Failed to load FinBERT: {exc}")
        return None


class NewsAnalyzer:
    """
    Runs FinBERT sentiment analysis on news headlines.
    Falls back to keyword-based scoring if model unavailable.
    """

    def __init__(self, min_confidence: float = _MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        self._pipe = None   # lazy-loaded on first call

    def _get_pipe(self):
        if self._pipe is None:
            self._pipe = _load_pipeline()
        return self._pipe

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(self, items: list) -> list[SentimentResult]:
        """
        Analyze a list of NewsItem objects.
        Returns SentimentResult for each with a score.
        """
        if not items:
            return []

        pipe = self._get_pipe()
        results: list[SentimentResult] = []

        if pipe is not None:
            texts = [f"{item.title}. {item.raw_text[:200]}" for item in items]
            try:
                outputs = pipe(texts, batch_size=8)
                for item, out in zip(items, outputs):
                    label = out["label"].lower()
                    score = float(out["score"])
                    sentiment = _LABELS.get(label, 0.0) if score >= self.min_confidence else 0.0
                    results.append(SentimentResult(
                        title=item.title,
                        label=label,
                        score=round(score, 4),
                        sentiment=round(sentiment, 4),
                        symbols=item.symbols,
                    ))
            except Exception as exc:
                logger.warning(f"FinBERT inference error: {exc} — falling back to keywords")
                results = [self._keyword_score(item) for item in items]
        else:
            results = [self._keyword_score(item) for item in items]

        pos = sum(1 for r in results if r.sentiment > 0)
        neg = sum(1 for r in results if r.sentiment < 0)
        logger.info(f"Sentiment: {len(results)} items | +{pos} / -{neg}")
        return results

    def aggregate_by_symbol(
        self, results: list[SentimentResult]
    ) -> dict[str, float]:
        """
        Returns { symbol: avg_sentiment } for each symbol found in results.
        Score is in [-1, +1]. Only includes symbols with >= 1 tagged result.
        """
        buckets: dict[str, list[float]] = {}
        for r in results:
            for sym in r.symbols:
                buckets.setdefault(sym, []).append(r.sentiment)

        return {
            sym: round(sum(scores) / len(scores), 4)
            for sym, scores in buckets.items()
            if scores
        }

    # ── Keyword fallback ───────────────────────────────────────────────────

    @staticmethod
    def _keyword_score(item) -> SentimentResult:
        text = (item.title + " " + item.raw_text).lower()
        pos_words = [
            "surge", "rally", "bullish", "breakout", "record", "high",
            "gain", "rise", "boom", "adoption", "launch", "partnership",
            "upgrade", "positive", "growth", "buy",
        ]
        neg_words = [
            "crash", "drop", "bearish", "hack", "ban", "fraud", "scam",
            "loss", "sell", "collapse", "concern", "risk", "lawsuit",
            "regulation", "investigation", "fear", "dump",
        ]
        pos = sum(1 for w in pos_words if w in text)
        neg = sum(1 for w in neg_words if w in text)

        if pos > neg:
            label, score, sentiment = "positive", 0.65, 1.0
        elif neg > pos:
            label, score, sentiment = "negative", 0.65, -1.0
        else:
            label, score, sentiment = "neutral", 0.50, 0.0

        return SentimentResult(
            title=item.title,
            label=label,
            score=score,
            sentiment=sentiment,
            symbols=item.symbols,
        )
