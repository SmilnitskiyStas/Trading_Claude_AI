from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.lgbm_model import CLASS_NAMES, LGBMTradingModel, MODELS_DIR
from src.models.trainer import WalkForwardTrainer
from src.utils.logger import logger

MIN_CONFIDENCE = 0.55   # ignore signals below this threshold


class SignalGenerator:
    """
    Generates trading signals from a trained LGBMTradingModel.

    Signal dict format:
    {
        "symbol":        "BTC/USDT",
        "action":        "buy|sell|hold",
        "signal":        1 | 0 | -1,
        "confidence":    0.73,
        "probabilities": {"sell": 0.12, "hold": 0.15, "buy": 0.73},
        "timestamp":     "2024-01-15T14:00:00Z",
        "exchange":      "binance",
    }
    """

    def __init__(
        self,
        model: LGBMTradingModel,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self.model = model
        self.min_confidence = min_confidence
        self._trainer = WalkForwardTrainer()

    # ── Single signal ──────────────────────────────────────────────────────

    def generate(
        self,
        features: np.ndarray | pd.Series,
        symbol: str,
        exchange: str = "binance",
        timestamp: datetime | None = None,
    ) -> dict:
        """Generate signal for a single row of features."""
        if isinstance(features, pd.Series):
            X = features.values.reshape(1, -1).astype(np.float32)
        else:
            X = np.array(features).reshape(1, -1).astype(np.float32)

        signals, confidences = self.model.predict(X)
        probas = self.model.predict_proba(X)[0]

        signal = int(signals[0])
        conf   = float(confidences[0])

        # Downgrade to Hold if below confidence threshold
        if conf < self.min_confidence:
            signal = 0

        action = {1: "buy", 0: "hold", -1: "sell"}[signal]
        ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "symbol":        symbol,
            "action":        action,
            "signal":        signal,
            "confidence":    round(conf, 4),
            "probabilities": {
                "sell": round(float(probas[0]), 4),
                "hold": round(float(probas[1]), 4),
                "buy":  round(float(probas[2]), 4),
            },
            "timestamp": ts,
            "exchange":  exchange,
        }

    # ── Batch signals ──────────────────────────────────────────────────────

    def generate_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str = "binance",
    ) -> pd.DataFrame:
        """
        Generate signals for a full DataFrame of features.
        Returns df with added columns: signal, confidence, action.
        """
        X, _, _ = self._trainer.prepare_features(df)
        signals, confidences = self.model.predict(X)

        # Downgrade low-confidence signals to Hold
        signals[confidences < self.min_confidence] = 0

        result = df.copy()
        result["signal"]     = signals
        result["confidence"] = confidences
        result["action"]     = pd.Series(signals, index=df.index).map(
            {1: "buy", 0: "hold", -1: "sell"}
        )
        return result

    # ── Latest signal ──────────────────────────────────────────────────────

    async def latest_signal(
        self,
        symbol: str,
        timeframe: str = "1h",
        exchange: str = "binance",
        lookback: int = 300,
    ) -> dict:
        """
        Load the most recent OHLCV candles, compute features,
        and return the current trading signal.
        """
        from src.data_pipeline.aggregator import DataAggregator
        from src.data_pipeline.processor import FeatureProcessor

        aggregator = DataAggregator()
        processor  = FeatureProcessor()

        df = await aggregator.load_ohlcv(exchange, symbol, timeframe, limit=lookback)
        if df.empty:
            return {"symbol": symbol, "action": "hold", "signal": 0, "confidence": 0.0,
                    "error": "no data"}

        processed = processor.process(df)
        processed.dropna(inplace=True)

        if processed.empty:
            return {"symbol": symbol, "action": "hold", "signal": 0, "confidence": 0.0,
                    "error": "insufficient rows after processing"}

        last_row = processed.iloc[[-1]]
        X, _, _ = self._trainer.prepare_features(last_row)

        return self.generate(X[0], symbol=symbol, exchange=exchange,
                             timestamp=processed.index[-1].to_pydatetime())

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        model_path: str | Path | None = None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> "SignalGenerator":
        path = Path(model_path) if model_path else MODELS_DIR / "lgbm_final.pkl"
        model = LGBMTradingModel.load(path)
        return cls(model, min_confidence=min_confidence)
