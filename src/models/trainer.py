from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
    roc_auc_score,
)

from src.data_pipeline.aggregator import DataAggregator
from src.data_pipeline.processor import FeatureProcessor
from src.models.lgbm_model import (
    CLASS_TO_SIGNAL, SIGNAL_TO_CLASS, LGBMTradingModel, MODELS_DIR,
)
from src.utils.config import SYMBOLS, ACTIVE_EXCHANGES
from src.utils.logger import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Walk-forward parameters
TRAIN_MONTHS = 12
VAL_MONTHS   = 2
GAP_DAYS     = 7
STEP_MONTHS  = 2   # advance fold by 2 months → ~12 folds on 3 years

# Features to normalize (divide by close to make scale-invariant)
_PRICE_FEATURES = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "sma_20", "sma_50",
    "bb_upper", "bb_lower", "bb_middle",
    "keltner_upper", "keltner_lower",
    "vwap",
]
_MACD_FEATURES = ["macd", "macd_signal", "macd_hist"]

# Columns to drop before model input
_DROP_COLS = {
    "open", "high", "low", "close", "volume",
    "target", "close_vwap",
    *_PRICE_FEATURES,
    *_MACD_FEATURES,
    "obv",           # cumulative — not stationary
    "volume_sma_20", # absolute volume
}


@dataclass
class FoldResult:
    fold: int
    train_from: str
    train_to: str
    val_from: str
    val_to: str
    accuracy: float
    f1_macro: float
    precision_buy: float
    recall_buy: float
    roc_auc: float
    best_iteration: int
    n_train: int
    n_val: int


@dataclass
class TrainingResult:
    folds: list[FoldResult] = field(default_factory=list)
    final_model_path: str = ""

    @property
    def avg_accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def avg_f1(self) -> float:
        return float(np.mean([f.f1_macro for f in self.folds]))

    @property
    def avg_roc_auc(self) -> float:
        return float(np.mean([f.roc_auc for f in self.folds]))

    def summary(self) -> str:
        lines = [
            "=== Walk-Forward Validation Summary ===",
            f"Folds      : {len(self.folds)}",
            f"Accuracy   : {self.avg_accuracy:.4f}",
            f"F1 macro   : {self.avg_f1:.4f}",
            f"ROC-AUC    : {self.avg_roc_auc:.4f}",
        ]
        if self.final_model_path:
            lines.append(f"Model saved: {self.final_model_path}")
        return "\n".join(lines)


class WalkForwardTrainer:

    def __init__(
        self,
        train_months: int = TRAIN_MONTHS,
        val_months:   int = VAL_MONTHS,
        gap_days:     int = GAP_DAYS,
        step_months:  int = STEP_MONTHS,
    ) -> None:
        self.train_months = train_months
        self.val_months   = val_months
        self.gap_days     = gap_days
        self.step_months  = step_months
        self._processor   = FeatureProcessor()
        self._aggregator  = DataAggregator()

    # ── Data loading ───────────────────────────────────────────────────────

    async def load_all_symbols(
        self,
        symbols: list[str] = SYMBOLS,
        timeframe: str = "1h",
        exchange: str = "binance",
    ) -> pd.DataFrame:
        """
        Load, process and combine features for all symbols.
        Adds an integer `symbol_id` column for the model.
        """
        frames = []
        sym_map = {s: i for i, s in enumerate(symbols)}

        for sym in symbols:
            df = await self._aggregator.load_ohlcv(exchange, sym, timeframe)
            if df.empty:
                logger.warning(f"No data for {sym}/{timeframe} on {exchange}")
                continue

            processed = self._processor.process(df)
            processed = self._processor.add_target(processed)
            processed.dropna(inplace=True)

            if processed.empty:
                continue

            processed["symbol_id"] = sym_map[sym]
            processed["symbol"]    = sym
            frames.append(processed)
            logger.info(f"Loaded {sym}: {len(processed)} rows")

        if not frames:
            raise RuntimeError("No data loaded — run --mode download first")

        combined = pd.concat(frames).sort_index()
        logger.info(f"Combined dataset: {len(combined)} rows, {combined['symbol'].nunique()} symbols")
        return combined

    # ── Feature preparation ────────────────────────────────────────────────

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Normalize and select features.
        Returns X (np.ndarray), y (np.ndarray), feature_names (list).
        """
        out = df.copy()
        close = out["close"]

        # Normalize price-level features → % deviation from close
        for col in _PRICE_FEATURES:
            if col in out.columns:
                out[f"{col}_pct"] = (out[col] - close) / (close + 1e-10)

        # Normalize MACD by close
        for col in _MACD_FEATURES:
            if col in out.columns:
                out[f"{col}_pct"] = out[col] / (close + 1e-10)

        # OBV: use per-candle flow instead of cumulative
        if "obv" in out.columns:
            out["obv_flow"] = out["obv"].diff() / (out["volume"] + 1e-10)

        # Drop excluded cols and non-feature cols
        drop = [c for c in _DROP_COLS if c in out.columns] + ["symbol"]
        feature_cols = [c for c in out.columns if c not in drop and c != "target"]

        X = out[feature_cols].values.astype(np.float32)
        if "target" in out.columns:
            y = out["target"].values.astype(np.int8)
        else:
            y = np.zeros(len(out), dtype=np.int8)
        return X, y, feature_cols

    # ── Walk-forward fold generation ───────────────────────────────────────

    def generate_folds(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Yield (train_df, val_df) pairs.
        Window advances by `step_months` each fold.
        """
        idx = df.index
        start = idx.min().replace(tzinfo=timezone.utc) if idx.tzinfo is None else idx.min()

        while True:
            train_end = start + pd.DateOffset(months=self.train_months)
            gap_end   = train_end + timedelta(days=self.gap_days)
            val_end   = gap_end   + pd.DateOffset(months=self.val_months)

            if val_end > idx.max():
                break

            train_df = df[(df.index >= start) & (df.index < train_end)]
            val_df   = df[(df.index >= gap_end) & (df.index < val_end)]

            if len(train_df) > 500 and len(val_df) > 100:
                yield train_df, val_df

            start = start + pd.DateOffset(months=self.step_months)

    # ── Metrics ────────────────────────────────────────────────────────────

    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, probas: np.ndarray) -> dict:
        # Map signals back to 3-class for sklearn
        y_t = np.vectorize(SIGNAL_TO_CLASS.get)(y_true)
        y_p = np.vectorize(SIGNAL_TO_CLASS.get)(y_pred)

        try:
            roc = roc_auc_score(y_t, probas, multi_class="ovr", average="macro")
        except Exception:
            roc = 0.0

        # Buy class = class 2
        buy_mask = y_t == 2
        precision_buy = precision_score(y_t, y_p, labels=[2], average="micro", zero_division=0)
        recall_buy    = recall_score   (y_t, y_p, labels=[2], average="micro", zero_division=0)

        return {
            "accuracy":      accuracy_score(y_t, y_p),
            "f1_macro":      f1_score(y_t, y_p, average="macro", zero_division=0),
            "precision_buy": precision_buy,
            "recall_buy":    recall_buy,
            "roc_auc":       roc,
        }

    # ── Walk-forward run ───────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        model_params: dict | None = None,
    ) -> TrainingResult:
        """Run full walk-forward validation. Returns TrainingResult."""
        result = TrainingResult()
        folds = list(self.generate_folds(df))
        logger.info(f"Running walk-forward validation: {len(folds)} folds")

        for i, (train_df, val_df) in enumerate(folds, 1):
            X_tr, y_tr, feat_names = self.prepare_features(train_df)
            X_v,  y_v,  _         = self.prepare_features(val_df)

            model = LGBMTradingModel(params=model_params)
            info  = model.fit(X_tr, y_tr, X_v, y_v, feature_names=feat_names)

            y_pred, _  = model.predict(X_v)
            probas     = model.predict_proba(X_v)
            metrics    = self._evaluate(y_v, y_pred, probas)

            fold_result = FoldResult(
                fold=i,
                train_from=str(train_df.index.min().date()),
                train_to=  str(train_df.index.max().date()),
                val_from=  str(val_df.index.min().date()),
                val_to=    str(val_df.index.max().date()),
                accuracy=      metrics["accuracy"],
                f1_macro=      metrics["f1_macro"],
                precision_buy= metrics["precision_buy"],
                recall_buy=    metrics["recall_buy"],
                roc_auc=       metrics["roc_auc"],
                best_iteration=info["best_iteration"],
                n_train=len(X_tr),
                n_val=  len(X_v),
            )
            result.folds.append(fold_result)

            logger.info(
                f"Fold {i:2d} | "
                f"train {fold_result.train_from}→{fold_result.train_to} | "
                f"val {fold_result.val_from}→{fold_result.val_to} | "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
                f"roc={metrics['roc_auc']:.4f}"
            )

        logger.info(result.summary())
        return result

    def train_final(
        self,
        df: pd.DataFrame,
        model_params: dict | None = None,
        model_name: str = "lgbm_final",
    ) -> LGBMTradingModel:
        """Train final model on ALL available data, save to disk."""
        logger.info("Training final model on full dataset...")
        X, y, feat_names = self.prepare_features(df)

        # Use 10% of last data as pseudo-validation for early stopping
        split = int(len(X) * 0.9)
        X_tr, y_tr = X[:split], y[:split]
        X_v,  y_v  = X[split:], y[split:]

        model = LGBMTradingModel(params=model_params)
        model.fit(X_tr, y_tr, X_v, y_v, feature_names=feat_names)
        model.log_feature_importance(20)

        path = MODELS_DIR / f"{model_name}.pkl"
        model.save(path)
        return model

    # ── Optuna hyperparameter tuning ───────────────────────────────────────

    def tune(
        self,
        df: pd.DataFrame,
        n_trials: int = 50,
        timeout: int = 3600,  # 1 hour
    ) -> dict:
        """
        Optuna search over LightGBM hyperparams.
        Uses first 3 folds for speed. Returns best params dict.
        """
        folds = list(self.generate_folds(df))[:3]
        if not folds:
            raise RuntimeError("Not enough data for tuning")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "n_estimators":      500,
            }

            scores = []
            for train_df, val_df in folds:
                X_tr, y_tr, feat_names = self.prepare_features(train_df)
                X_v,  y_v,  _         = self.prepare_features(val_df)

                model = LGBMTradingModel(params=params)
                model.fit(X_tr, y_tr, X_v, y_v, feature_names=feat_names)
                y_pred, _ = model.predict(X_v)
                metrics   = self._evaluate(y_v, y_pred, model.predict_proba(X_v))
                scores.append(metrics["f1_macro"])

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        best = study.best_params
        logger.info(f"Optuna best F1={study.best_value:.4f} params: {best}")
        return best
