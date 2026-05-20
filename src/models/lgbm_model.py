from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np

from src.utils.logger import logger

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Class mapping: LightGBM needs 0-based consecutive integers
# Sell(-1) → 0,  Hold(0) → 1,  Buy(1) → 2
SIGNAL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_SIGNAL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = {0: "Sell", 1: "Hold", 2: "Buy"}

DEFAULT_PARAMS: dict = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "n_jobs":            -1,          # use all 20 cores of i7-14700
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "class_weight":      "balanced",  # crucial: Hold >> Buy/Sell
    "verbose":           -1,
    "random_state":      42,
}


class LGBMTradingModel:
    """
    LightGBM 3-class trading classifier.
    Classes: 0=Sell, 1=Hold, 2=Buy  (mapped from signal -1/0/+1)
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Train with early stopping on validation set. Returns train/val metrics."""
        # Map signals (-1,0,1) to classes (0,1,2)
        y_tr = np.vectorize(SIGNAL_TO_CLASS.get)(y_train)
        y_v  = np.vectorize(SIGNAL_TO_CLASS.get)(y_val)

        p = {k: v for k, v in self.params.items()
             if k not in ("n_estimators", "early_stopping_rounds")}

        self.model = lgb.LGBMClassifier(
            n_estimators=self.params["n_estimators"],
            **p,
        )
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        self.model.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_v)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        best_iter = self.model.best_iteration_
        logger.info(f"Best iteration: {best_iter}")
        return {"best_iteration": best_iter}

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            signals     : array of int (-1, 0, 1)
            confidences : array of float (max class probability)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        proba = self.model.predict_proba(X)           # shape (N, 3)
        classes = proba.argmax(axis=1)                # 0/1/2
        signals = np.vectorize(CLASS_TO_SIGNAL.get)(classes)
        confidences = proba.max(axis=1)
        return signals, confidences

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "params": self.params}, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LGBMTradingModel":
        data = joblib.load(path)
        obj = cls(params=data["params"])
        obj.model = data["model"]
        obj.feature_names = data["features"]
        logger.info(f"Model loaded ← {path} ({len(obj.feature_names)} features)")
        return obj

    # ── Feature importance ─────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> list[tuple[str, int]]:
        if self.model is None:
            return []
        imp = self.model.feature_importances_
        pairs = sorted(zip(self.feature_names, imp), key=lambda x: -x[1])
        return pairs[:top_n]

    def log_feature_importance(self, top_n: int = 20) -> None:
        logger.info(f"Top-{top_n} feature importance:")
        for name, score in self.feature_importance(top_n):
            logger.info(f"  {name:<35} {score:>6}")
