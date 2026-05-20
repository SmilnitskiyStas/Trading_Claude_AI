from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logger

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ML target parameters
TARGET_HORIZON = 4      # candles ahead
TARGET_THRESHOLD = 0.01  # 1% move = signal


class FeatureProcessor:
    """
    Computes all technical features listed in the trading system spec.
    Input: OHLCV DataFrame with datetime index, columns [open,high,low,close,volume].
    Output: DataFrame with 50+ feature columns + target.
    """

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 50:
            logger.warning("DataFrame too short for feature computation (need ≥50 rows)")
            return pd.DataFrame()

        out = df.copy()
        self._add_trend(out)
        self._add_momentum(out)
        self._add_volatility(out)
        self._add_volume(out)
        self._add_price_action(out)
        self._add_seasonality(out)
        return out

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target column:
          1  = Buy  (close rises >1% within next TARGET_HORIZON candles)
         -1  = Sell (close falls >1% within next TARGET_HORIZON candles)
          0  = Hold
        """
        if df.empty:
            return df
        out = df.copy()
        future_max = out["close"].shift(-TARGET_HORIZON).rolling(TARGET_HORIZON).max().shift(-(TARGET_HORIZON - 1))
        future_min = out["close"].shift(-TARGET_HORIZON).rolling(TARGET_HORIZON).min().shift(-(TARGET_HORIZON - 1))

        future_return_max = (future_max - out["close"]) / out["close"]
        future_return_min = (future_min - out["close"]) / out["close"]

        conditions = [
            future_return_max > TARGET_THRESHOLD,
            future_return_min < -TARGET_THRESHOLD,
        ]
        choices = [1, -1]
        out["target"] = np.select(conditions, choices, default=0)
        out["target"] = out["target"].astype(int)
        return out

    def process_and_save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        exchange: str = "binance",
    ) -> pd.DataFrame:
        """Full pipeline: features + target → save parquet → return DataFrame."""
        processed = self.process(df)
        if processed.empty:
            return processed
        processed = self.add_target(processed)
        processed.dropna(inplace=True)

        filename = PROCESSED_DIR / f"{exchange}_{symbol.replace('/', '_')}_{timeframe}.parquet"
        processed.to_parquet(filename)
        logger.info(f"Saved {len(processed)} rows to {filename}")
        return processed

    # ── Trend ──────────────────────────────────────────────────────────────

    def _add_trend(self, df: pd.DataFrame) -> None:
        close = df["close"]
        for span in [9, 21, 50, 200]:
            df[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
        for window in [20, 50]:
            df[f"sma_{window}"] = close.rolling(window).mean()

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # ADX
        df["adx_14"] = self._adx(df, 14)

    # ── Momentum ───────────────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> None:
        for period in [14, 7]:
            df[f"rsi_{period}"] = self._rsi(df["close"], period)

        # Stochastic (14,3)
        low14  = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # CCI
        typical = (df["high"] + df["low"] + df["close"]) / 3
        mean_dev = typical.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df["cci_20"] = (typical - typical.rolling(20).mean()) / (0.015 * mean_dev + 1e-10)

        # Williams %R
        df["williams_r"] = -100 * (df["high"].rolling(14).max() - df["close"]) / (
            df["high"].rolling(14).max() - df["low"].rolling(14).min() + 1e-10
        )

        # Rate of Change
        df["roc_10"] = df["close"].pct_change(10) * 100

    # ── Volatility ─────────────────────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame) -> None:
        # Bollinger Bands (20)
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_upper"]   = sma20 + 2 * std20
        df["bb_middle"]  = sma20
        df["bb_lower"]   = sma20 - 2 * std20
        df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
        df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # ATR (14)
        atr = self._atr(df, 14)
        df["atr_14"]      = atr
        df["atr_percent"] = atr / (df["close"] + 1e-10)

        # Keltner Channels (20, multiplier=2)
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        df["keltner_upper"] = ema20 + 2 * atr
        df["keltner_lower"] = ema20 - 2 * atr

    # ── Volume ─────────────────────────────────────────────────────────────

    def _add_volume(self, df: pd.DataFrame) -> None:
        vol = df["volume"]
        df["volume_sma_20"] = vol.rolling(20).mean()
        df["volume_ratio"]  = vol / (df["volume_sma_20"] + 1e-10)

        # OBV
        direction = np.sign(df["close"].diff()).fillna(0)
        df["obv"] = (direction * vol).cumsum()

        # VWAP (cumulative since start — reset per day in production)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)

        # CMF (20)
        mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + 1e-10
        ) * vol
        df["cmf_20"] = mfv.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

    # ── Price action ───────────────────────────────────────────────────────

    def _add_price_action(self, df: pd.DataFrame) -> None:
        close = df["close"]
        df["price_change_1h"]  = close.pct_change(1)
        df["price_change_4h"]  = close.pct_change(4)
        df["price_change_24h"] = close.pct_change(24)
        df["high_low_ratio"]   = df["high"] / (df["low"] + 1e-10)
        df["close_position"]   = (close - df["low"]) / (df["high"] - df["low"] + 1e-10)

    # ── Seasonality ────────────────────────────────────────────────────────

    def _add_seasonality(self, df: pd.DataFrame) -> None:
        idx = df.index
        df["hour_of_day"]  = idx.hour
        df["day_of_week"]  = idx.dayofweek
        df["day_of_month"] = idx.day
        df["month"]        = idx.month
        df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)

    # ── Technical helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        up   = high.diff()
        down = -low.diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        tr_series = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr   = pd.Series(plus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean()
        atr_t = tr_series.ewm(com=period - 1, adjust=False).mean()

        plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=period - 1, adjust=False).mean() / (atr_t + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean() / (atr_t + 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        return dx.ewm(com=period - 1, adjust=False).mean()
