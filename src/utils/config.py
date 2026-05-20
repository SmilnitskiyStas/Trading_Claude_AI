from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _get(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required env variable '{key}' is not set. Check your .env file.")
    return value


# ── Exchanges ──────────────────────────────────────────────────────────────

ACTIVE_EXCHANGES: list[str] = _get("ACTIVE_EXCHANGES", "binance").split(",")
PRIMARY_EXCHANGE: str = _get("PRIMARY_EXCHANGE", "binance")

EXCHANGE_CREDENTIALS: dict[str, dict] = {
    "binance": {
        "api_key": _get("BINANCE_API_KEY", ""),
        "secret":  _get("BINANCE_SECRET", ""),
    },
    "bybit": {
        "api_key": _get("BYBIT_API_KEY", ""),
        "secret":  _get("BYBIT_SECRET", ""),
    },
    "kraken": {
        "api_key": _get("KRAKEN_API_KEY", ""),
        "secret":  _get("KRAKEN_SECRET", ""),
    },
    "okx": {
        "api_key":    _get("OKX_API_KEY", ""),
        "secret":     _get("OKX_SECRET", ""),
        "passphrase": _get("OKX_PASSPHRASE", ""),
    },
}

EXCHANGE_RATE_LIMITS: dict[str, int] = {
    "binance": 1200,
    "bybit":   600,
    "kraken":  60,
    "okx":     300,
}

# ── Symbols ────────────────────────────────────────────────────────────────

SYMBOLS: list[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "TRX/USDT", "LINK/USDT",
]

TIMEFRAMES: list[str] = ["1h", "4h", "1d"]
PRIMARY_TIMEFRAME: str = "1h"

# ── Database ───────────────────────────────────────────────────────────────

DATABASE_URL: str = _get("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/trading.db")

# ── Redis ──────────────────────────────────────────────────────────────────

REDIS_URL: str = _get("REDIS_URL", "redis://localhost:6379")
REDIS_DB: int = int(_get("REDIS_DB", "0"))

# ── Trading ────────────────────────────────────────────────────────────────

PAPER_TRADING: bool = _get("PAPER_TRADING", "true").lower() == "true"
INITIAL_BALANCE: float = float(_get("INITIAL_BALANCE", "10000"))
MAX_POSITION_SIZE: float = float(_get("MAX_POSITION_SIZE", "0.10"))
STOP_LOSS: float = float(_get("STOP_LOSS", "0.03"))
TAKE_PROFIT: float = float(_get("TAKE_PROFIT", "0.06"))
MAX_DAILY_LOSS: float = float(_get("MAX_DAILY_LOSS", "0.05"))

# ── AI Agent ───────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = _get("OPENAI_API_KEY", "")
# "anthropic" or "openai" — whichever key is set; anthropic takes priority
AGENT_PROVIDER: str = _get("AGENT_PROVIDER", "anthropic")
AGENT_MODEL: str = "claude-sonnet-4-6"
OPENAI_MODEL: str = _get("OPENAI_MODEL", "gpt-4o-mini")
AGENT_CALL_INTERVAL_SECONDS: int = 3600

# ── News ───────────────────────────────────────────────────────────────────

CRYPTOPANIC_API_KEY: str = _get("CRYPTOPANIC_API_KEY", "")

# ── Telegram ───────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN: str = _get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = _get("TELEGRAM_CHAT_ID", "")

# ── Dashboard ──────────────────────────────────────────────────────────────

DASHBOARD_HOST: str = _get("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT: int = int(_get("DASHBOARD_PORT", "8080"))

# ── Logging ────────────────────────────────────────────────────────────────

LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")
LOG_DIR: Path = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
