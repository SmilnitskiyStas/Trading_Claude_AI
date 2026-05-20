from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    BigInteger, Boolean, Double, Integer, SmallInteger,
    String, Text, UniqueConstraint, Index,
    func,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.utils.config import DATABASE_URL
from src.utils.logger import logger


def _make_async_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


_ASYNC_URL = _make_async_url(DATABASE_URL)
_IS_SQLITE = "sqlite" in _ASYNC_URL

engine = create_async_engine(
    _ASYNC_URL,
    echo=False,
    pool_pre_ping=True,
    **( {"connect_args": {"check_same_thread": False}} if _IS_SQLITE else
        {"pool_size": 5, "max_overflow": 10} ),
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── ORM Models ─────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class OHLCVData(Base):
    __tablename__ = "ohlcv_data"
    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timeframe", "timestamp"),
        Index("idx_ohlcv_symbol_tf", "symbol", "timeframe", "timestamp"),
    )

    id:        Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange:  Mapped[str]   = mapped_column(String(20), nullable=False)
    symbol:    Mapped[str]   = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str]   = mapped_column(String(5),  nullable=False)
    timestamp: Mapped[int]   = mapped_column(BigInteger, nullable=False)
    open:      Mapped[float | None] = mapped_column(Double)
    high:      Mapped[float | None] = mapped_column(Double)
    low:       Mapped[float | None] = mapped_column(Double)
    close:     Mapped[float | None] = mapped_column(Double)
    volume:    Mapped[float | None] = mapped_column(Double)


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("idx_trades_symbol", "symbol", "entry_time"),
    )

    id:              Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol:          Mapped[str | None]    = mapped_column(String(20))
    exchange:        Mapped[str | None]    = mapped_column(String(20))
    side:            Mapped[str | None]    = mapped_column(String(10))
    entry_price:     Mapped[float | None]  = mapped_column(Double)
    exit_price:      Mapped[float | None]  = mapped_column(Double)
    quantity:        Mapped[float | None]  = mapped_column(Double)
    pnl:             Mapped[float | None]  = mapped_column(Double)
    pnl_percent:     Mapped[float | None]  = mapped_column(Double)
    entry_time:      Mapped[datetime | None] = mapped_column()
    exit_time:       Mapped[datetime | None] = mapped_column()
    exit_reason:     Mapped[str | None]    = mapped_column(String(50))
    ml_confidence:   Mapped[float | None]  = mapped_column(Double)
    agent_sentiment: Mapped[str | None]    = mapped_column(String(20))
    is_paper:        Mapped[bool]          = mapped_column(Boolean, default=True)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id:              Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp:       Mapped[datetime]      = mapped_column(server_default=func.now())
    total_value:     Mapped[float | None]  = mapped_column(Double)
    cash_balance:    Mapped[float | None]  = mapped_column(Double)
    positions_value: Mapped[float | None]  = mapped_column(Double)
    daily_pnl:       Mapped[float | None]  = mapped_column(Double)
    total_pnl:       Mapped[float | None]  = mapped_column(Double)
    drawdown:        Mapped[float | None]  = mapped_column(Double)
    sharpe_ratio:    Mapped[float | None]  = mapped_column(Double)


class ExchangeStatusRecord(Base):
    __tablename__ = "exchange_status"

    id:            Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange:      Mapped[str | None]   = mapped_column(String(20))
    timestamp:     Mapped[datetime]     = mapped_column(server_default=func.now())
    is_online:     Mapped[bool | None]  = mapped_column(Boolean)
    latency_ms:    Mapped[int | None]   = mapped_column(Integer)
    error_message: Mapped[str | None]   = mapped_column(Text)


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"

    id:              Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol:          Mapped[str | None]    = mapped_column(String(20))
    timestamp:       Mapped[datetime | None] = mapped_column()
    headline:        Mapped[str | None]    = mapped_column(Text)
    sentiment_score: Mapped[float | None]  = mapped_column(Double)
    source:          Mapped[str | None]    = mapped_column(String(100))
    url:             Mapped[str | None]    = mapped_column(Text)


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id:            Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp:     Mapped[datetime | None] = mapped_column()
    symbol:        Mapped[str | None]    = mapped_column(String(20))
    exchange:      Mapped[str | None]    = mapped_column(String(20))
    signal:        Mapped[int | None]    = mapped_column(SmallInteger)
    confidence:    Mapped[float | None]  = mapped_column(Double)
    features_json: Mapped[str | None]    = mapped_column(Text)


# ── Lifecycle ──────────────────────────────────────────────────────────────

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified via ORM")


async def check_connection() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info(f"Database connection OK: {DATABASE_URL.split('?')[0]}")
        return True
    except Exception as e:
        logger.error(f"Database connection FAILED: {e}")
        return False


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session
