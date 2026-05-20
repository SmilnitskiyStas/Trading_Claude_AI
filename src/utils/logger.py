import sys
import io
from loguru import logger
from src.utils.config import LOG_LEVEL, LOG_DIR

# Ensure UTF-8 output on Windows to avoid UnicodeEncodeError with box-drawing chars
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup_logger() -> None:
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stdout, level=LOG_LEVEL, format=fmt, colorize=True)

    logger.add(
        LOG_DIR / "trading_{time:YYYY-MM-DD}.log",
        level=LOG_LEVEL,
        format=fmt,
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    logger.add(
        LOG_DIR / "errors.log",
        level="ERROR",
        format=fmt,
        rotation="100 MB",
        retention="90 days",
        compression="zip",
        encoding="utf-8",
    )


setup_logger()

__all__ = ["logger"]
