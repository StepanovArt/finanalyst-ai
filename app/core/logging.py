import sys

from loguru import logger

from app.core.config import settings

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[request_id]}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def setup_logging() -> None:
    logger.remove()
    logger.configure(extra={"request_id": "-"})
    logger.add(
        sys.stdout,
        format=_LOG_FORMAT,
        level=settings.log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )
