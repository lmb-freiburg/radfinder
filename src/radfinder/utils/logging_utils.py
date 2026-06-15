from __future__ import annotations

from typing import Any

from loguru import logger

from packg.log import configure_logger, get_logger_level_from_args
from visiontext.distutils import is_main_process

_LOG_ONCE_KEYS: set[str] = set()


def configure_logging(
    args: Any | None = None,
    main_level: str = "INFO",
    worker_level: str = "ERROR",
) -> dict[str, Any]:
    level = get_logger_level_from_args(args) if args is not None else main_level
    if not is_main_process():
        level = worker_level
    return configure_logger(level)


def log_info(message: str) -> None:
    logger.info(message)


def log_warning(message: str) -> None:
    logger.warning(message)


def log_error(message: str) -> None:
    logger.error(message)


def log_debug(message: str) -> None:
    logger.debug(message)


def log_once(key: str, message: str, level: str = "INFO") -> None:
    if key in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(key)
    logger.log(level, message)
