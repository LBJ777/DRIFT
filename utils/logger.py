"""
utils/logger.py
---------------
Structured logging configuration for DRIFT.

Responsibility:
    Provide a consistent, configurable logging setup that writes to both
    the console and an optional file sink.  All DRIFT modules obtain their
    loggers via the standard ``logging.getLogger(__name__)`` pattern and
    benefit from whatever handlers are configured here.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "DRIFT",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = "drift.log",
    use_console: bool = True,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return the root DRIFT logger.

    Should be called once at program entry (e.g. in a training script) before
    any other DRIFT import so that all subsequent ``getLogger()`` calls inherit
    the configured handlers.

    Args:
        name: Logger name (default ``"DRIFT"``).  Child loggers use
            ``logging.getLogger("DRIFT.submodule")``.
        log_level: Log level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``.
        log_dir: Directory for the log file.  If ``None``, no file handler
            is added.
        log_filename: Name of the log file inside *log_dir*.
        use_console: Whether to emit log records to ``sys.stdout``.
        fmt: ``logging.Formatter`` format string.
        datefmt: Date/time format string.

    Returns:
        The configured ``logging.Logger`` instance.

    Example::

        from DRIFT.utils.logger import setup_logger
        logger = setup_logger(log_level="DEBUG", log_dir="./logs")
        logger.info("Training started.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False  # Prevent double-logging to root logger

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    if use_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if log_dir is not None and log_filename is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_path)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the ``"DRIFT"`` namespace.

    Convenience wrapper so sub-modules do not need to construct the full
    dotted name manually.

    Args:
        name: Short name for the sub-module, e.g. ``"trainer"``.

    Returns:
        A ``logging.Logger`` named ``"DRIFT.<name>"``.

    Example::

        logger = get_logger("trainer")
        logger.info("Epoch 1 started.")
    """
    return logging.getLogger(f"DRIFT.{name}")
