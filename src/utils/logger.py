# ============================================================
# src/utils/logger.py
# ============================================================
# Structured logging setup using structlog.
#
# Why structured logging?
# Plain print() or logging.info("msg") produces flat strings
# that are hard to search and filter in production log tools
# (e.g. Datadog, CloudWatch, Azure Monitor).
#
# Structlog produces JSON in production:
#   {"event": "forecast_complete", "horizon": 24, "latency_ms": 312}
#
# In development it renders human-readable coloured output.
# ============================================================

import structlog
import logging
import sys
from src.utils.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """
    Configure the root logger and structlog processors.
    Call this once at application startup (in main.py).
    """
    # Set the standard library logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    # Choose renderer based on environment:
    # - development → ConsoleRenderer (coloured, human-readable)
    # - production  → JSONRenderer (machine-parseable)
    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.app_env == "development"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            # Merges any context variables set via structlog.contextvars
            structlog.contextvars.merge_contextvars,
            # Adds the log level string (INFO, WARNING, etc.)
            structlog.processors.add_log_level,
            # Adds ISO-8601 timestamps to every log line
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str = __name__):
    """
    Return a bound structlog logger for a given module.
    Usage: logger = get_logger(__name__)
    """
    return structlog.get_logger(name)
