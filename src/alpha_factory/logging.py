"""structlog setup: console for people, JSON for the mini PC and Modal."""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure structlog once per process. Logs go to stderr."""
    lvl = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger."""
    return structlog.get_logger(name)
