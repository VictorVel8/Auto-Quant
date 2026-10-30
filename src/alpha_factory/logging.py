"""structlog setup: console for people, JSON for unattended runs and Modal."""

import logging
import sys

import structlog


def _stderr_logger(*_args: object) -> structlog.PrintLogger:
    """A logger on the *current* sys.stderr, so a stream that was swapped and closed (a CLI test
    runner, a service manager) is never written to later."""
    return structlog.PrintLogger(file=sys.stderr)


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
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger."""
    return structlog.get_logger(name)
