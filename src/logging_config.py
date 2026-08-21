"""Logging configuration with per-level ANSI colors.

Call :func:`configure_logging` once at program start (the CLI entry
point does this) to get colored console output:

* DEBUG   -> gray
* INFO    -> blue
* WARNING -> yellow
* ERROR / CRITICAL -> red
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Optional

# ANSI escape codes used by the formatter.
_GRAY = "\033[90m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

_LEVEL_COLORS: Dict[int, str] = {
    logging.DEBUG: _GRAY,
    logging.INFO: _BLUE,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _RED,
}


class ColorFormatter(logging.Formatter):
    """Formatter that wraps each record's level name in its level color."""

    def __init__(self, fmt: str) -> None:
        """Create a formatter with *fmt* as the base format string."""
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        """Return *record* rendered with a colored level name."""
        color = _LEVEL_COLORS.get(record.levelno, _RESET)
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            # Restore the record so other handlers are unaffected.
            record.levelname = original_levelname


class SafeStreamHandler(logging.StreamHandler):  # pragma: no cover
    """Stream handler that survives closed/broken output streams.

    Prevents a crash at interpreter shutdown when logging after the
    stream has been torn down.
    """

    def __init__(self, stream: Optional[object] = None) -> None:
        """Initialize the handler on an optional *stream*."""
        target = sys.stderr if stream is None else stream
        super().__init__(target)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit *record*, swallowing any I/O error."""
        try:
            super().emit(record)
            self.flush()
        except (ValueError, OSError):
            pass

    def handleError(self, record: logging.LogRecord) -> None:
        """Silently ignore errors instead of printing a traceback."""


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with colored console output.

    Args:
        level: Minimum severity to display (default ``INFO``).
    """
    handler = SafeStreamHandler(sys.stderr)
    handler.setFormatter(
        ColorFormatter("%(asctime)s %(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


__all__ = [
    "ColorFormatter",
    "SafeStreamHandler",
    "configure_logging",
]
