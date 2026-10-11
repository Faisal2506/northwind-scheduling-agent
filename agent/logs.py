"""Python logs for the clinic process. Same lines go to the terminal and the page."""

from __future__ import annotations

import logging
import sys

_LOGGERS = ("northwind", "httpx", "httpcore", "google_genai", "langchain", "langgraph")


class MemoryLog(logging.Handler):
    def __init__(self, limit: int = 300):
        super().__init__()
        self.limit = limit
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))
        del self.lines[: -self.limit]


MEMORY = MemoryLog()
MEMORY.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))


def setup_logging() -> None:
    if getattr(setup_logging, "_ready", False):
        return
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(MEMORY.formatter)
    for name in _LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(MEMORY)
        logger.addHandler(stream)
        logger.propagate = False
    setup_logging._ready = True


def recent(limit: int = 80) -> str:
    if not MEMORY.lines:
        return "No Python log lines yet. Send a message, then look here and in the terminal that is running streamlit."
    return "\n".join(MEMORY.lines[-limit:])


log = logging.getLogger("northwind")
