"""Tracing hooks.

Supports LangSmith (if key is set) and falls back to local JSON trace.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

_trace_log: list[dict[str, Any]] = []


def _langsmith_available() -> bool:
    settings = get_settings()
    return bool(settings.langsmith_api_key)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context with LangSmith integration and local JSON trace."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    if _langsmith_available():
        settings = get_settings()
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key or "")
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        _trace_log.append({
            "name": name,
            "attributes": span["attributes"],
            "duration_seconds": span["duration_seconds"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug("Span %s completed in %.3fs", name, span["duration_seconds"])


def export_traces(output_dir: str = "reports") -> str:
    """Export collected traces to a JSON file."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    filepath.write_text(json.dumps(_trace_log, indent=2, default=str), encoding="utf-8")
    logger.info("Traces exported to %s", filepath)
    return str(filepath)


def get_traces() -> list[dict[str, Any]]:
    """Return collected trace spans."""
    return list(_trace_log)
