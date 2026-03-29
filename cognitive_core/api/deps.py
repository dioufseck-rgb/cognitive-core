"""
Shared singletons for the API layer.
Set at startup via set_*(); accessed at request time via get_*().
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognitive_core.coordinator.runtime import Coordinator

_coordinator: "Coordinator | None" = None
_executor: ThreadPoolExecutor | None = None


def get_coordinator() -> "Coordinator":
    if _coordinator is None:
        raise RuntimeError("Coordinator not initialized — startup may have failed")
    return _coordinator


def get_executor() -> ThreadPoolExecutor:
    if _executor is None:
        raise RuntimeError("Executor not initialized — startup may have failed")
    return _executor


def set_coordinator(c: "Coordinator") -> None:
    global _coordinator
    _coordinator = c


def set_executor(e: ThreadPoolExecutor) -> None:
    global _executor
    _executor = e
