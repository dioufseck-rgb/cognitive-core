"""
Cognitive Core — Trace Infrastructure

Lightweight tracing protocol for step-by-step execution monitoring.
No heavy dependencies (no langchain, no pydantic, no langgraph).
"""

from __future__ import annotations
from typing import Protocol


class TraceCallback(Protocol):
    def on_step_start(self, step_name: str, primitive: str, loop_iteration: int) -> None: ...
    def on_llm_start(self, step_name: str, prompt_chars: int) -> None: ...
    def on_llm_end(self, step_name: str, response_chars: int, elapsed: float) -> None: ...
    def on_parse_result(self, step_name: str, primitive: str, output: dict) -> None: ...
    def on_parse_error(self, step_name: str, error: str) -> None: ...
    def on_route_decision(self, from_step: str, to_step: str, decision_type: str, reason: str) -> None: ...
    def on_retrieve_start(self, step_name: str, source_name: str) -> None: ...
    def on_retrieve_end(self, step_name: str, source_name: str, status: str, latency_ms: float) -> None: ...


class NullTrace:
    """No-op tracer when tracing is disabled."""
    def on_step_start(self, *a, **kw): pass
    def on_llm_start(self, *a, **kw): pass
    def on_llm_end(self, *a, **kw): pass
    def on_parse_result(self, *a, **kw): pass
    def on_parse_error(self, *a, **kw): pass
    def on_route_decision(self, *a, **kw): pass
    def on_retrieve_start(self, *a, **kw): pass
    def on_retrieve_end(self, *a, **kw): pass


# Global trace callback — set by the runner before execution
_trace: TraceCallback = NullTrace()


def set_trace(callback: TraceCallback):
    global _trace
    _trace = callback


def get_trace() -> TraceCallback:
    return _trace
