"""Minimal langgraph.graph shim for offline execution.

Implements the LangGraph StateGraph/CompiledGraph API contract:
  - Node functions receive the FULL accumulated state
  - Node functions return a PARTIAL update (delta)
  - The framework MERGES the delta back into state using reducers
  - List fields (steps, routing_log): append semantics (operator.add)
  - Dict fields (loop_counts): shallow merge
  - Scalar fields (current_step, input, metadata): replace
"""

from __future__ import annotations
from typing import Any, Callable, Type
import copy

END = "__end__"

# Fields that use append (operator.add) reducer in WorkflowState
_APPEND_FIELDS = {"steps", "routing_log"}

# Fields that use dict-merge reducer
_MERGE_FIELDS = {"loop_counts"}


def _merge_delta(state: dict, delta: dict) -> dict:
    """Merge a node's partial return into the accumulated state.

    Matches LangGraph's reducer semantics:
      - List fields (steps, routing_log): append new items
      - Dict fields (loop_counts): shallow merge
      - Everything else: replace
    """
    merged = dict(state)
    for key, value in delta.items():
        if key in _APPEND_FIELDS and isinstance(value, list):
            merged[key] = merged.get(key, []) + value
        elif key in _MERGE_FIELDS and isinstance(value, dict):
            prev = merged.get(key, {})
            merged[key] = {**prev, **value}
        else:
            merged[key] = value
    return merged


class CompiledGraph:
    """Compiled graph that can be invoked or streamed."""

    def __init__(self, entry: str, nodes: dict, edges: dict, cond_edges: dict):
        self._entry = entry
        self._nodes = nodes          # name → callable
        self._edges = edges          # name → next_name
        self._cond_edges = cond_edges  # name → router_fn

    def invoke(self, state: Any, config: dict | None = None) -> Any:
        current = self._entry
        while current != END:
            if current not in self._nodes:
                raise ValueError(f"Node '{current}' not found in graph")
            fn = self._nodes[current]
            delta = fn(state)
            state = _merge_delta(state, delta)

            if current in self._cond_edges:
                current = self._cond_edges[current](state)
            elif current in self._edges:
                current = self._edges[current]
            else:
                current = END
        return state

    def stream(self, state: Any, config: dict | None = None):
        """Step-by-step execution yielding {node_name: delta} after each node.

        Matches LangGraph .stream() default mode:
          1. Node receives full accumulated state
          2. Node returns partial update (delta)
          3. Delta is merged into accumulated state
          4. Yield {node_name: delta} to caller
        """
        current = self._entry
        while current != END:
            if current not in self._nodes:
                raise ValueError(f"Node '{current}' not found in graph")

            fn = self._nodes[current]
            delta = fn(state)

            # Merge delta into accumulated state (node only returns changed keys)
            state = _merge_delta(state, delta)

            # Yield the raw delta (not the full state) — matches LangGraph contract
            yield {current: delta}

            if current in self._cond_edges:
                current = self._cond_edges[current](state)
            elif current in self._edges:
                current = self._edges[current]
            else:
                current = END


class StateGraph:
    """Minimal StateGraph matching the langgraph API."""

    def __init__(self, state_schema: Type | None = None):
        self._state_schema = state_schema
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
        self._cond_edges: dict[str, Callable] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: Callable) -> None:
        self._nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        self._edges[source] = target

    def add_conditional_edges(self, source: str, router: Callable) -> None:
        self._cond_edges[source] = router

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def set_finish_point(self, name: str) -> None:
        self._edges[name] = END

    def compile(self, **kwargs) -> CompiledGraph:
        if not self._entry:
            raise ValueError("No entry point set")
        return CompiledGraph(
            entry=self._entry,
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            cond_edges=dict(self._cond_edges),
        )
