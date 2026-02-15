"""
Cognitive Core - Tool Registry

Data source abstraction for the Retrieve primitive.

A "tool" is a callable that takes a query context (dict) and returns
data (dict). Tools are registered by name and organized by domain.
The Retrieve primitive's node calls tools, assembles results, then
hands the assembled data + metadata to the LLM for quality assessment.

Usage:
    # Register tools for a domain
    registry = ToolRegistry()
    registry.register("member_profile", my_member_lookup_fn)
    registry.register("transaction_detail", my_txn_lookup_fn)

    # At workflow build time, pass registry to create_retrieve_node
    node = create_retrieve_node(..., tool_registry=registry)

In production, tools wrap API calls to core banking, card processing,
fraud scoring, etc. In dev/test, tools return canned data from the
case JSON — making the three-layer architecture work unchanged.
"""

import json
import time
from typing import Any, Callable, Protocol
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tool protocol
# ---------------------------------------------------------------------------

class DataTool(Protocol):
    """A data source that can be called to retrieve information."""
    def __call__(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            context: Query context — typically contains case input
                     and any relevant workflow state.
        Returns:
            Retrieved data as a dict.
        Raises:
            Exception: If retrieval fails.
        """
        ...


@dataclass
class ToolResult:
    """Result of calling a single tool."""
    source: str
    status: str  # success | failed | skipped
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    freshness: str | None = None


@dataclass
class ToolSpec:
    """Registration entry for a tool."""
    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""
    latency_hint_ms: float = 0.0  # expected latency, for planning
    required: bool = False  # if True, workflow fails on tool failure


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central registry of data source tools.

    Supports two patterns:
    1. Global registration (tools available to all workflows)
    2. Domain-scoped registration (tools for specific domains)
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
        latency_hint_ms: float = 0.0,
        required: bool = False,
    ):
        """Register a data source tool."""
        self._tools[name] = ToolSpec(
            name=name,
            fn=fn,
            description=description,
            latency_hint_ms=latency_hint_ms,
            required=required,
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> str:
        """Human-readable description of all registered tools (for LLM prompts)."""
        if not self._tools:
            return "No data sources registered."
        lines = []
        for spec in self._tools.values():
            req = " (REQUIRED)" if spec.required else ""
            lines.append(f"  - {spec.name}{req}: {spec.description}")
            if spec.latency_hint_ms > 0:
                lines.append(f"    Expected latency: ~{spec.latency_hint_ms:.0f}ms")
        return "\n".join(lines)

    def call(self, name: str, context: dict[str, Any]) -> ToolResult:
        """Call a tool by name with the given context."""
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                source=name,
                status="failed",
                error=f"Tool '{name}' not registered",
            )

        t0 = time.time()
        try:
            data = spec.fn(context)
            elapsed_ms = (time.time() - t0) * 1000
            return ToolResult(
                source=name,
                status="success",
                data=data,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            return ToolResult(
                source=name,
                status="failed",
                error=str(e),
                latency_ms=elapsed_ms,
            )

    def call_many(
        self,
        names: list[str],
        context: dict[str, Any],
    ) -> list[ToolResult]:
        """Call multiple tools sequentially. Returns results in order."""
        return [self.call(name, context) for name in names]


# ---------------------------------------------------------------------------
# Case-passthrough registry — for dev/test
# ---------------------------------------------------------------------------

def create_case_registry(case_data: dict[str, Any]) -> ToolRegistry:
    """
    Create a ToolRegistry that serves data from a case JSON.

    In dev/test, the case file IS the data source. Each top-level key
    in the case JSON becomes a tool that returns that key's value.

    This means existing case files work unchanged — the Retrieve step
    just formalizes what was previously implicit (everything loaded upfront).

    Example case JSON:
        {
            "member_profile": {...},
            "transaction_detail": {...},
            "fraud_score": {...}
        }

    Produces tools: member_profile, transaction_detail, fraud_score
    """
    registry = ToolRegistry()

    for key, value in case_data.items():
        # Capture key/value in closure
        _key = key
        _value = value

        def make_fn(k, v):
            def tool_fn(context: dict[str, Any]) -> dict[str, Any]:
                if isinstance(v, dict):
                    return v
                return {"value": v}
            return tool_fn

        registry.register(
            name=_key,
            fn=make_fn(_key, _value),
            description=f"Case data: {_key}",
        )

    return registry


# ---------------------------------------------------------------------------
# Production registry builder — template for real integrations
# ---------------------------------------------------------------------------

def create_production_registry() -> ToolRegistry:
    """
    Template for production tool registration.

    In production, replace these stubs with actual API clients.
    Each tool wraps a service call and returns structured data.
    """
    registry = ToolRegistry()

    # Example: these would be actual API calls in production
    # registry.register(
    #     name="member_profile",
    #     fn=lambda ctx: core_banking_client.get_member(ctx["member_id"]),
    #     description="Member profile from core banking (demographics, tenure, products)",
    #     latency_hint_ms=50,
    #     required=True,
    # )
    # registry.register(
    #     name="transaction_detail",
    #     fn=lambda ctx: card_processor.get_transaction(ctx["transaction_id"]),
    #     description="Full transaction record from card processor",
    #     latency_hint_ms=80,
    #     required=True,
    # )
    # registry.register(
    #     name="fraud_score",
    #     fn=lambda ctx: fraud_scoring_api.score(ctx["transaction_id"]),
    #     description="Real-time fraud risk score (0-1000)",
    #     latency_hint_ms=200,
    # )

    return registry
