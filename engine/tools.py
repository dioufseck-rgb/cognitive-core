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

    Two modes:
    1. Legacy: case_data contains get_* keys with embedded tool data
    2. Fixture-based: case_data has only identity fields (claim_id, etc.)
       and tool data is loaded from cases/fixtures/<case>_tools.json

    In production, tools are MCP endpoints. This mock simulates MCP
    responses using fixture data, maintaining the same contract.
    """
    registry = ToolRegistry()

    # Check if this is a fixture-based case (no get_* keys in case_data)
    has_embedded_tools = any(
        k.startswith("get_") for k in case_data.keys()
    )

    if not has_embedded_tools:
        # Load from fixtures
        tool_data = _load_fixtures_for_case(case_data)
    else:
        # Legacy: use embedded data
        tool_data = case_data

    for key, value in tool_data.items():
        if key.startswith("_"):
            continue  # skip metadata keys

        # Check if this is a deterministic tool that needs a real implementation
        if isinstance(value, dict) and value.get("_type") == "deterministic_tool":
            _register_deterministic_tool(registry, key)
            continue

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
            description=f"MCP mock: {_key}",
        )

    # Always register identity fields from case_data as tools
    for key in ("claim_id", "policy_number", "policyholder"):
        if key in case_data and key not in tool_data:
            _k, _v = key, case_data[key]
            registry.register(
                name=_k,
                fn=lambda ctx, v=_v: {"value": v},
                description=f"Case identity: {_k}",
            )

    # Always register deterministic function tools.
    # These perform exact arithmetic that LLMs must never approximate.
    for tool_name in _DETERMINISTIC_TOOL_MAP:
        if tool_name not in registry.list_tools():
            _register_deterministic_tool(registry, tool_name)

    return registry


def _load_fixtures_for_case(case_data: dict[str, Any]) -> dict[str, Any]:
    """
    Load tool fixtures for a case. Scans cases/fixtures/ for a matching
    file based on claim_id metadata inside each fixture file, or by
    naming convention.
    """
    from pathlib import Path

    fixtures_dir = Path("cases/fixtures")
    if not fixtures_dir.exists():
        return case_data  # fallback to embedded

    claim_id = case_data.get("claim_id", "")

    # Scan all fixture files for a matching _case_id or _claim_id
    for f in sorted(fixtures_dir.glob("*_tools.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
            case_id = data.get("_case_id") or data.get("_claim_id", "")
            if case_id == claim_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue

    return case_data  # fallback


# ---------------------------------------------------------------------------
# Deterministic tool imports — real implementations without MCP
# ---------------------------------------------------------------------------

# Maps tool names to (module_path, function_name) for lazy import.
# These are tools that must produce exact results (no LLM approximation).
_DETERMINISTIC_TOOL_MAP: dict[str, tuple[str, str]] = {
    "calculate_settlement": ("engine.settlement", "calculate_settlement_from_context"),
}


def _register_deterministic_tool(registry: ToolRegistry, tool_name: str) -> None:
    """
    Register a deterministic tool by importing its real implementation.

    When running without MCP, fixture files mark deterministic tools with
    {"_type": "deterministic_tool"}. This function imports the actual
    implementation and wraps it as a tool.

    Uses spec_from_file_location to avoid triggering engine/__init__.py
    which pulls heavy dependencies (pydantic, langgraph, etc).
    """
    if tool_name not in _DETERMINISTIC_TOOL_MAP:
        registry.register(
            name=tool_name,
            fn=lambda ctx: {"error": f"Deterministic tool '{tool_name}' has no registered implementation"},
            description=f"Unimplemented deterministic tool: {tool_name}",
        )
        return

    module_path, fn_name = _DETERMINISTIC_TOOL_MAP[tool_name]

    try:
        import importlib.util
        from pathlib import Path

        # Convert dotted module path to file path
        # e.g. "engine.settlement" → "engine/settlement.py"
        parts = module_path.split(".")
        file_path = Path(*parts).with_suffix(".py")
        if not file_path.is_absolute():
            # Try relative to this file's parent (project root)
            project_root = Path(__file__).parent.parent
            file_path = project_root / file_path

        spec = importlib.util.spec_from_file_location(module_path, str(file_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        real_fn = getattr(mod, fn_name)
    except (ImportError, AttributeError, FileNotFoundError, OSError) as e:
        registry.register(
            name=tool_name,
            fn=lambda ctx, err=str(e): {"error": f"Failed to import {tool_name}: {err}"},
            description=f"Failed import: {tool_name}",
        )
        return

    def wrapper(context: dict[str, Any], _fn=real_fn) -> dict[str, Any]:
        """
        Adapt the tool to ToolRegistry's (dict → dict) signature.

        Context-aware tools (name ends with _from_context) receive the
        full context dict. Raw tools receive keyword arguments from
        context["tool_input"] or the full context.
        """
        if fn_name.endswith("_from_context"):
            # Context-aware: pass entire context dict
            result = _fn(context)
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return {"raw_result": result}
            return result if isinstance(result, dict) else {"raw_result": result}

        # Raw tool: pass as kwargs
        tool_input = context.get("tool_input", context)
        result_str = _fn(**tool_input)

        try:
            return json.loads(result_str) if isinstance(result_str, str) else result_str
        except (json.JSONDecodeError, TypeError):
            return {"raw_result": result_str}

    registry.register(
        name=tool_name,
        fn=wrapper,
        description=f"Deterministic tool: {tool_name}",
    )


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
