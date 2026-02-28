"""
Cognitive Core - Engine Package

Lazy-loading module: LangGraph and other heavy dependencies are only
imported when their symbols are actually used. This allows modules
that don't need LangGraph (coordinator, tests, tools) to import
from engine.state, engine.tools, etc. without triggering the full
dependency chain.

Light imports (no LangGraph dependency):
  - engine.state: WorkflowState, StepResult, get_step_output, etc.
  - engine.tools: ToolRegistry, create_case_registry
  - engine.actions: ActionRegistry, ActionResult, ActionSpec

Heavy imports (require LangGraph):
  - engine.composer: compose_workflow, compile_workflow, run_workflow, etc.
  - engine.agentic: compose_agentic_workflow, etc.
  - engine.nodes: create_node, create_retrieve_node, etc.
"""

# Light imports — always available, no LangGraph needed
from engine.state import (
    WorkflowState, StepResult, RoutingDecision,
    get_step_output, get_latest_output, get_loop_count,
    resolve_param, build_context_from_state,
)
from engine.tools import ToolRegistry, create_case_registry, create_production_registry
from engine.actions import ActionRegistry, ActionResult, ActionSpec, AuthLevel, create_simulation_registry


def __getattr__(name):
    """Lazy-load heavy symbols that require LangGraph."""
    # Composer symbols
    _composer_symbols = {
        "load_use_case", "load_three_layer", "merge_workflow_domain",
        "validate_use_case", "compose_workflow", "compile_workflow",
        "run_workflow", "compose_subgraph", "compile_subgraph",
        "run_workflow_from_step",
    }
    if name in _composer_symbols:
        import engine.composer as _composer
        return getattr(_composer, name)

    # Agentic symbols
    _agentic_symbols = {
        "compose_agentic_workflow", "compile_agentic_workflow",
        "run_agentic_workflow", "validate_agentic_config",
    }
    if name in _agentic_symbols:
        import engine.agentic as _agentic
        return getattr(_agentic, name)

    # Node symbols
    _node_symbols = {
        "create_node", "create_retrieve_node", "create_act_node",
        "create_llm", "create_agent_router", "set_trace",
    }
    if name in _node_symbols:
        import engine.nodes as _nodes
        return getattr(_nodes, name)

    raise AttributeError(f"module 'engine' has no attribute {name!r}")

