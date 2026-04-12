"""
Cognitive Core - Engine Package

Lazy-loading module: heavy dependencies are only imported when their
symbols are actually used.

Light imports (always available):
  - engine.state: WorkflowState, StepResult, get_step_output, etc.
  - engine.tools: ToolRegistry, create_case_registry
  - engine.actions: ActionRegistry, ActionResult, ActionSpec

Heavy imports:
  - engine.composer: compose_workflow, compile_workflow, run_workflow, etc.
  - engine.agentic_devs: run_agentic_workflow_devs, AgenticWorkflowExecutor
  - engine.nodes: create_node, create_retrieve_node, etc.
"""

# Light imports — always available
from cognitive_core.engine.state import (
    WorkflowState, StepResult, RoutingDecision,
    get_step_output, get_latest_output, get_loop_count,
    resolve_param, build_context_from_state,
)
from cognitive_core.engine.tools import ToolRegistry, create_case_registry, create_production_registry
from cognitive_core.engine.actions import ActionRegistry, ActionResult, ActionSpec, AuthLevel, create_simulation_registry


def __getattr__(name):
    """Lazy-load heavy symbols."""
    _composer_symbols = {
        "load_use_case", "load_three_layer", "merge_workflow_domain",
        "validate_use_case", "compose_workflow", "compile_workflow",
        "run_workflow", "compose_subgraph", "compile_subgraph",
        "run_workflow_from_step",
    }
    if name in _composer_symbols:
        import cognitive_core.engine.composer as _composer
        return getattr(_composer, name)

    _agentic_symbols = {
        "run_agentic_workflow_devs", "AgenticWorkflowExecutor",
        "OrchestratorStep", "AgenticWorkflowModel",
    }
    if name in _agentic_symbols:
        import cognitive_core.engine.agentic_devs as _agentic
        return getattr(_agentic, name)

    _node_symbols = {
        "create_node", "create_retrieve_node", "create_govern_node",
        "create_llm", "create_agent_router", "set_trace",
    }
    if name in _node_symbols:
        import cognitive_core.engine.nodes as _nodes
        return getattr(_nodes, name)

    raise AttributeError(f"module 'engine' has no attribute {name!r}")
