"""
Cognitive Core — Agentic Composer

Builds a hub-and-spoke LangGraph where a central orchestrator LLM
decides which primitive to invoke next. The workflow YAML declares
the available primitives, constraints, and goal — but the orchestrator
chooses the sequence at runtime.

Graph structure:
    orchestrator ──→ retrieve ──→ orchestrator
                 ──→ classify ──→ orchestrator
                 ──→ investigate ──→ orchestrator
                 ──→ verify ──→ orchestrator
                 ──→ generate ──→ orchestrator
                 ──→ challenge ──→ orchestrator
                 ──→ END

The orchestrator sees all accumulated state and decides:
  - Which primitive to invoke next (or END)
  - What step_name to give it (e.g., "classify_inquiry" vs "classify_urgency")
  - What params to pass (from the domain config's primitives section)
"""

import json
import time
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
from engine.llm import create_llm

from registry.primitives import render_prompt, get_schema_class, PRIMITIVE_CONFIGS
from engine.state import (
    WorkflowState,
    StepResult,
    RoutingDecision,
    build_context_from_state,
    get_step_output,
)
from engine.nodes import (
    create_node,
    create_retrieve_node,
    get_trace,
    extract_json,
)
from engine.tools import ToolRegistry, create_case_registry


# ---------------------------------------------------------------------------
# Orchestrator prompt — loaded from registry like all other primitives
# ---------------------------------------------------------------------------

_PROMPT_FILE = Path(__file__).parent.parent / "registry" / "prompts" / "orchestrator.txt"

def _load_orchestrator_prompt() -> str:
    return _PROMPT_FILE.read_text()



# ---------------------------------------------------------------------------
# Validation for agentic configs
# ---------------------------------------------------------------------------

def validate_agentic_config(config: dict[str, Any]) -> list[str]:
    """Validate an agentic workflow configuration."""
    errors = []
    if config.get("mode") != "agentic":
        errors.append("Config mode must be 'agentic'")
    if "goal" not in config:
        errors.append("Missing 'goal'")
    if "available_primitives" not in config:
        errors.append("Missing 'available_primitives'")
    else:
        valid = set(PRIMITIVE_CONFIGS.keys())
        for p in config["available_primitives"]:
            if p not in valid:
                errors.append(f"Unknown primitive: '{p}' (valid: {valid})")

    constraints = config.get("constraints", {})
    if "max_steps" not in constraints:
        errors.append("Missing constraints.max_steps")

    if "orchestrator" not in config:
        errors.append("Missing 'orchestrator' section")

    if "primitive_configs" not in config:
        errors.append("Missing 'primitive_configs' — domain-specific params for each invocation")

    return errors


# ---------------------------------------------------------------------------
# Agentic composer
# ---------------------------------------------------------------------------

def compose_agentic_workflow(
    config: dict[str, Any],
    model: str = "default",
    temperature: float = 0.1,
    tool_registry: ToolRegistry | None = None,
) -> StateGraph:
    """
    Build a hub-and-spoke LangGraph from an agentic workflow config.

    The graph has:
      - One orchestrator node (hub)
      - One node per available primitive (spokes)
      - Conditional edges from orchestrator to each primitive or END
      - Unconditional edges from each primitive back to orchestrator
    """
    errors = validate_agentic_config(config)
    if errors:
        raise ValueError("Invalid agentic config:\n" + "\n".join(f"  - {e}" for e in errors))

    available = config["available_primitives"]
    constraints = config.get("constraints", {})
    max_steps = constraints.get("max_steps", 10)
    max_repeat = constraints.get("max_repeat", 3)
    must_include = set(constraints.get("must_include", []))
    must_end_with = constraints.get("must_end_with", None)
    challenge_must_pass = constraints.get("challenge_must_pass", True)

    orchestrator_config = config.get("orchestrator", {})
    orch_model_name = orchestrator_config.get("model", model)
    strategy = orchestrator_config.get("strategy", "")

    goal = config["goal"]
    primitive_configs = config.get("primitive_configs", {})

    trace = get_trace()

    # Build LLM for orchestrator
    orch_llm = create_llm(
        model=orch_model_name,
        temperature=temperature,
    )

    # ── Orchestrator node ─────────────────────────────────────────

    def orchestrator_node(state: WorkflowState) -> dict:
        """Central decision-maker. Decides which primitive to invoke next."""
        steps_completed = state.get("steps", [])
        step_count = len(steps_completed)
        routing_log = state.get("routing_log", [])

        # Build steps summary
        if steps_completed:
            steps_summary_parts = []
            for s in steps_completed:
                out = s["output"]
                summary = f"  {s['step_name']} ({s['primitive']}): "

                # Flag failed steps prominently
                if out.get("error"):
                    summary += f"⚠ FAILED: {str(out['error'])[:100]}"
                    steps_summary_parts.append(summary)
                    continue

                if s["primitive"] == "classify":
                    summary += f"→ {out.get('category', '?')} (conf: {out.get('confidence', '?')})"
                elif s["primitive"] == "investigate":
                    summary += f"→ {str(out.get('finding', '?'))[:150]}"
                elif s["primitive"] == "generate":
                    summary += f"→ {str(out.get('artifact', '?'))[:150]}"
                elif s["primitive"] == "challenge":
                    surv = out.get("survives", "?")
                    vulns = len(out.get("vulnerabilities", []))
                    summary += f"→ survives={surv}, vulnerabilities={vulns}"
                elif s["primitive"] == "verify":
                    summary += f"→ conforms={out.get('conforms', '?')}"
                elif s["primitive"] == "retrieve":
                    data_keys = list(out.get("data", {}).keys())
                    summary += f"→ sources: {data_keys}"
                elif s["primitive"] == "think":
                    conclusions = out.get("conclusions", [])
                    summary += f"→ {len(conclusions)} conclusions, decision: {out.get('decision', 'none')}"
                steps_summary_parts.append(summary)
            steps_summary = "\n".join(steps_summary_parts)
        else:
            steps_summary = "  (none — this is the first step)"

        routing_summary = "\n".join(
            f"  {r['from_step']} → {r['to_step']} ({r['decision_type']}): {r['reason']}"
            for r in routing_log
        ) if routing_log else "  (no routing decisions yet)"

        # Build available primitives description
        prims_desc = []
        for p in available:
            configs_for_p = [k for k, v in primitive_configs.items()
                            if v.get("primitive") == p]
            prims_desc.append(f"  {p}: configs={configs_for_p}")
        prims_text = "\n".join(prims_desc)

        # Build primitive configs summary
        configs_text = json.dumps(
            {k: {"primitive": v.get("primitive"), "params": list(v.get("params", {}).keys())}
             for k, v in primitive_configs.items()},
            indent=2,
        )

        constraints_text = (
            f"max_steps: {max_steps}\n"
            f"max_repeat per step_name: {max_repeat}\n"
            f"must_include: {list(must_include)}\n"
            f"must_end_with: {must_end_with}\n"
            f"challenge_must_pass: {challenge_must_pass}"
        )

        prompt = _load_orchestrator_prompt().format(
            goal=goal,
            available_primitives=prims_text,
            primitive_configs=configs_text,
            constraints=constraints_text,
            steps_completed=steps_summary,
            routing_log=routing_summary,
            step_count=step_count,
            max_steps=max_steps,
            max_repeat=max_repeat,
            strategy=strategy,
        )

        # ── Deterministic overrides before calling LLM ────────────

        # Force END if max steps reached
        if step_count >= max_steps:
            trace.on_route_decision(
                "orchestrator", "__end__", "loop_limit",
                f"Reached max_steps={max_steps}"
            )
            return {
                "routing_log": [RoutingDecision(
                    from_step="orchestrator",
                    to_step="__end__",
                    decision_type="loop_limit",
                    reason=f"Reached max_steps={max_steps}",
                    agent_reasoning="",
                )],
                "metadata": {**state.get("metadata", {}), "_orch_action": "end"},
            }

        # If last step was a challenge that passed AND must_end_with is challenge
        if steps_completed:
            last = steps_completed[-1]
            if (last["primitive"] == "challenge"
                    and last["output"].get("survives") is True
                    and must_end_with == "challenge"):
                # Check must_include satisfaction
                executed_prims = {s["primitive"] for s in steps_completed}
                missing = must_include - executed_prims
                if not missing:
                    trace.on_route_decision(
                        "orchestrator", "__end__", "deterministic",
                        "Challenge passed, all must_include satisfied"
                    )
                    return {
                        "routing_log": [RoutingDecision(
                            from_step="orchestrator",
                            to_step="__end__",
                            decision_type="deterministic",
                            reason="Challenge passed, all must_include satisfied",
                            agent_reasoning="",
                        )],
                        "metadata": {**state.get("metadata", {}), "_orch_action": "end"},
                    }

        # ── LLM decides ──────────────────────────────────────────

        trace.on_step_start("orchestrator", "orchestrator", 0)
        trace.on_llm_start("orchestrator", len(prompt))
        t0 = time.time()
        response = orch_llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.time() - t0
        trace.on_llm_end("orchestrator", len(response.content), elapsed)

        try:
            decision = extract_json(response.content)
        except Exception:
            # Can't parse — force end
            decision = {"action": "end", "reasoning": "Failed to parse orchestrator response"}

        action = decision.get("action", "end")

        if action == "end":
            trace.on_route_decision(
                "orchestrator", "__end__", "agent",
                decision.get("reasoning", "Goal met")
            )
            return {
                "routing_log": [RoutingDecision(
                    from_step="orchestrator",
                    to_step="__end__",
                    decision_type="agent",
                    reason=decision.get("reasoning", "Goal met"),
                    agent_reasoning=json.dumps(decision),
                )],
                "metadata": {**state.get("metadata", {}), "_orch_action": "end"},
            }

        # Action is "invoke"
        primitive = decision.get("primitive", "")
        step_name = decision.get("step_name", primitive)
        params_key = decision.get("params_key", step_name)

        # Validate primitive
        if primitive not in available:
            # Fall back to end
            trace.on_route_decision(
                "orchestrator", "__end__", "agent",
                f"Invalid primitive '{primitive}', ending"
            )
            return {
                "routing_log": [RoutingDecision(
                    from_step="orchestrator",
                    to_step="__end__",
                    decision_type="agent",
                    reason=f"Invalid primitive '{primitive}'",
                    agent_reasoning=json.dumps(decision),
                )],
                "metadata": {**state.get("metadata", {}), "_orch_action": "end"},
            }

        trace.on_route_decision(
            "orchestrator", step_name, "agent",
            decision.get("reasoning", "")
        )

        return {
            "routing_log": [RoutingDecision(
                from_step="orchestrator",
                to_step=step_name,
                decision_type="agent",
                reason=decision.get("reasoning", ""),
                agent_reasoning=json.dumps(decision),
            )],
            "metadata": {
                **state.get("metadata", {}),
                "_orch_action": "invoke",
                "_orch_primitive": primitive,
                "_orch_step_name": step_name,
                "_orch_params_key": params_key,
            },
        }

    # ── Primitive executor node ───────────────────────────────────

    def primitive_executor(state: WorkflowState) -> dict:
        """
        Reads the orchestrator's decision from metadata and executes
        the chosen primitive. This is a single node that dispatches
        to any primitive based on the orchestrator's instruction.
        """
        meta = state.get("metadata", {})
        primitive = meta.get("_orch_primitive", "")
        step_name = meta.get("_orch_step_name", primitive)
        params_key = meta.get("_orch_params_key", step_name)

        # Get params from primitive_configs
        prim_config = primitive_configs.get(params_key, {})
        params = prim_config.get("params", {})

        # LLM parameter cascade: primitive_config → workflow default
        step_model = prim_config.get("model", model)
        step_temperature = prim_config.get("temperature", temperature)

        # Resolve ${...} references in params
        from engine.state import resolve_param
        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str):
                resolved_params[k] = resolve_param(v, state)
            else:
                resolved_params[k] = v

        # Auto-build context if not provided
        if "context" not in resolved_params:
            input_ctx = json.dumps(state["input"], indent=2) if state.get("input") else ""
            prior_ctx = build_context_from_state(state)
            resolved_params["context"] = f"Workflow Input:\n{input_ctx}\n\n{prior_ctx}"

        if primitive == "retrieve":
            if tool_registry is None:
                raise ValueError("Retrieve primitive requires a tool_registry")
            node_fn = create_retrieve_node(
                step_name=step_name,
                params=resolved_params,
                tool_registry=tool_registry,
                model=step_model,
                temperature=step_temperature,
            )
            return node_fn(state)
        else:
            node_fn = create_node(
                step_name=step_name,
                primitive_name=primitive,
                params=resolved_params,
                model=step_model,
                temperature=step_temperature,
            )
            return node_fn(state)

    # ── Router function ───────────────────────────────────────────

    def orchestrator_router(state: WorkflowState) -> str:
        """Route from orchestrator to primitive_executor or END."""
        meta = state.get("metadata", {})
        action = meta.get("_orch_action", "end")
        if action == "end":
            return END
        return "primitive_executor"

    # ── Build the graph ───────────────────────────────────────────

    graph = StateGraph(WorkflowState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("primitive_executor", primitive_executor)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", orchestrator_router)
    graph.add_edge("primitive_executor", "orchestrator")

    return graph


# ---------------------------------------------------------------------------
# Compile / run helpers
# ---------------------------------------------------------------------------

def compile_agentic_workflow(config, model="default", temperature=0.1, tool_registry=None):
    return compose_agentic_workflow(config, model, temperature, tool_registry).compile()


def run_agentic_workflow(config, workflow_input, model="default", temperature=0.1, tool_registry=None):
    compiled = compile_agentic_workflow(config, model, temperature, tool_registry)
    initial: WorkflowState = {
        "input": workflow_input,
        "steps": [],
        "current_step": "",
        "metadata": {
            "use_case": config.get("name", ""),
            "description": config.get("description", ""),
            "mode": "agentic",
        },
        "loop_counts": {},
        "routing_log": [],
    }
    return compiled.invoke(initial)
