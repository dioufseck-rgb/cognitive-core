"""
Cognitive Core - Composition Engine

Three-layer architecture:
  Workflow (structure) + Domain (expertise) + Case (data) → Runnable graph

Also supports single-file configs for backward compatibility.
"""

import yaml
import re
import json
import copy
from typing import Any
from pathlib import Path

from langgraph.graph import StateGraph, END

from registry.primitives import validate_use_case_step
from engine.state import WorkflowState, get_step_output, get_loop_count
from engine.nodes import create_node, create_retrieve_node, create_agent_router, get_trace
from engine.tools import ToolRegistry, create_case_registry


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_three_layer(
    workflow_path: str | Path,
    domain_path: str | Path,
    case_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load workflow + domain + case. Returns (merged_config, case_input)."""
    workflow = load_yaml(workflow_path)
    domain = load_yaml(domain_path)
    case_input = {}
    if case_path:
        p = Path(case_path)
        if p.suffix == ".json":
            case_input = load_json(p)
        else:
            d = load_yaml(p)
            case_input = d.get("input", d)
    merged = merge_workflow_domain(workflow, domain)
    return merged, case_input


def load_use_case(config_path: str | Path) -> dict[str, Any]:
    """Load single-file config (backward compat)."""
    return load_yaml(config_path)


# ---------------------------------------------------------------------------
# Domain merge
# ---------------------------------------------------------------------------

def merge_workflow_domain(workflow: dict[str, Any], domain: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(workflow)
    domain_name = domain.get("domain_name", "unknown")
    merged["name"] = f"{workflow.get('name', 'workflow')}_{domain_name}"
    merged["description"] = (
        f"{workflow.get('description', '')}\n"
        f"Domain: {domain.get('description', domain_name)}"
    )

    if merged.get("mode") == "agentic":
        # Agentic mode: resolve domain refs across the entire config
        _deep_resolve_domain(merged, domain)
    else:
        # Sequential mode: resolve domain refs in step params
        for step in merged.get("steps", []):
            step_name = step.get("name", "")
            params = step.get("params", {})
            resolved = {}
            for k, v in params.items():
                if isinstance(v, str):
                    resolved[k] = _resolve_domain_refs(v, domain, step_name)
                else:
                    resolved[k] = v
            step["params"] = resolved
    return merged


def _deep_resolve_domain(obj: Any, domain: dict[str, Any], parent_key: str = "") -> Any:
    """
    Recursively resolve ${domain.*} references throughout a nested structure.
    For string values that are ENTIRELY a single ${domain.X} reference pointing
    to a non-string (dict, list), inject the actual object instead of stringifying.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _deep_resolve_domain(v, domain, k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _deep_resolve_domain(v, domain, parent_key)
    elif isinstance(obj, str):
        # Check if the entire string is a single ${domain.X} reference
        single_ref = re.fullmatch(r'\$\{domain\.([^}]+)\}', obj.strip())
        if single_ref:
            # Resolve to the actual object (could be dict, list, string, etc.)
            resolved = _resolve_domain_ref_to_object(single_ref.group(1), domain)
            if resolved is not None:
                return copy.deepcopy(resolved)
        # Otherwise do normal string interpolation
        return _resolve_domain_refs(obj, domain, parent_key)
    return obj


def _resolve_domain_ref_to_object(ref: str, domain: dict[str, Any]) -> Any:
    """Resolve a domain reference path to its actual value (preserving type)."""
    parts = ref.split(".")
    obj = domain
    for p in parts:
        if isinstance(obj, dict):
            obj = obj.get(p)
            if obj is None:
                return None
        else:
            return None
    return obj


def _resolve_domain_refs(value: str, domain: dict[str, Any], step_name: str) -> str:
    if "${domain." not in value:
        return value

    def _replace(match):
        ref = match.group(1)
        parts = ref.split(".")
        if len(parts) == 2:
            cfg = domain.get(parts[0], {})
            if isinstance(cfg, dict):
                r = cfg.get(parts[1])
                if r is not None:
                    return str(r)
            return f"[domain.{ref} not found]"
        elif len(parts) == 1:
            cfg = domain.get(step_name, {})
            if isinstance(cfg, dict):
                r = cfg.get(parts[0])
                if r is not None:
                    return str(r)
            r = domain.get(parts[0])
            if r is not None:
                return str(r)
            return f"[domain.{ref} not found]"
        else:
            obj = domain
            for p in parts:
                if isinstance(obj, dict):
                    obj = obj.get(p)
                    if obj is None:
                        return f"[domain.{ref} not found]"
                else:
                    return f"[domain.{ref} not found]"
            return str(obj)

    return re.sub(r'\$\{domain\.([^}]+)\}', _replace, value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_use_case(config: dict[str, Any]) -> list[str]:
    errors = []
    if "name" not in config:
        errors.append("Missing 'name'")
    if "steps" not in config:
        errors.append("Missing 'steps'")
        return errors
    if not isinstance(config["steps"], list) or len(config["steps"]) == 0:
        errors.append("'steps' must be a non-empty list")
        return errors

    step_names = set()
    for i, step in enumerate(config["steps"]):
        if "name" not in step:
            errors.append(f"Step {i} missing 'name'")
        elif step["name"] in step_names:
            errors.append(f"Duplicate step name: {step['name']}")
        else:
            step_names.add(step["name"])
        step_errors = validate_use_case_step(step)
        errors.extend(step_errors)

    for step in config["steps"]:
        for t in step.get("transitions", []):
            if isinstance(t, dict):
                target = t.get("goto")
                if target and target != "__end__" and target not in step_names:
                    errors.append(f"'{step['name']}' references unknown target: '{target}'")
                for opt in t.get("agent_decide", {}).get("options", []):
                    s = opt.get("step", "")
                    if s != "__end__" and s not in step_names:
                        errors.append(f"'{step['name']}' agent option references unknown: '{s}'")
    return errors


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _evaluate_condition(condition: str, state: WorkflowState, step_name: str) -> bool:
    condition = condition.strip()
    if condition.startswith("_loop_count"):
        m = re.match(r'_loop_count\s*(>=|<=|>|<|==|!=)\s*(\d+)', condition)
        if m:
            return _compare(get_loop_count(state, step_name), m.group(1), int(m.group(2)))
        return False

    m = re.match(r"output\.([a-zA-Z_][a-zA-Z0-9_.]*)\s*(==|!=|>=|<=|>|<)\s*(.+)", condition)
    if not m:
        return False

    field_path, op, raw_value = m.group(1), m.group(2), m.group(3).strip()
    output = get_step_output(state, step_name)
    if output is None:
        return False

    obj = output
    for part in field_path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
            if obj is None:
                return False
        else:
            return False

    return _compare(obj, op, _parse_value(raw_value))


def _parse_value(raw: str) -> Any:
    raw = raw.strip().strip("'\"")
    if raw.lower() == "true": return True
    if raw.lower() == "false": return False
    try: return int(raw)
    except ValueError: pass
    try: return float(raw)
    except ValueError: pass
    return raw


def _compare(actual, op, expected) -> bool:
    try:
        if op == "==": return actual == expected
        if op == "!=": return actual != expected
        if op == ">": return float(actual) > float(expected)
        if op == "<": return float(actual) < float(expected)
        if op == ">=": return float(actual) >= float(expected)
        if op == "<=": return float(actual) <= float(expected)
    except (TypeError, ValueError):
        return False
    return False


# ---------------------------------------------------------------------------
# Workflow composition
# ---------------------------------------------------------------------------

def compose_workflow(
    config: dict[str, Any],
    model: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    tool_registry: ToolRegistry | None = None,
) -> StateGraph:

    errors = validate_use_case(config)
    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))

    graph = StateGraph(WorkflowState)
    steps = config["steps"]
    step_names = [s["name"] for s in steps]

    for step in steps:
        if step["primitive"] == "retrieve":
            if tool_registry is None:
                raise ValueError(
                    f"Step '{step['name']}' uses retrieve primitive but no "
                    f"tool_registry was provided. Pass a ToolRegistry to "
                    f"compose_workflow() or use create_case_registry()."
                )
            node = create_retrieve_node(
                step_name=step["name"],
                params=step.get("params", {}),
                tool_registry=tool_registry,
                model=step.get("model", model),
                temperature=step.get("temperature", temperature),
            )
        else:
            node = create_node(
                step_name=step["name"],
                primitive_name=step["primitive"],
                params=step.get("params", {}),
                model=step.get("model", model),
                temperature=step.get("temperature", temperature),
            )
        graph.add_node(step["name"], node)

    graph.set_entry_point(step_names[0])

    for i, step in enumerate(steps):
        transitions = step.get("transitions", [])
        default_next = step_names[i + 1] if i + 1 < len(step_names) else "__end__"

        if not transitions:
            graph.add_edge(step["name"], END if default_next == "__end__" else default_next)
        else:
            _add_transitions(graph, step["name"], transitions, default_next,
                             step.get("max_loops"), step.get("loop_fallback", "__end__"), model)

    return graph


def _add_transitions(graph, step_name, transitions, default_next, max_loops, loop_fallback, model):
    det_routes = []
    agent_config = None
    explicit_default = None

    for t in transitions:
        if "when" in t and "goto" in t:
            det_routes.append((t["when"], t["goto"]))
        elif "agent_decide" in t:
            agent_config = t["agent_decide"]
        elif "default" in t:
            explicit_default = t["default"]

    final_default = explicit_default or default_next
    agent_router = None
    if agent_config:
        agent_router = create_agent_router(step_name, agent_config, model)

    trace = get_trace()

    def composite_router(state: WorkflowState) -> str:
        if max_loops is not None:
            count = get_loop_count(state, step_name)
            if count >= max_loops:
                target = loop_fallback
                trace.on_route_decision(step_name, target, "loop_limit",
                                        f"Count {count} >= max {max_loops}")
                return END if target == "__end__" else target

        for condition, target in det_routes:
            if _evaluate_condition(condition, state, step_name):
                trace.on_route_decision(step_name, target, "deterministic", condition)
                return END if target == "__end__" else target

        if agent_router:
            chosen = agent_router(state)
            return END if chosen == "__end__" else chosen

        trace.on_route_decision(step_name, final_default, "default", "no conditions matched")
        return END if final_default == "__end__" else final_default

    composite_router.__name__ = f"router_{step_name}"
    graph.add_conditional_edges(step_name, composite_router)


def compile_workflow(config, model="gemini-2.0-flash", temperature=0.1, tool_registry=None):
    return compose_workflow(config, model, temperature, tool_registry).compile()


def run_workflow(config, workflow_input, model="gemini-2.0-flash", temperature=0.1, tool_registry=None):
    compiled = compile_workflow(config, model, temperature, tool_registry)
    initial: WorkflowState = {
        "input": workflow_input,
        "steps": [],
        "current_step": "",
        "metadata": {"use_case": config.get("name", ""), "description": config.get("description", "")},
        "loop_counts": {},
        "routing_log": [],
    }
    return compiled.invoke(initial)
