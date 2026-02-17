"""
Cognitive Core - Shared State
"""

from typing import Any, TypedDict, Annotated
import operator
import json


class StepResult(TypedDict):
    step_name: str
    primitive: str
    output: dict[str, Any]
    raw_response: str
    prompt_used: str


class RoutingDecision(TypedDict):
    from_step: str
    to_step: str
    decision_type: str
    reason: str
    agent_reasoning: str


class WorkflowState(TypedDict):
    input: dict[str, Any]
    steps: Annotated[list[StepResult], operator.add]
    current_step: str
    metadata: dict[str, Any]
    loop_counts: dict[str, int]
    routing_log: Annotated[list[RoutingDecision], operator.add]


def get_step_output(state: WorkflowState, step_name: str) -> dict[str, Any] | None:
    result = None
    for step in state["steps"]:
        if step["step_name"] == step_name:
            result = step["output"]
    return result


def get_all_step_outputs(state: WorkflowState, step_name: str) -> list[dict[str, Any]]:
    return [s["output"] for s in state["steps"] if s["step_name"] == step_name]


def get_latest_output(state: WorkflowState) -> dict[str, Any] | None:
    if state["steps"]:
        return state["steps"][-1]["output"]
    return None


def get_latest_step(state: WorkflowState) -> StepResult | None:
    if state["steps"]:
        return state["steps"][-1]
    return None


def get_loop_count(state: WorkflowState, step_name: str) -> int:
    return state.get("loop_counts", {}).get(step_name, 0)


def resolve_param(value: str, state: WorkflowState) -> str:
    if "${" not in str(value):
        return value

    import re

    def _replace_ref(match):
        ref_path = match.group(1)
        parts = ref_path.split(".")

        if parts[0] == "_loop_count":
            if len(parts) > 1:
                return str(get_loop_count(state, parts[1]))
            return str(get_loop_count(state, state.get("current_step", "")))

        if parts[0] == "_delegations":
            # ${_delegations.wo_id.field} → delegation result reference
            delegations = state.get("delegation_results", {})
            if len(parts) > 1:
                obj = delegations.get(parts[1], delegations)
                for part in parts[2:]:
                    if isinstance(obj, dict):
                        obj = obj.get(part, f"[field '{part}' not found]")
                    else:
                        return f"[cannot navigate into {type(obj).__name__}]"
                if isinstance(obj, (dict, list)):
                    return json.dumps(obj, indent=2)
                return str(obj)
            if isinstance(delegations, (dict, list)):
                return json.dumps(delegations, indent=2)
            return str(delegations)

        if parts[0] == "input":
            obj = state["input"]
            parts = parts[1:]
        elif parts[0] == "previous":
            obj = get_latest_output(state)
            if obj is None:
                return "[no previous step output]"
            parts = parts[1:]
        elif parts[0].startswith("_last_"):
            # ${_last_generate.artifact} → most recent generate step's output
            primitive_name = parts[0][len("_last_"):]
            obj = _get_last_primitive_output(state, primitive_name)
            if obj is None:
                return f"[no completed '{primitive_name}' step found]"
            parts = parts[1:]
        else:
            step_name = parts[0]
            obj = get_step_output(state, step_name)
            if obj is None:
                return f"[step '{step_name}' not found]"
            parts = parts[1:]

        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part, f"[field '{part}' not found]")
            elif isinstance(obj, list):
                try:
                    obj = obj[int(part)]
                except (ValueError, IndexError):
                    return f"[invalid index '{part}']"
            else:
                return f"[cannot navigate into {type(obj).__name__}]"

        if isinstance(obj, (dict, list)):
            return json.dumps(obj, indent=2)
        return str(obj)

    return re.sub(r'\$\{([^}]+)\}', _replace_ref, str(value))


def _get_last_primitive_output(state: WorkflowState, primitive_name: str) -> dict[str, Any] | None:
    """Find the most recent step that used the given primitive, return its output."""
    result = None
    for step in state.get("steps", []):
        if step["primitive"] == primitive_name:
            result = step["output"]
    return result


def build_context_from_state(state: WorkflowState) -> str:
    if not state["steps"]:
        return "No prior steps completed."

    parts = ["Previous steps in this workflow:"]
    step_counts: dict[str, int] = {}

    for step in state["steps"]:
        output = step["output"]
        name = step["step_name"]
        step_counts[name] = step_counts.get(name, 0) + 1
        iteration = step_counts[name]
        iter_label = f" (iteration {iteration})" if iteration > 1 else ""

        parts.append(f"\n--- {name} ({step['primitive']}){iter_label} ---")

        primitive = step["primitive"]
        if primitive == "classify":
            parts.append(f"Category: {output.get('category', 'N/A')}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
            parts.append(f"Reasoning: {output.get('reasoning', 'N/A')}")
        elif primitive == "investigate":
            parts.append(f"Finding: {output.get('finding', 'N/A')}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
            parts.append(f"Reasoning: {output.get('reasoning', 'N/A')}")
            for h in output.get("hypotheses_tested", []):
                parts.append(f"  Hypothesis: {h.get('hypothesis', '')} -> {h.get('status', '')}")
        elif primitive == "verify":
            parts.append(f"Conforms: {output.get('conforms', 'N/A')}")
            parts.append(f"Reasoning: {output.get('reasoning', 'N/A')}")
            for v in output.get("violations", []):
                parts.append(f"  Violation [{v.get('severity', '')}]: {v.get('description', '')}")
        elif primitive == "generate":
            parts.append(f"Artifact preview: {str(output.get('artifact', ''))[:200]}...")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
        elif primitive == "challenge":
            parts.append(f"Survives: {output.get('survives', 'N/A')}")
            parts.append(f"Assessment: {output.get('overall_assessment', 'N/A')}")
            for v in output.get("vulnerabilities", []):
                parts.append(f"  Vulnerability [{v.get('severity', '')}]: {v.get('description', '')}")
        elif primitive == "retrieve":
            data = output.get("data", {})
            parts.append(f"Sources retrieved: {list(data.keys())}")
            for key, val in data.items():
                parts.append(f"  {key}: {json.dumps(val, indent=2)[:200]}...")
        elif primitive == "think":
            parts.append(f"Thought: {str(output.get('thought', ''))[:300]}...")
            conclusions = output.get("conclusions", [])
            if conclusions:
                parts.append(f"Conclusions: {conclusions}")
            decision = output.get("decision")
            if decision:
                parts.append(f"Decision: {decision}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")

    routing_log = state.get("routing_log", [])
    if routing_log:
        parts.append("\n--- Routing decisions ---")
        for rd in routing_log:
            parts.append(f"  {rd['from_step']} -> {rd['to_step']} ({rd['decision_type']}): {rd['reason']}")

    return "\n".join(parts)
