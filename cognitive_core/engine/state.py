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

        # Context is truncated aggressively — downstream steps need structured
        # outputs (category, finding, recommendation) not full prose reasoning.
        # Reasoning strings are capped at 150 chars; retrieve data at 120 chars.
        # This keeps context token count bounded regardless of LLM verbosity.
        primitive = step["primitive"]
        if primitive == "classify":
            parts.append(f"Category: {output.get('category', 'N/A')}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
            reasoning = str(output.get('reasoning', ''))
            if reasoning:
                parts.append(f"Reasoning: {reasoning[:150]}")
            alts = output.get('alternative_categories', [])
            if alts:
                parts.append(f"Alternatives considered: {', '.join(a.get('category','') for a in alts[:3])}")
        elif primitive == "investigate":
            parts.append(f"Finding: {str(output.get('finding', 'N/A'))[:300]}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
            for h in output.get("hypotheses_tested", [])[:3]:
                parts.append(f"  Hypothesis: {h.get('hypothesis', '')[:80]} -> {h.get('status', '')}")
            flags = output.get("evidence_flags", [])
            if flags:
                parts.append(f"  Evidence flags: {flags[:3]}")
        elif primitive == "verify":
            parts.append(f"Conforms: {output.get('conforms', 'N/A')}")
            for v in output.get("violations", []):
                parts.append(f"  Violation [{v.get('severity', '')}]: {str(v.get('description', ''))[:100]}")
            if output.get('conforms') and not output.get('violations'):
                parts.append("  All rules passed.")
        elif primitive == "generate":
            parts.append(f"Artifact preview: {str(output.get('artifact', ''))[:150]}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
        elif primitive == "challenge":
            parts.append(f"Survives: {output.get('survives', 'N/A')}")
            assessment = str(output.get('overall_assessment', ''))
            if assessment:
                parts.append(f"Assessment: {assessment[:150]}")
            for v in output.get("vulnerabilities", [])[:3]:
                parts.append(f"  Vulnerability [{v.get('severity', '')}]: {str(v.get('description', ''))[:80]}")
        elif primitive == "retrieve":
            data = output.get("data", {})
            parts.append(f"Sources retrieved: {list(data.keys())}")
            for key, val in list(data.items())[:5]:
                parts.append(f"  {key}: {json.dumps(val)[:120]}")
        elif primitive == "deliberate":
            situation = str(output.get('situation_summary', ''))
            if situation:
                parts.append(f"Situation: {situation[:200]}")
            action = output.get("recommended_action")
            if action:
                parts.append(f"Recommended action: {action}")
            warrant = str(output.get('warrant', ''))
            if warrant:
                parts.append(f"Warrant: {warrant[:150]}")
            parts.append(f"Confidence: {output.get('confidence', 'N/A')}")
        elif primitive == "govern":
            tier = output.get("tier_applied", "?")
            disposition = output.get("disposition", "?")
            parts.append(f"Governance tier: {tier}")
            parts.append(f"Disposition: {disposition}")
            rationale = str(output.get("tier_rationale", ""))
            if rationale:
                parts.append(f"Rationale: {rationale[:150]}")

    routing_log = state.get("routing_log", [])
    if routing_log:
        parts.append("\n--- Routing decisions ---")
        for rd in routing_log:
            parts.append(f"  {rd['from_step']} -> {rd['to_step']} ({rd['decision_type']}): {rd['reason']}")

    # Epistemic summary appended when available so deliberate and govern
    # primitives can see coherence flags and open gaps without re-deriving them.
    ep_record = state.get("epistemic", {}).get("_record")
    if ep_record:
        flags = ep_record.get("coherence_flags", [])
        gaps = ep_record.get("open_evidence_gaps", [])
        warranted = ep_record.get("warranted", True)
        overall = ep_record.get("workflow_overall")
        low = ep_record.get("low_confidence_steps", [])
        if flags or gaps or not warranted or low:
            parts.append("\n--- Epistemic state ---")
            if overall is not None:
                parts.append(f"Workflow epistemic overall: {overall:.2f}")
            if not warranted:
                parts.append("WARNING: one or more steps are not warranted")
            for f in flags:
                parts.append(f"  Coherence flag: {f}")
            for g in gaps:
                parts.append(f"  Open evidence gap: {g[:120]}")
            for s in low[:3]:
                step_flags = s.get("flags", [])
                ec = s.get("evidence_completeness")
                rc = s.get("rule_coverage")
                line = (f"  Low epistemic step: {s['step']} ({s['primitive']}) "
                        f"overall={s['overall']:.2f}")
                if ec is not None:
                    line += f" evidence_completeness={ec:.2f}"
                if rc is not None:
                    line += f" rule_coverage={rc:.2f}"
                if step_flags:
                    line += f" flags={step_flags}"
                parts.append(line)

    return "\n".join(parts)