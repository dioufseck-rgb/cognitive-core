"""
Cognitive Core - Node Factory

Creates LangGraph nodes with built-in tracing.
Every node emits trace events before/after LLM calls so the runner
can display real-time progress.
"""

import json
import re
import sys
import time
from typing import Any, Callable, Protocol

from langchain_core.messages import HumanMessage
from engine.llm import create_llm

from registry.primitives import render_prompt, get_schema_class
from engine.state import (
    WorkflowState,
    StepResult,
    RoutingDecision,
    resolve_param,
    build_context_from_state,
)


# ---------------------------------------------------------------------------
# Trace callback protocol
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    brace_start = text.find('{')
    if brace_start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    # Find matching closing brace, respecting string contents
    depth = 0
    in_string = False
    escape_next = False
    json_end = None

    for i in range(brace_start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    json_str = text[brace_start:json_end] if json_end else text[brace_start:]

    # Attempt 1: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix unescaped backslashes
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: replace raw control characters inside string values
    # (common with Gemini when generating multi-line text in JSON)
    def _clean_control_chars(s: str) -> str:
        result = []
        in_str = False
        esc = False
        for ch in s:
            if esc:
                result.append(ch)
                esc = False
                continue
            if ch == '\\' and in_str:
                result.append(ch)
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                result.append(ch)
                continue
            if in_str and ord(ch) < 32:
                # Replace raw control chars with their escape sequences
                if ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                else:
                    result.append(f'\\u{ord(ch):04x}')
            else:
                result.append(ch)
        return ''.join(result)

    cleaned = _clean_control_chars(fixed)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 4: if JSON appears truncated (no closing brace), try to
    # close open structures and parse a partial result
    if json_end is None:
        # Count unclosed braces/brackets
        partial = cleaned
        open_braces = partial.count('{') - partial.count('}')
        open_brackets = partial.count('[') - partial.count(']')
        # If we're inside a string, close it
        quote_count = partial.count('"') - partial.count('\\"')
        if quote_count % 2 == 1:
            partial += '"'
        partial += ']' * max(0, open_brackets)
        partial += '}' * max(0, open_braces)
        try:
            return json.loads(partial)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON: {json_str[:500]}")


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def create_node(
    step_name: str,
    primitive_name: str,
    params: dict[str, str],
    model: str = "default",
    temperature: float = 0.1,
    auto_context: bool = True,
) -> Callable[[WorkflowState], dict]:

    llm = create_llm(model=model, temperature=temperature)
    schema_cls = get_schema_class(primitive_name)

    def node_fn(state: WorkflowState) -> dict:
        # Compute loop iteration
        current_counts = dict(state.get("loop_counts", {}))
        iteration = current_counts.get(step_name, 0) + 1
        current_counts[step_name] = iteration

        _trace.on_step_start(step_name, primitive_name, iteration)

        # Resolve params
        resolved_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved_params[key] = resolve_param(value, state)
            elif isinstance(value, list):
                resolved_params[key] = "\n".join(
                    resolve_param(str(v), state) for v in value
                )
            else:
                resolved_params[key] = str(value)

        if auto_context and "context" not in resolved_params:
            prior_context = build_context_from_state(state)
            input_context = json.dumps(state["input"], indent=2) if state["input"] else ""
            resolved_params["context"] = f"Workflow Input:\n{input_context}\n\n{prior_context}"
        elif auto_context and "${" not in params.get("context", ""):
            prior_context = build_context_from_state(state)
            if prior_context != "No prior steps completed.":
                resolved_params["context"] = resolved_params["context"] + "\n\n" + prior_context

        if "input" not in resolved_params:
            resolved_params["input"] = json.dumps(state["input"], indent=2)

        rendered_prompt = render_prompt(primitive_name, resolved_params)

        # LLM call with tracing
        _trace.on_llm_start(step_name, len(rendered_prompt))
        t0 = time.time()

        messages = [HumanMessage(content=rendered_prompt)]
        response = llm.invoke(messages)
        raw_response = response.content

        elapsed = time.time() - t0
        _trace.on_llm_end(step_name, len(raw_response), elapsed)

        # Parse
        try:
            parsed = extract_json(raw_response)
            validated = schema_cls.model_validate(parsed)
            output = validated.model_dump()
            _trace.on_parse_result(step_name, primitive_name, output)
        except Exception as e:
            output = {
                "error": str(e),
                "raw_response": raw_response[:500],
                "confidence": 0.0,
                "reasoning": f"Failed to parse LLM response: {e}",
                "evidence_used": [],
                "evidence_missing": [],
            }
            _trace.on_parse_error(step_name, str(e))

        step_result: StepResult = {
            "step_name": step_name,
            "primitive": primitive_name,
            "output": output,
            "raw_response": raw_response,
            "prompt_used": rendered_prompt,
        }

        return {
            "steps": [step_result],
            "current_step": step_name,
            "loop_counts": current_counts,
        }

    node_fn.__name__ = f"node_{step_name}"
    return node_fn


# ---------------------------------------------------------------------------
# Agent router with tracing
# ---------------------------------------------------------------------------

def create_agent_router(
    from_step: str,
    agent_config: dict[str, Any],
    model: str = "default",
) -> Callable[[WorkflowState], str]:

    llm = create_llm(model=model, temperature=0.1)
    options = agent_config["options"]
    option_descriptions = "\n".join(f"  - {opt['step']}: {opt['description']}" for opt in options)
    valid_steps = [opt["step"] for opt in options]

    def router_fn(state: WorkflowState) -> str:
        context = build_context_from_state(state)
        agent_prompt = resolve_param(agent_config.get("prompt", ""), state)

        prompt = f"""You are a workflow routing agent. Decide which step should execute next.

WORKFLOW STATE:
{context}

ROUTING CONTEXT:
{agent_prompt}

AVAILABLE NEXT STEPS:
{option_descriptions}

Respond with JSON: {{"chosen_step": "step_name", "reasoning": "why"}}
You MUST choose one of: {valid_steps}
Respond ONLY with JSON."""

        _trace.on_llm_start(f"agent_router({from_step})", len(prompt))
        t0 = time.time()

        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        elapsed = time.time() - t0
        _trace.on_llm_end(f"agent_router({from_step})", len(response.content), elapsed)

        try:
            parsed = extract_json(response.content)
            chosen = parsed.get("chosen_step", "")
            reasoning = parsed.get("reasoning", "")
            if chosen not in valid_steps:
                chosen = valid_steps[0]
                reasoning = f"Invalid choice '{parsed.get('chosen_step')}', defaulting to {chosen}"
        except Exception as e:
            chosen = valid_steps[0]
            reasoning = f"Parse failed: {e}"

        _trace.on_route_decision(from_step, chosen, "agent", reasoning)
        return chosen

    router_fn.__name__ = f"agent_router_{from_step}"
    return router_fn


# ---------------------------------------------------------------------------
# Retrieve node — calls external tools, then LLM assesses quality
# ---------------------------------------------------------------------------

def create_retrieve_node(
    step_name: str,
    params: dict[str, str],
    tool_registry: Any,  # engine.tools.ToolRegistry
    model: str = "default",
    temperature: float = 0.1,
) -> Callable[[WorkflowState], dict]:
    """
    Create a Retrieve primitive node.

    Unlike other primitives which are pure LLM calls, Retrieve has two phases:
    1. TOOL PHASE: Call registered data sources to fetch data
    2. LLM PHASE: Assess data completeness and quality

    For deterministic strategy: parse specification, call all listed sources.
    For agentic strategy: LLM decides which sources to call based on state.
    """
    from engine.tools import ToolRegistry, ToolResult
    from registry.primitives import render_prompt, get_schema_class

    llm = create_llm(model=model, temperature=temperature)
    schema_cls = get_schema_class("retrieve")

    def node_fn(state: WorkflowState) -> dict:
        current_counts = dict(state.get("loop_counts", {}))
        iteration = current_counts.get(step_name, 0) + 1
        current_counts[step_name] = iteration

        _trace.on_step_start(step_name, "retrieve", iteration)

        # Resolve params
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = resolve_param(value, state)
            else:
                resolved[key] = str(value)

        strategy = resolved.get("strategy", "deterministic")
        specification = resolved.get("specification", "")

        # Build query context from case input + workflow state
        query_context = dict(state.get("input", {}))
        # Add any prior step outputs that might inform retrieval
        for step in state.get("steps", []):
            query_context[f"_step_{step['step_name']}"] = step["output"]

        # Phase 1: Call tools
        available_tools = tool_registry.list_tools()

        if strategy == "agentic":
            # In agentic mode, ask LLM which sources to fetch
            sources_to_fetch = _plan_agentic_retrieval(
                llm, specification, tool_registry.describe(),
                build_context_from_state(state), available_tools,
                step_name,
            )
        else:
            # Deterministic: parse specification for source names
            sources_to_fetch = _parse_required_sources(
                specification, available_tools
            )

        # Execute retrievals with tracing
        results: list[ToolResult] = []
        assembled_data = {}

        for source_name in sources_to_fetch:
            _trace.on_retrieve_start(step_name, source_name)
            t0 = time.time()

            result = tool_registry.call(source_name, query_context)
            results.append(result)

            elapsed_ms = (time.time() - t0) * 1000
            _trace.on_retrieve_end(
                step_name, source_name,
                result.status, elapsed_ms,
            )

            if result.status == "success" and result.data:
                assembled_data[source_name] = result.data

        skipped = [t for t in available_tools if t not in sources_to_fetch]

        # Phase 2: LLM assesses completeness
        # Send a SUMMARY of retrieved data to the LLM, not the full payload.
        # The raw data goes to downstream steps via assembled_data; the LLM
        # only needs to know what was fetched, how much, and whether it's sufficient.
        data_summary = {}
        for src_name, src_data in assembled_data.items():
            if isinstance(src_data, dict):
                data_summary[src_name] = {
                    "keys": list(src_data.keys()),
                    "record_count": sum(
                        len(v) if isinstance(v, (list, dict)) else 1
                        for v in src_data.values()
                    ),
                    "sample": {
                        k: (f"[{len(v)} items]" if isinstance(v, list) and len(v) > 3
                            else f"{{{len(v)} keys}}" if isinstance(v, dict) and len(str(v)) > 200
                            else v)
                        for k, v in list(src_data.items())[:5]
                    },
                }
            elif isinstance(src_data, list):
                data_summary[src_name] = {
                    "type": "array",
                    "count": len(src_data),
                    "sample": src_data[:2] if len(src_data) > 2 else src_data,
                }
            elif isinstance(src_data, str) and len(src_data) > 500:
                data_summary[src_name] = {
                    "type": "text",
                    "length": len(src_data),
                    "preview": src_data[:200] + "...",
                }
            else:
                data_summary[src_name] = src_data

        source_results_text = json.dumps({
            "assembled_summary": data_summary,
            "results": [
                {
                    "source": r.source,
                    "status": r.status,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }, indent=2)

        resolved["sources"] = tool_registry.describe()
        resolved["source_results"] = source_results_text
        if "context" not in resolved:
            input_ctx = json.dumps(state["input"], indent=2) if state["input"] else ""
            prior_ctx = build_context_from_state(state)
            resolved["context"] = f"Workflow Input:\n{input_ctx}\n\n{prior_ctx}"

        rendered_prompt = render_prompt("retrieve", resolved)

        _trace.on_llm_start(step_name, len(rendered_prompt))
        t0 = time.time()
        response = llm.invoke([HumanMessage(content=rendered_prompt)])
        raw_response = response.content
        elapsed = time.time() - t0
        _trace.on_llm_end(step_name, len(raw_response), elapsed)

        # Parse LLM assessment
        try:
            parsed = extract_json(raw_response)
            # Inject the actual assembled data (LLM may summarize it,
            # but we want the raw data available to downstream steps)
            parsed["data"] = assembled_data
            # Build sources_queried from actual results
            parsed["sources_queried"] = [
                {
                    "source": r.source,
                    "status": r.status,
                    "data": None,  # don't duplicate in the schema output
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                    "freshness": r.freshness,
                }
                for r in results
            ]
            parsed["sources_skipped"] = skipped
            validated = schema_cls.model_validate(parsed)
            output = validated.model_dump()
            _trace.on_parse_result(step_name, "retrieve", output)
        except Exception as e:
            # Even if LLM assessment fails, the data is still valid
            # and downstream steps can use it. Mark confidence based on
            # whether we actually got data, not whether the LLM parsed.
            has_data = bool(assembled_data)
            output = {
                "data": assembled_data,
                "sources_queried": [
                    {
                        "source": r.source,
                        "status": r.status,
                        "data": None,
                        "error": r.error,
                        "latency_ms": r.latency_ms,
                        "freshness": r.freshness,
                    }
                    for r in results
                ],
                "sources_skipped": skipped,
                "retrieval_plan": "",
                "confidence": 0.8 if has_data else 0.0,
                "reasoning": f"Data retrieved successfully ({len(assembled_data)} sources) but LLM assessment parse failed: {e}" if has_data else f"No data retrieved and LLM assessment failed: {e}",
                "evidence_used": [],
                "evidence_missing": [],
            }
            if has_data:
                _trace.on_parse_result(step_name, "retrieve", output)
            else:
                _trace.on_parse_error(step_name, str(e))

        step_result: StepResult = {
            "step_name": step_name,
            "primitive": "retrieve",
            "output": output,
            "raw_response": raw_response,
            "prompt_used": rendered_prompt,
        }

        return {
            "steps": [step_result],
            "current_step": step_name,
            "loop_counts": current_counts,
        }

    node_fn.__name__ = f"node_{step_name}"
    return node_fn


# ---------------------------------------------------------------------------
# Act node — calls action registry, then LLM documents what happened
# ---------------------------------------------------------------------------

def create_act_node(
    step_name: str,
    params: dict[str, str],
    action_registry: Any,  # engine.actions.ActionRegistry
    model: str = "default",
    temperature: float = 0.1,
    dry_run: bool = True,
) -> Callable[[WorkflowState], dict]:
    """
    Create an Act primitive node.

    Like Retrieve, Act has two phases:
    1. ACTION PHASE: Execute (or simulate) actions via the action registry
    2. LLM PHASE: Document what was done, check authorization, assess side effects

    The LLM decides which actions to take based on workflow state and
    available actions. The action registry enforces authorization and
    handles execution/simulation.
    """
    from engine.actions import ActionRegistry, ActionResult
    from registry.primitives import render_prompt, get_schema_class

    llm = create_llm(model=model, temperature=temperature)
    schema_cls = get_schema_class("act")

    def node_fn(state: WorkflowState) -> dict:
        current_counts = dict(state.get("loop_counts", {}))
        iteration = current_counts.get(step_name, 0) + 1
        current_counts[step_name] = iteration

        _trace.on_step_start(step_name, "act", iteration)

        # Resolve params
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = resolve_param(value, state)
            else:
                resolved[key] = str(value)

        # Inject available actions from registry
        if "actions" not in resolved or resolved["actions"] == "":
            resolved["actions"] = action_registry.describe()

        # Build context
        if "context" not in resolved:
            input_ctx = json.dumps(state["input"], indent=2) if state["input"] else ""
            prior_ctx = build_context_from_state(state)
            resolved["context"] = f"Workflow Input:\n{input_ctx}\n\n{prior_ctx}"

        if "input" not in resolved:
            resolved["input"] = json.dumps(state["input"], indent=2)

        # Set execution mode
        mode = resolved.get("mode", "dry_run" if dry_run else "execution")
        resolved["mode"] = mode

        # LLM call: decide what actions to take and document them
        rendered_prompt = render_prompt("act", resolved)
        _trace.on_llm_start(step_name, len(rendered_prompt))
        t0 = time.time()
        response = llm.invoke([HumanMessage(content=rendered_prompt)])
        raw_response = response.content
        elapsed = time.time() - t0
        _trace.on_llm_end(step_name, len(raw_response), elapsed)

        # Parse LLM response
        try:
            parsed = extract_json(raw_response)

            # Phase 2: Actually execute/simulate actions through registry
            actions_taken = parsed.get("actions_taken", [])
            executed_actions = []

            for action_spec in actions_taken:
                action_name = action_spec.get("action", "")
                action_params = action_spec.get("response_data", {}) or {}

                # Build exec_params: start with what the LLM provided
                exec_params = {**action_params}

                # Deterministically wire params from workflow state.
                # The LLM decides WHICH action; the framework wires HOW.
                # Framework-sourced params OVERRIDE LLM-provided params
                # to ensure correctness.
                case_input = state.get("input", {})

                if action_name == "send_email":
                    # Recipient: ALWAYS from case member_profile.email
                    member = case_input.get("member_profile", {})
                    recipient = member.get("email", "")
                    if recipient:
                        exec_params["recipient"] = recipient

                    # Body: ALWAYS from the most recent generate step's artifact
                    for step in reversed(state.get("steps", [])):
                        if step["primitive"] == "generate" and step["output"].get("artifact"):
                            exec_params["body"] = step["output"]["artifact"]
                            break

                    # Subject: extract from artifact first line if "Subject:" prefix,
                    # otherwise build from case data
                    body = exec_params.get("body", "")
                    if body.startswith("Subject:"):
                        first_line, _, rest = body.partition("\n")
                        exec_params["subject"] = first_line.replace("Subject:", "").strip()
                        exec_params["body"] = rest.strip()
                    elif not exec_params.get("subject"):
                        complaint = case_input.get("complaint", {})
                        exec_params["subject"] = (
                            f"Re: {complaint.get('subject', 'Your recent inquiry')}"
                            f" — {complaint.get('complaint_id', '')}"
                        )
                else:
                    # For non-email actions, pull flat member data
                    for k, v in case_input.items():
                        if isinstance(v, (str, int, float, bool)):
                            exec_params.setdefault(k, v)
                        elif isinstance(v, dict):
                            for sk, sv in v.items():
                                if isinstance(sv, (str, int, float, bool)):
                                    exec_params.setdefault(sk, sv)

                # Check authorization
                auth_level = resolved.get("authorization_level", "agent")
                amount = exec_params.get("amount")
                authorized, auth_reason = action_registry.check_authorization(
                    action_name, auth_level, amount
                )

                if not authorized:
                    action_spec["status"] = "blocked"
                    action_spec["error"] = auth_reason
                elif mode == "dry_run":
                    result = action_registry.execute(action_name, exec_params, dry_run=True)
                    action_spec["status"] = "simulated"
                    action_spec["confirmation_id"] = result.confirmation_id
                    action_spec["response_data"] = result.response_data
                else:
                    result = action_registry.execute(action_name, exec_params, dry_run=False)
                    action_spec["status"] = result.status
                    action_spec["confirmation_id"] = result.confirmation_id
                    action_spec["response_data"] = result.response_data
                    action_spec["error"] = result.error
                    action_spec["latency_ms"] = result.latency_ms
                    action_spec["rollback_handle"] = result.rollback_handle

                executed_actions.append(action_spec)

            parsed["actions_taken"] = executed_actions
            parsed["mode"] = mode

            validated = schema_cls.model_validate(parsed)
            output = validated.model_dump()
            _trace.on_parse_result(step_name, "act", output)

        except Exception as e:
            output = {
                "mode": mode,
                "error": str(e),
                "raw_response": raw_response[:500] if raw_response else "",
                "confidence": 0.0,
                "reasoning": f"Failed to parse or execute actions: {e}",
                "actions_taken": [],
                "authorization_checks": [],
                "side_effects": [],
                "requires_human_approval": True,
                "approval_brief": f"Act failed — manual review needed: {e}",
                "evidence_used": [],
                "evidence_missing": [],
            }
            _trace.on_parse_error(step_name, str(e))

        step_result: StepResult = {
            "step_name": step_name,
            "primitive": "act",
            "output": output,
            "raw_response": raw_response,
            "prompt_used": rendered_prompt,
        }

        return {
            "steps": [step_result],
            "current_step": step_name,
            "loop_counts": current_counts,
        }

    node_fn.__name__ = f"node_{step_name}"
    return node_fn


def _parse_required_sources(
    specification: str,
    available_tools: list[str],
) -> list[str]:
    """
    Parse a specification string to find source names that match
    registered tools. Matches lines like:
        - member_profile: description
        - transaction_detail: description
    """
    import re
    found = []
    for tool_name in available_tools:
        # Match tool name appearing as a key in the specification
        if re.search(rf'\b{re.escape(tool_name)}\b', specification):
            found.append(tool_name)
    # If nothing matched, fall back to all available tools
    if not found:
        return available_tools
    return found


def _plan_agentic_retrieval(
    llm,
    specification: str,
    tools_description: str,
    workflow_state: str,
    available_tools: list[str],
    step_name: str,
) -> list[str]:
    """
    Ask the LLM to decide which data sources to fetch.
    Returns a list of tool names to call.
    """
    prompt = f"""You are a data retrieval planner. Decide which data sources to query.

WORKFLOW STATE:
{workflow_state}

RETRIEVAL SPECIFICATION:
{specification}

AVAILABLE DATA SOURCES:
{tools_description}

Based on the current workflow state and what data is needed, decide which
sources to retrieve. Consider:
- What does the specification require?
- What does the current workflow state already provide?
- Which sources are most valuable given what we know so far?

Respond with JSON: {{"sources": ["source_name1", "source_name2"], "plan": "explanation"}}
You may only choose from: {available_tools}
Respond ONLY with JSON."""

    _trace.on_llm_start(f"retrieval_planner({step_name})", len(prompt))
    t0 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    elapsed = time.time() - t0
    _trace.on_llm_end(f"retrieval_planner({step_name})", len(response.content), elapsed)

    try:
        parsed = extract_json(response.content)
        sources = parsed.get("sources", [])
        valid = [s for s in sources if s in available_tools]
        return valid if valid else available_tools
    except Exception:
        return available_tools
