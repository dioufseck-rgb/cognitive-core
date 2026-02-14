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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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


class NullTrace:
    """No-op tracer when tracing is disabled."""
    def on_step_start(self, *a, **kw): pass
    def on_llm_start(self, *a, **kw): pass
    def on_llm_end(self, *a, **kw): pass
    def on_parse_result(self, *a, **kw): pass
    def on_parse_error(self, *a, **kw): pass
    def on_route_decision(self, *a, **kw): pass


# Global trace callback — set by the runner before execution
_trace: TraceCallback = NullTrace()


def set_trace(callback: TraceCallback):
    global _trace
    _trace = callback


def get_trace() -> TraceCallback:
    return _trace


# ---------------------------------------------------------------------------
# LLM and JSON extraction
# ---------------------------------------------------------------------------

def create_llm(model: str = "gemini-2.0-flash", temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def extract_json(text: str) -> dict:
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    brace_start = text.find('{')
    if brace_start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    depth = 0
    json_str = None
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                json_str = text[brace_start:i + 1]
                break

    if json_str is None:
        json_str = text[brace_start:]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}\nFirst 500 chars: {json_str[:500]}")


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def create_node(
    step_name: str,
    primitive_name: str,
    params: dict[str, str],
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
