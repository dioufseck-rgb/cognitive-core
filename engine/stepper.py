"""
Cognitive Core — Step-by-Step Executor

Replaces the single .invoke() call with a step-by-step execution
loop that yields control to the coordinator between steps.

A workflow moves forward through its steps. At any step, the agent
might realize it can't proceed — it needs a decision, analysis,
data, authorization, or any other resource it doesn't have. When
that happens, it produces a ResourceRequest in its output, and the
stepper pauses execution.

The coordinator then dispatches whatever is needed (another workflow,
a human task, an external service). When the resource arrives, the
workflow resumes forward from where it paused, now with the resource
available.

This is not two modes. It's one workflow moving forward, with the
ability to pause when it needs help and resume when it gets it.

─────────────────────────────────────────────────────────────────

IMPLEMENTATION:

LangGraph's compiled graph supports .stream() which yields
(node_name, state_delta) after each node. The stepper uses this
to inspect output between steps via a callback.

The callback can:
  - Return None → continue forward
  - Return a StepInterrupt → pause, return partial state

No LangGraph dependency at import time.
"""

from __future__ import annotations

import copy
from typing import Any, Callable
from dataclasses import dataclass, field


# ─── Interrupt Signal ─────────────────────────────────────────────

@dataclass
class StepInterrupt:
    """
    Signal from the coordinator to pause workflow execution.

    Returned by the step callback when a step produces blocking
    ResourceRequests (backward chaining) or when any other
    mid-execution condition requires suspension.
    """
    reason: str
    suspended_at_step: str
    # The partial state at the point of interruption
    state_at_interrupt: dict[str, Any] = field(default_factory=dict)
    # Resource requests that triggered the interrupt
    resource_requests: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of step-by-step execution."""
    completed: bool              # True if workflow ran to completion
    final_state: dict[str, Any]  # Final or partial state
    interrupt: StepInterrupt | None = None  # Set if interrupted


# Type for the step callback
# Called after each step with (step_name, step_output, full_state)
# Return None to continue, StepInterrupt to pause
StepCallback = Callable[
    [str, dict[str, Any], dict[str, Any]],
    StepInterrupt | None
]


# ─── No-Op Callback ──────────────────────────────────────────────

def no_interrupt_callback(
    step_name: str,
    step_output: dict[str, Any],
    state: dict[str, Any],
) -> StepInterrupt | None:
    """Default callback: never interrupt. Forward-only mode."""
    return None


# ─── Resource Request Callback ────────────────────────────────────

def resource_request_callback(
    step_name: str,
    step_output: dict[str, Any],
    state: dict[str, Any],
) -> StepInterrupt | None:
    """
    Callback that inspects step output for blocking ResourceRequests.

    If any step output contains resource_requests with blocking=True,
    returns a StepInterrupt to pause execution. The coordinator will
    then dispatch providers and resume the workflow when they complete.
    """
    requests = step_output.get("resource_requests", [])
    blocking = [r for r in requests if r.get("blocking", True)]

    if not blocking:
        return None

    # Filter out needs that are already fulfilled in delegation results.
    # When the LLM re-requests something it already has, the coordinator
    # would dispatch again creating an infinite loop.
    existing = set(state.get("input", {}).get("delegation", {}).keys())
    if existing:
        filtered = [r for r in blocking if r.get("need", "") not in existing]
        if len(filtered) < len(blocking):
            skipped = [r.get("need") for r in blocking if r.get("need", "") in existing]
            import sys
            print(f"  [stepper] ⚠ Filtered {len(skipped)} already-fulfilled need(s): {skipped}",
                  file=sys.stderr)
        blocking = filtered

    if not blocking:
        return None

    return StepInterrupt(
        reason=f"Step '{step_name}' produced {len(blocking)} blocking resource request(s)",
        suspended_at_step=step_name,
        state_at_interrupt=state,
        resource_requests=blocking,
    )


# ─── Combined Callback ───────────────────────────────────────────

def combined_callback(
    *callbacks: StepCallback,
) -> StepCallback:
    """
    Combine multiple step callbacks. Returns the first interrupt
    produced by any callback, or None if all return None.
    """
    def _combined(
        step_name: str,
        step_output: dict[str, Any],
        state: dict[str, Any],
    ) -> StepInterrupt | None:
        for cb in callbacks:
            result = cb(step_name, step_output, state)
            if result is not None:
                return result
        return None
    return _combined


# ─── Step-by-Step Executor ────────────────────────────────────────

def step_execute(
    config: dict[str, Any],
    workflow_input: dict[str, Any],
    model: str = "default",
    temperature: float = 0.1,
    tool_registry: Any = None,
    action_registry: Any = None,
    step_callback: StepCallback = no_interrupt_callback,
) -> StepResult:
    """
    Execute a workflow step-by-step with interception between steps.

    Uses LangGraph's .stream() API to get control after each node
    executes. After each step, calls step_callback with the step
    name, step output, and current state. If the callback returns
    a StepInterrupt, execution pauses and returns partial state.

    Args:
        config: Merged workflow+domain config
        workflow_input: Case input data
        model: LLM model identifier
        temperature: LLM temperature
        tool_registry: Tool registry for retrieve steps
        action_registry: Action registry for act steps
        step_callback: Called after each step. Return None to continue,
                       StepInterrupt to pause.

    Returns:
        StepResult with completed=True if workflow finished,
        or completed=False with interrupt details if paused.
    """
    from engine.composer import compile_workflow
    from engine.state import WorkflowState

    compiled = compile_workflow(
        config, model, temperature,
        tool_registry, action_registry,
    )

    initial: WorkflowState = {
        "input": workflow_input,
        "steps": [],
        "current_step": "",
        "metadata": {
            "use_case": config.get("name", ""),
            "description": config.get("description", ""),
        },
        "loop_counts": {},
        "routing_log": [],
    }

    # Use stream to get control between steps
    current_state = dict(initial)

    for event in compiled.stream(initial):
        # event is {node_name: state_delta}
        for node_name, delta in event.items():
            # Merge delta into current state
            # LangGraph reducers handle the merge, but we need
            # to track the accumulated state ourselves for the callback
            current_state = _merge_state(current_state, delta)

            # Extract the latest step output
            latest_output = _get_latest_step_output(current_state, node_name)

            # Call the callback
            interrupt = step_callback(node_name, latest_output, current_state)

            if interrupt is not None:
                # Ensure interrupt has the current state
                interrupt.state_at_interrupt = copy.deepcopy(current_state)
                return StepResult(
                    completed=False,
                    final_state=current_state,
                    interrupt=interrupt,
                )

    # Workflow completed without interruption
    return StepResult(
        completed=True,
        final_state=current_state,
        interrupt=None,
    )


def step_resume(
    config: dict[str, Any],
    state_snapshot: dict[str, Any],
    resume_step: str,
    model: str = "default",
    temperature: float = 0.1,
    tool_registry: Any = None,
    action_registry: Any = None,
    step_callback: StepCallback = no_interrupt_callback,
) -> StepResult:
    """
    Resume a workflow from a saved state with step-by-step interception.

    Combines mid-graph resume (subgraph from resume_step forward)
    with step-by-step interception. The resumed workflow can itself
    be interrupted again by the callback — enabling recursive
    demand-driven chains.

    Args:
        config: Merged workflow+domain config
        state_snapshot: Saved state from suspension
        resume_step: Step name to resume at
        model, temperature, tool_registry, action_registry: Execution params
        step_callback: Called after each step, same as step_execute

    Returns:
        StepResult with completed=True if finished, False if interrupted again
    """
    from engine.composer import compile_subgraph
    from engine.resume import prepare_resume_state

    compiled = compile_subgraph(
        config, resume_step, model, temperature,
        tool_registry, action_registry,
    )

    initial = prepare_resume_state(config, state_snapshot, resume_step)
    current_state = dict(initial)

    for event in compiled.stream(initial):
        for node_name, delta in event.items():
            current_state = _merge_state(current_state, delta)
            latest_output = _get_latest_step_output(current_state, node_name)

            interrupt = step_callback(node_name, latest_output, current_state)

            if interrupt is not None:
                interrupt.state_at_interrupt = copy.deepcopy(current_state)
                return StepResult(
                    completed=False,
                    final_state=current_state,
                    interrupt=interrupt,
                )

    return StepResult(
        completed=True,
        final_state=current_state,
        interrupt=None,
    )


# ─── Internal Helpers ─────────────────────────────────────────────

def _merge_state(
    current: dict[str, Any],
    delta: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge a state delta into the current state.

    Handles the LangGraph reducer semantics:
    - steps: append (operator.add)
    - routing_log: append (operator.add)
    - everything else: overwrite
    """
    merged = dict(current)

    for key, value in delta.items():
        if key in ("steps", "routing_log") and isinstance(value, list):
            # Append reducer
            existing = merged.get(key, [])
            merged[key] = existing + value
        elif key == "loop_counts" and isinstance(value, dict):
            # Merge dict
            existing = merged.get(key, {})
            merged[key] = {**existing, **value}
        else:
            merged[key] = value

    return merged


def _get_latest_step_output(
    state: dict[str, Any],
    node_name: str,
) -> dict[str, Any]:
    """
    Get the output from the most recent step matching node_name.
    Returns empty dict if not found.
    """
    steps = state.get("steps", [])
    for step in reversed(steps):
        if step.get("step_name") == node_name:
            return step.get("output", {})
    return {}
