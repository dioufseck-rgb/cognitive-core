"""
Cognitive Core - Mid-Graph Resume Logic

Pure functions for building subgraphs and preparing state for
mid-graph resume. These functions have ZERO external dependencies
(no LangGraph, no LLM) and can be tested in isolation.

The graph-building functions (compose_subgraph, compile_subgraph,
run_workflow_from_step) are in engine/composer.py since they need
LangGraph's StateGraph. This module provides the logic they delegate to.
"""

import copy
from typing import Any


def collect_reachable_steps(
    steps: list[dict[str, Any]],
    step_names: list[str],
    resume_idx: int,
) -> set[str]:
    """
    Collect all step names reachable from resume_idx.

    Starts with all steps at or after resume_idx (forward slice),
    then does a BFS to add any steps reachable via transitions
    (including backward jumps like verify → generate loops).

    Args:
        steps: Full list of step configs from the workflow
        step_names: Ordered list of step names
        resume_idx: Index of the step to resume at

    Returns:
        Set of reachable step names
    """
    # Start with forward slice
    reachable = set(step_names[resume_idx:])

    # BFS: expand via transition targets
    queue = list(reachable)
    visited = set(reachable)

    while queue:
        current = queue.pop(0)
        step = next((s for s in steps if s["name"] == current), None)
        if not step:
            continue

        for t in step.get("transitions", []):
            target = None
            if isinstance(t, dict):
                target = t.get("goto") or t.get("default")
                # Agent-decide options
                for opt in t.get("agent_decide", {}).get("options", []):
                    opt_target = opt.get("step")
                    if opt_target and opt_target != "__end__" and opt_target not in visited:
                        visited.add(opt_target)
                        reachable.add(opt_target)
                        queue.append(opt_target)

            if target and target != "__end__" and target not in visited:
                visited.add(target)
                reachable.add(target)
                queue.append(target)

    return reachable


def clamp_transitions(
    transitions: list[dict[str, Any]],
    reachable: set[str],
) -> list[dict[str, Any]]:
    """
    Rewrite transitions so any target not in `reachable` becomes __end__.

    This is needed when building subgraphs: transitions that would jump
    to a step not included in the subgraph must terminate instead.

    Returns a new list (does not mutate the original).
    """
    clamped = []
    for t in transitions:
        t_copy = dict(t)
        if "goto" in t_copy and t_copy["goto"] != "__end__":
            if t_copy["goto"] not in reachable:
                t_copy["goto"] = "__end__"
        if "default" in t_copy and t_copy["default"] != "__end__":
            if t_copy["default"] not in reachable:
                t_copy["default"] = "__end__"
        if "agent_decide" in t_copy:
            ad = dict(t_copy["agent_decide"])
            opts = []
            for opt in ad.get("options", []):
                opt_copy = dict(opt)
                if opt_copy.get("step", "") not in reachable and opt_copy.get("step") != "__end__":
                    opt_copy["step"] = "__end__"
                opts.append(opt_copy)
            ad["options"] = opts
            t_copy["agent_decide"] = ad
        clamped.append(t_copy)
    return clamped


def prepare_resume_state(
    config: dict[str, Any],
    state_snapshot: dict[str, Any],
    resume_step: str,
) -> dict[str, Any]:
    """
    Prepare the initial state for a mid-graph resume invocation.

    Takes the saved state snapshot and:
    1. Keeps only step outputs from BEFORE the resume step
    2. Injects delegation results into input.delegation
    3. Resets loop counts for the resume step
    4. Preserves metadata, routing log, and other state

    Args:
        config: Merged workflow+domain config
        state_snapshot: Saved state from suspension
        resume_step: Step name to resume at

    Returns:
        WorkflowState dict ready to pass to compiled.invoke()
    """
    steps_config = config.get("steps", [])
    step_names = [s["name"] for s in steps_config]

    if resume_step not in step_names:
        raise ValueError(
            f"Resume step '{resume_step}' not found in workflow. "
            f"Available steps: {step_names}"
        )

    resume_idx = step_names.index(resume_step)
    steps_before_resume = set(step_names[:resume_idx])

    prior_steps = [
        s for s in state_snapshot.get("steps", [])
        if s["step_name"] in steps_before_resume
    ]

    # Merge delegation results into input
    merged_input = dict(state_snapshot.get("input", {}))
    delegation_results = state_snapshot.get("delegation_results", {})
    if delegation_results:
        merged_input["delegation"] = delegation_results

    initial = {
        "input": merged_input,
        "steps": prior_steps,
        "current_step": "",
        "metadata": state_snapshot.get("metadata", {
            "use_case": config.get("name", ""),
            "description": config.get("description", ""),
        }),
        "loop_counts": dict(state_snapshot.get("loop_counts", {})),
        "routing_log": list(state_snapshot.get("routing_log", [])),
    }

    # Reset loop counts for the resume step
    if resume_step in initial["loop_counts"]:
        initial["loop_counts"][resume_step] = 0

    return initial
