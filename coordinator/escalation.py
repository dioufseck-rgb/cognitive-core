"""
Cognitive Core — Escalation Brief Builder

When a workflow escalates to a human (via governance gate, quality gate,
confidence drop, or explicit escalation step), this module builds a
structured brief that makes the human faster and more accurate.

The brief contains:
  1. What was the case about (from retrieve data)
  2. What the automation determined (classifications, findings)
  3. What the automation was unsure about (low confidence steps, conflicts)
  4. What specific questions the human should focus on
  5. Evidence gathered so far (so they don't re-investigate)
  6. Recommended priority and any time-sensitive deadlines

Design principle: The human should be able to make a decision in minutes,
not hours. The automation did the legwork; the human applies judgment
to the ambiguous parts.
"""

from __future__ import annotations

import json
from typing import Any


def build_escalation_brief(
    workflow_type: str,
    domain: str,
    final_state: dict[str, Any],
    escalation_reason: str = "",
    quality_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a structured escalation brief from workflow state.

    Args:
        workflow_type: The workflow that ran (e.g., "claim_intake")
        domain: The domain config used (e.g., "synthetic_claim")
        final_state: The full workflow state at escalation time
        escalation_reason: Why this escalated (governance, quality gate, etc.)
        quality_gate: Quality gate details if that's what triggered escalation

    Returns:
        A structured dict suitable for the human review queue.
    """
    steps = final_state.get("steps", [])

    # ── 1. Extract case identity ──────────────────────────────
    case_summary = _extract_case_summary(steps)

    # ── 2. Extract what automation determined ─────────────────
    determinations = _extract_determinations(steps)

    # ── 3. Identify uncertainty ───────────────────────────────
    uncertainties = _identify_uncertainties(steps, quality_gate)

    # ── 4. Generate focus questions ───────────────────────────
    focus_questions = _generate_focus_questions(
        determinations, uncertainties, escalation_reason
    )

    # ── 5. Collect evidence ───────────────────────────────────
    evidence = _collect_evidence(steps)

    # ── 6. Assess priority ────────────────────────────────────
    priority = _assess_priority(determinations, uncertainties)

    brief = {
        "workflow": workflow_type,
        "domain": domain,
        "escalation_reason": escalation_reason,

        # What is this case?
        "case_summary": case_summary,

        # What did the automation figure out?
        "determinations": determinations,

        # What was it unsure about?
        "uncertainties": uncertainties,

        # What should the human focus on?
        "focus_questions": focus_questions,

        # Evidence gathered (don't re-investigate)
        "evidence": evidence,

        # How urgent is this?
        "priority": priority,

        # Raw step count for context
        "steps_completed": len(steps),
    }

    return brief


def _extract_case_summary(steps: list[dict]) -> dict[str, Any]:
    """Pull case identity from retrieve steps."""
    summary = {}
    for step in steps:
        if step.get("primitive") == "retrieve":
            data = step.get("output", {}).get("data", {})
            # Look for common identity fields across data sources
            for source_name, source_data in data.items():
                if not isinstance(source_data, dict):
                    continue
                for key in ["claim_id", "case_id", "dispute_id", "order_id",
                            "member_name", "customer_name", "caller_name",
                            "amount", "description"]:
                    if key in source_data and key not in summary:
                        summary[key] = source_data[key]
    return summary


def _extract_determinations(steps: list[dict]) -> list[dict[str, Any]]:
    """Extract what the automation determined at each step."""
    determinations = []
    for step in steps:
        primitive = step.get("primitive", "")
        step_name = step.get("step_name", "")
        output = step.get("output", {})
        confidence = output.get("confidence")

        if primitive == "classify":
            determinations.append({
                "step": step_name,
                "type": "classification",
                "result": output.get("category"),
                "confidence": confidence,
                "reasoning": _truncate(output.get("rationale")
                                       or output.get("reasoning"), 200),
            })

        elif primitive == "verify":
            determinations.append({
                "step": step_name,
                "type": "verification",
                "result": "passed" if output.get("conforms") else "failed",
                "violations": output.get("violations", []),
                "reasoning": _truncate(output.get("reasoning"), 200),
            })

        elif primitive == "investigate":
            determinations.append({
                "step": step_name,
                "type": "investigation",
                "finding": output.get("finding"),
                "confidence": confidence,
                "recommendation": output.get("recommendation"),
                "evidence_flags": output.get("evidence_flags", []),
            })

        elif primitive == "think":
            determinations.append({
                "step": step_name,
                "type": "assessment",
                "decision": output.get("decision")
                            or output.get("recommendation"),
                "confidence": confidence,
                "reasoning": _truncate(output.get("thought")
                                       or output.get("reasoning"), 300),
            })

        elif primitive == "generate":
            artifact = output.get("artifact")
            if isinstance(artifact, dict):
                # Include the artifact decision but not the full content
                decision = artifact.get("decision")
                recommendation = artifact.get("recommendation")
                if decision or recommendation:
                    determinations.append({
                        "step": step_name,
                        "type": "output",
                        "decision": decision or recommendation,
                        "confidence": confidence,
                    })

    return determinations


def _identify_uncertainties(
    steps: list[dict],
    quality_gate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find where the automation was unsure or conflicted."""
    uncertainties = []

    # Low confidence steps
    for step in steps:
        output = step.get("output", {})
        confidence = output.get("confidence")
        if confidence is not None and confidence < 0.7:
            uncertainties.append({
                "type": "low_confidence",
                "step": step.get("step_name"),
                "confidence": confidence,
                "description": (
                    f"{step.get('step_name')} had low confidence ({confidence:.0%}). "
                    f"The automation wasn't sure about this step."
                ),
            })

    # Quality gate trigger
    if quality_gate:
        uncertainties.append({
            "type": "quality_gate",
            "step": quality_gate.get("step_name"),
            "description": quality_gate.get("reason", "Quality gate fired"),
        })

    # Check for conflicting determinations
    # e.g., classification says low_risk but investigation found patterns
    classifications = {}
    findings = {}
    for step in steps:
        output = step.get("output", {})
        if step.get("primitive") == "classify":
            classifications[step.get("step_name")] = output.get("category")
        if step.get("primitive") == "investigate":
            findings[step.get("step_name")] = output.get("finding")

    # Flag if risk-related classification and investigation seem misaligned
    for cls_name, cls_val in classifications.items():
        for inv_name, inv_val in findings.items():
            if cls_val and inv_val:
                cls_lower = str(cls_val).lower()
                inv_lower = str(inv_val).lower()
                # Simple heuristic: "low" classification with "significant" finding
                if ("low" in cls_lower and "significant" in inv_lower):
                    uncertainties.append({
                        "type": "conflicting_signals",
                        "description": (
                            f"Classification ({cls_name}) said '{cls_val}' "
                            f"but investigation ({inv_name}) found '{inv_val}'. "
                            f"These may be in tension."
                        ),
                    })
                elif ("high" in cls_lower and "no_" in inv_lower):
                    uncertainties.append({
                        "type": "conflicting_signals",
                        "description": (
                            f"Classification ({cls_name}) said '{cls_val}' "
                            f"but investigation ({inv_name}) found '{inv_val}'. "
                            f"The classification flags may not reflect actual patterns."
                        ),
                    })

    return uncertainties


def _generate_focus_questions(
    determinations: list[dict],
    uncertainties: list[dict],
    escalation_reason: str,
) -> list[str]:
    """Generate specific questions the human should focus on."""
    questions = []

    # For each uncertainty, generate a targeted question
    for u in uncertainties:
        if u["type"] == "low_confidence":
            questions.append(
                f"Verify the {u['step']} result — the automation was "
                f"only {u.get('confidence', 0):.0%} confident."
            )
        elif u["type"] == "quality_gate":
            questions.append(
                f"Review flagged step: {u.get('description', '')}"
            )
        elif u["type"] == "conflicting_signals":
            questions.append(
                f"Resolve conflicting signals: {u.get('description', '')}"
            )

    # If there's a verification failure, ask about it
    for d in determinations:
        if d["type"] == "verification" and d["result"] == "failed":
            violations = d.get("violations", [])
            if violations:
                questions.append(
                    f"The claim failed verification: {', '.join(str(v) for v in violations[:3])}. "
                    f"Is this a legitimate denial or a data issue?"
                )

    # If no specific questions emerged, add a generic one
    if not questions:
        questions.append(
            "Review the automation's decision and confirm or override. "
            "The case was escalated because: " + (escalation_reason or "governance review required")
        )

    return questions


def _collect_evidence(steps: list[dict]) -> dict[str, Any]:
    """Collect all evidence gathered so the human doesn't re-investigate."""
    evidence = {
        "data_sources": {},
        "classifications": {},
        "investigation_findings": {},
    }

    for step in steps:
        output = step.get("output", {})
        step_name = step.get("step_name", "")

        if step.get("primitive") == "retrieve":
            data = output.get("data", {})
            for source_name, source_data in data.items():
                evidence["data_sources"][source_name] = source_data

        elif step.get("primitive") == "classify":
            evidence["classifications"][step_name] = {
                "category": output.get("category"),
                "confidence": output.get("confidence"),
            }

        elif step.get("primitive") == "investigate":
            evidence["investigation_findings"][step_name] = {
                "finding": output.get("finding"),
                "confidence": output.get("confidence"),
                "evidence_flags": output.get("evidence_flags", []),
            }

    return evidence


def _assess_priority(
    determinations: list[dict],
    uncertainties: list[dict],
) -> dict[str, Any]:
    """Assess priority level for the human reviewer."""
    # More uncertainties = higher priority
    uncertainty_count = len(uncertainties)
    has_conflicts = any(u["type"] == "conflicting_signals" for u in uncertainties)
    has_quality_gate = any(u["type"] == "quality_gate" for u in uncertainties)

    if has_quality_gate or has_conflicts:
        level = "high"
        reason = "Quality gate fired or conflicting signals detected"
    elif uncertainty_count >= 2:
        level = "high"
        reason = f"{uncertainty_count} areas of uncertainty"
    elif uncertainty_count == 1:
        level = "medium"
        reason = "One area of uncertainty"
    else:
        level = "standard"
        reason = "Routine governance review"

    return {"level": level, "reason": reason}


def _truncate(text: str | None, max_len: int = 200) -> str:
    """Truncate text for brief readability."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
