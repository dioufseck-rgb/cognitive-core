"""
cognitive_core/engine/epistemic.py

Step-level epistemic state computation for Cognitive Core.

Three-layer design (from Session 5 handoff spec):

  Layer 1: Mechanical — computed by framework from existing output fields.
           No LLM changes required. Purely deterministic.
           evidence_completeness, rule_coverage, citation_rate

  Layer 2: Judgment — LLM-reported, per-component scores.
           reasoning_quality, outcome_certainty, alternative_separation
           (Added to prompts in a later session — not implemented here)

  Layer 3: Coherence — computed by framework across accumulated steps.
           Flags structural problems the LLM cannot self-report reliably.
           CLASSIFY_DELIBERATE_MISMATCH, VERIFY_DELIBERATE_TENSION,
           CONFIDENCE_DROP, UNRESOLVED_GAPS

The `overall` score is framework-computed, never LLM-reported:
  overall = mean(mechanical) * coherence_multiplier

The `warranted` flag is binary: is this step's output supportable
given what the framework can verify mechanically?

These are accumulated into a WorkflowEpistemicRecord that travels
alongside the workflow state and is attached to the escalation brief.
The reviewer work order changes from "confidence: 0.72" to structured
epistemic context: what was measured, what gaps exist, what conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Coherence flag types
# ─────────────────────────────────────────────────────────────────────────────

class CoherenceFlag(str, Enum):
    """
    Structural problems detected by the framework across steps.
    Each flag has a defined reduction to the coherence multiplier.
    """
    # classify said X, deliberate recommended action inconsistent with X
    CLASSIFY_DELIBERATE_MISMATCH    = "CLASSIFY_DELIBERATE_MISMATCH"
    # verify found violations, deliberate still recommended proceed/approve
    VERIFY_DELIBERATE_TENSION       = "VERIFY_DELIBERATE_TENSION"
    # any step confidence dropped >0.25 from the prior step
    CONFIDENCE_DROP                 = "CONFIDENCE_DROP"
    # evidence_missing items from retrieve/investigate were never addressed
    UNRESOLVED_EVIDENCE_GAPS        = "UNRESOLVED_EVIDENCE_GAPS"
    # govern tier is higher than what deliberate confidence would suggest
    GOVERN_ESCALATION_UNEXPLAINED   = "GOVERN_ESCALATION_UNEXPLAINED"
    # deliberate recommended action has no warrant field
    UNWARRANTED_RECOMMENDATION      = "UNWARRANTED_RECOMMENDATION"


# Coherence multiplier reductions per flag (additive, floored at 0.3)
COHERENCE_PENALTIES: dict[CoherenceFlag, float] = {
    CoherenceFlag.CLASSIFY_DELIBERATE_MISMATCH:  0.20,
    CoherenceFlag.VERIFY_DELIBERATE_TENSION:     0.25,
    CoherenceFlag.CONFIDENCE_DROP:               0.10,
    CoherenceFlag.UNRESOLVED_EVIDENCE_GAPS:      0.10,
    CoherenceFlag.GOVERN_ESCALATION_UNEXPLAINED: 0.15,
    CoherenceFlag.UNWARRANTED_RECOMMENDATION:    0.15,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-step epistemic state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepEpistemicState:
    """
    Epistemic state for a single workflow step.

    Mechanical components are computed from existing output fields —
    no LLM prompt changes required.

    Judgment components (reasoning_quality, outcome_certainty,
    alternative_separation) are placeholders here — they'll be
    LLM-reported once we add them to the primitive prompts.
    """
    step_name: str
    primitive: str

    # ── Layer 1: Mechanical (framework-computed) ──────────────────
    # retrieve: sources with data / sources specified in prompt
    evidence_completeness: float | None = None
    # verify: rules checked / rules applicable (from rules_checked list)
    rule_coverage: float | None = None
    # deliberate/classify: claims with cited evidence / total claims
    # Proxy: evidence_used count / max(1, claim sentences in output)
    citation_rate: float | None = None

    # ── Layer 2: Judgment (LLM-reported — not yet wired) ─────────
    reasoning_quality: float | None = None    # is the logic sound?
    outcome_certainty: float | None = None    # how clearly does evidence support conclusion?
    alternative_separation: float | None = None  # classify only: margin over next-best

    # ── Layer 3: Coherence (framework-computed cross-step) ────────
    coherence_flags: list[CoherenceFlag] = field(default_factory=list)
    inherited_gaps: list[str] = field(default_factory=list)   # unresolved from prior steps
    resolved_gaps: list[str] = field(default_factory=list)    # gaps this step addressed

    # ── Derived ───────────────────────────────────────────────────
    overall: float = 0.0       # framework-computed aggregate
    warranted: bool = True     # binary: is this step's output supportable?

    # LLM-reported confidence (pass-through from output)
    llm_confidence: float | None = None

    def mechanical_scores(self) -> list[float]:
        """Return non-None mechanical scores for aggregation."""
        return [s for s in [
            self.evidence_completeness,
            self.rule_coverage,
            self.citation_rate,
        ] if s is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "primitive": self.primitive,
            "evidence_completeness": self.evidence_completeness,
            "rule_coverage": self.rule_coverage,
            "citation_rate": self.citation_rate,
            "reasoning_quality": self.reasoning_quality,
            "outcome_certainty": self.outcome_certainty,
            "alternative_separation": self.alternative_separation,
            "coherence_flags": [f.value for f in self.coherence_flags],
            "inherited_gaps": self.inherited_gaps,
            "resolved_gaps": self.resolved_gaps,
            "overall": round(self.overall, 3),
            "warranted": self.warranted,
            "llm_confidence": self.llm_confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Workflow-level epistemic record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkflowEpistemicRecord:
    """
    Accumulated epistemic state across all steps in a workflow.
    Built incrementally as each step completes.
    Attached to the escalation brief so the reviewer sees structured
    epistemic context, not just a scalar confidence number.
    """
    instance_id: str
    steps: list[StepEpistemicState] = field(default_factory=list)
    all_flags: list[tuple[str, CoherenceFlag]] = field(default_factory=list)  # (step_name, flag)
    open_gaps: list[str] = field(default_factory=list)  # evidence gaps not yet resolved

    def add_step(self, step_state: StepEpistemicState) -> None:
        self.steps.append(step_state)
        for flag in step_state.coherence_flags:
            self.all_flags.append((step_state.step_name, flag))
        # Track open gaps
        for gap in step_state.inherited_gaps:
            if gap not in self.open_gaps:
                self.open_gaps.append(gap)
        for gap in step_state.resolved_gaps:
            if gap in self.open_gaps:
                self.open_gaps.remove(gap)

    def workflow_overall(self) -> float:
        """Aggregate overall score across all steps."""
        if not self.steps:
            return 0.0
        scores = [s.overall for s in self.steps if s.overall > 0]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def flag_summary(self) -> list[str]:
        """Human-readable flag summary for the escalation brief."""
        if not self.all_flags:
            return []
        seen = {}
        for step_name, flag in self.all_flags:
            key = flag.value
            if key not in seen:
                seen[key] = step_name
        return [f"{flag} (at {step})" for flag, step in seen.items()]

    def reviewer_context(self) -> dict[str, Any]:
        """
        Structured context for the human reviewer work order.
        Replaces the single 'confidence' scalar with actionable information.
        """
        low_confidence_steps = [
            {"step": s.step_name, "primitive": s.primitive,
             "evidence_completeness": s.evidence_completeness,
             "rule_coverage": s.rule_coverage,
             "overall": s.overall,
             "flags": [f.value for f in s.coherence_flags]}
            for s in self.steps
            if s.overall < 0.65 or s.coherence_flags
        ]
        return {
            "workflow_overall": self.workflow_overall(),
            "coherence_flags": self.flag_summary(),
            "open_evidence_gaps": self.open_gaps,
            "low_confidence_steps": low_confidence_steps,
            "step_count": len(self.steps),
            "warranted": all(s.warranted for s in self.steps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "workflow_overall": self.workflow_overall(),
            "steps": [s.to_dict() for s in self.steps],
            "all_flags": [(sn, f.value) for sn, f in self.all_flags],
            "open_gaps": self.open_gaps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mechanical computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mechanical(
    primitive: str,
    output: dict[str, Any],
    workflow_spec_steps: list[dict] | None = None,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute (evidence_completeness, rule_coverage, citation_rate)
    from existing output fields. Purely deterministic.

    Returns None for metrics that don't apply to this primitive.
    """
    evidence_completeness = None
    rule_coverage = None
    citation_rate = None

    if primitive == "retrieve":
        sources_queried = output.get("sources_queried", [])
        if sources_queried:
            successful = sum(
                1 for s in sources_queried
                if s.get("status") == "success"
                and s.get("data") is not None
                and s.get("data") != {}
            )
            # Use record_count > 0 as a proxy for "has real data"
            with_data = sum(
                1 for s in sources_queried
                if s.get("status") == "success" and (
                    s.get("record_count") is None  # not tracked → assume present
                    or s.get("record_count", 0) > 0
                )
            )
            total = len(sources_queried)
            if total > 0:
                evidence_completeness = round(with_data / total, 3)

        # Fallback: count data keys vs skipped
        if evidence_completeness is None:
            data = output.get("data", {})
            skipped = output.get("sources_skipped", [])
            total = len(data) + len(skipped)
            if total > 0:
                evidence_completeness = round(len(data) / total, 3)

    elif primitive == "verify":
        rules_checked = output.get("rules_checked", [])
        violations = output.get("violations", [])
        # Now that verify.txt instructs the LLM to list ALL rules in
        # rules_checked (both passing and failing), this list is the
        # ground truth for coverage. Previously only violations were
        # listed, making coverage appear low even on clean passes.
        #
        # rule_coverage = rules_checked / max(rules_checked, violations)
        # This handles:
        #   rules_checked=4, violations=1 → 4/4 = 1.0 (full coverage, one violation)
        #   rules_checked=0, violations=1 → 0/1 = 0.0 (no coverage recorded)
        #   rules_checked=2, violations=3 → 2/3 ≈ 0.67 (old behaviour: incomplete)
        if rules_checked:
            applicable = max(len(rules_checked), len(violations))
            rule_coverage = round(len(rules_checked) / applicable, 3)
        elif violations:
            # Violations without rules_checked — coverage unknown but clearly incomplete
            rule_coverage = 0.0
        else:
            # No rules checked, no violations — nothing to verify, skip
            rule_coverage = None

    elif primitive in ("deliberate", "classify", "govern", "generate"):
        # Citation rate: evidence_used count as proxy for citation
        evidence_used = output.get("evidence_used", [])
        # Claim proxy: number of sentences in warrant/reasoning
        warrant = output.get("warrant", "") or output.get("reasoning", "") or ""
        claim_sentences = max(1, warrant.count(".") + warrant.count(";"))
        if evidence_used:
            # Each piece of evidence cited supports ~2 claims on average
            supported = min(len(evidence_used) * 2, claim_sentences)
            citation_rate = round(supported / claim_sentences, 3)
        elif warrant:
            # Has reasoning but no evidence_used — low citation
            citation_rate = 0.1
        else:
            citation_rate = 0.0

    return evidence_completeness, rule_coverage, citation_rate


# ─────────────────────────────────────────────────────────────────────────────
# Coherence flag computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_coherence_flags(
    current_step: dict[str, Any],
    prior_steps: list[dict[str, Any]],
    open_gaps: list[str],
) -> tuple[list[CoherenceFlag], list[str], list[str]]:
    """
    Compute coherence flags, inherited gaps, and resolved gaps for a step.

    Args:
        current_step: The step that just completed (with output)
        prior_steps: All steps that completed before this one
        open_gaps: Evidence gaps not yet resolved from prior steps

    Returns:
        (flags, inherited_gaps, resolved_gaps)
    """
    flags: list[CoherenceFlag] = []
    inherited = list(open_gaps)  # carry forward
    resolved: list[str] = []

    primitive = current_step.get("primitive", "")
    output = current_step.get("output", {})
    current_confidence = output.get("confidence")

    # ── CLASSIFY_DELIBERATE_MISMATCH ─────────────────────────────
    # deliberate recommended_action is inconsistent with classify category
    if primitive == "deliberate":
        recommended = (output.get("recommended_action") or "").lower()
        for step in prior_steps:
            if step.get("primitive") == "classify":
                category = (step.get("output", {}).get("category") or "").lower()
                if _classify_deliberate_conflict(category, recommended):
                    flags.append(CoherenceFlag.CLASSIFY_DELIBERATE_MISMATCH)
                    break

    # ── VERIFY_DELIBERATE_TENSION ────────────────────────────────
    # verify found violations AND deliberate said approve/proceed,
    # AND the warrant does not explicitly resolve the tension.
    #
    # Resolution check: if the deliberate warrant explicitly acknowledges
    # the violation and explains why approval is still correct (e.g. the
    # MND pathway, mitigation measures, or conditions of approval), the
    # tension is resolved and the flag should not fire.
    # This prevents false positives on the CEQA MND pattern where a failed
    # categorical exemption check is the expected finding that triggers the
    # MND pathway — and the correct deliberate response IS to approve with
    # conditions, not to deny.
    if primitive == "deliberate":
        recommended = (output.get("recommended_action") or "").lower()
        warrant = (output.get("warrant") or "").lower()
        has_approve = any(w in recommended for w in (
            "approve", "proceed", "grant", "accept", "certify"
        ))
        # Warrant resolution indicators: explicit acknowledgment of the
        # violation pathway and why approval is still warranted
        RESOLUTION_INDICATORS = (
            "mnd", "mitigated negative declaration",
            "mitigation", "mitigated",
            "conditions of approval", "with conditions",
            "initial study", "mnd pathway",
            "negative declaration",
        )
        warrant_resolves = any(ind in warrant for ind in RESOLUTION_INDICATORS)
        for step in prior_steps:
            if step.get("primitive") == "verify":
                violations = step.get("output", {}).get("violations", [])
                conforms = step.get("output", {}).get("conforms", True)
                if not conforms and violations and has_approve:
                    if not warrant_resolves:
                        # Genuine tension: approved despite violations with no
                        # explicit resolution in the warrant
                        flags.append(CoherenceFlag.VERIFY_DELIBERATE_TENSION)
                    # If warrant_resolves: tension acknowledged and addressed —
                    # no flag. The govern step still sees the violation in its
                    # workflow_state and can escalate to GATE on other grounds.
                    break

    # ── CONFIDENCE_DROP ──────────────────────────────────────────
    # current confidence drops >0.25 from the immediately prior step
    if current_confidence is not None and prior_steps:
        prior_confidences = [
            s.get("output", {}).get("confidence")
            for s in prior_steps[-3:]  # look back 3 steps
            if s.get("output", {}).get("confidence") is not None
        ]
        if prior_confidences:
            prior_avg = sum(prior_confidences) / len(prior_confidences)
            if prior_avg - current_confidence > 0.25:
                flags.append(CoherenceFlag.CONFIDENCE_DROP)

    # ── UNRESOLVED_EVIDENCE_GAPS ─────────────────────────────────
    # retrieve/investigate declared evidence_missing; later steps ignored it
    if primitive in ("deliberate", "govern"):
        if open_gaps:
            # Check if any upstream gaps are referenced in the current output
            reasoning_text = (
                (output.get("reasoning") or "") +
                (output.get("warrant") or "") +
                (output.get("situation_summary") or "") +
                (output.get("tier_rationale") or "")
            ).lower()
            still_open = []
            for gap in open_gaps:
                if gap.lower()[:20] in reasoning_text:
                    resolved.append(gap)
                else:
                    still_open.append(gap)
            if still_open:
                flags.append(CoherenceFlag.UNRESOLVED_EVIDENCE_GAPS)
            inherited = still_open

    # ── UNWARRANTED_RECOMMENDATION ───────────────────────────────
    # deliberate has a recommended_action but no warrant
    if primitive == "deliberate":
        recommended = output.get("recommended_action")
        warrant = output.get("warrant")
        if recommended and not warrant:
            flags.append(CoherenceFlag.UNWARRANTED_RECOMMENDATION)

    # ── GOVERN_ESCALATION_UNEXPLAINED ────────────────────────────
    # govern escalated to gate/hold but prior deliberate was high confidence
    if primitive == "govern":
        tier = str(output.get("tier_applied", "")).lower().replace("governancetier.", "")
        if tier in ("gate", "hold"):
            for step in prior_steps:
                if step.get("primitive") == "deliberate":
                    delib_conf = step.get("output", {}).get("confidence", 0)
                    tier_rationale = (output.get("tier_rationale") or "").strip()
                    if delib_conf >= 0.85 and not tier_rationale:
                        flags.append(CoherenceFlag.GOVERN_ESCALATION_UNEXPLAINED)
                    break

    # ── Collect new evidence gaps from current step ───────────────
    evidence_missing = output.get("evidence_missing", [])
    for item in evidence_missing:
        if isinstance(item, dict):
            desc = item.get("description", item.get("source", str(item)))
        else:
            desc = str(item)
        if desc and desc not in inherited and desc not in resolved:
            inherited.append(desc)

    return flags, inherited, resolved


def _classify_deliberate_conflict(category: str, recommended_action: str) -> bool:
    """
    Heuristic: does the deliberate recommendation conflict with the classification?
    Extend this as domain knowledge accumulates.
    """
    # Exempt category + approve/proceed = fine
    if "exempt" in category and any(w in recommended_action for w in ("exempt", "proceed", "approve")):
        return False
    # Prohibited + approve = conflict
    if "prohibit" in category and any(w in recommended_action for w in ("approve", "grant", "proceed")):
        return True
    # High-risk + no mention of conditions/review = borderline, not flagged here
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Overall score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_overall(
    mechanical_scores: list[float],
    judgment_scores: list[float],
    coherence_flags: list[CoherenceFlag],
) -> tuple[float, bool]:
    """
    Compute overall epistemic score and warranted flag.

    When only mechanical scores are present:
      overall = mean(mechanical) * coherence_multiplier

    When judgment scores are also present (LLM-reported):
      overall = (0.6 * mean(mechanical) + 0.4 * mean(judgment)) * coherence_multiplier

    Judgment scores are weighted lower than mechanical scores because they
    are self-reported — the LLM cannot reliably assess its own quality.
    Mechanical scores are computed from observable output structure.

    If no mechanical or judgment scores, fall back to 1.0 (not penalised).
    warranted = overall >= 0.5 and no CRITICAL flags
    """
    mechanical_mean = (
        sum(mechanical_scores) / len(mechanical_scores)
        if mechanical_scores else 1.0  # no measurement = assume ok
    )
    if judgment_scores:
        judgment_mean = sum(judgment_scores) / len(judgment_scores)
        combined_mean = 0.6 * mechanical_mean + 0.4 * judgment_mean
    else:
        combined_mean = mechanical_mean

    # Coherence multiplier: start at 1.0, reduce by each flag's penalty
    coherence_multiplier = 1.0
    for flag in coherence_flags:
        coherence_multiplier -= COHERENCE_PENALTIES.get(flag, 0.10)
    coherence_multiplier = max(0.3, coherence_multiplier)

    overall = round(combined_mean * coherence_multiplier, 3)

    # Critical flags that always make warranted=False regardless of scores
    critical_flags = {
        CoherenceFlag.CLASSIFY_DELIBERATE_MISMATCH,
        CoherenceFlag.VERIFY_DELIBERATE_TENSION,
    }
    warranted = (
        overall >= 0.5
        and not any(f in critical_flags for f in coherence_flags)
    )

    return overall, warranted


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: compute full step epistemic state
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_epistemic_state(
    step_output: dict[str, Any],
    prior_steps: list[dict[str, Any]],
    open_gaps: list[str],
    workflow_spec_steps: list[dict] | None = None,
) -> StepEpistemicState:
    """
    Compute the full epistemic state for a completed step.
    Called from the step_callback in the coordinator after each step completes.

    Args:
        step_output: The completed step dict (step_name, primitive, output, confidence)
        prior_steps: All previously completed steps in this workflow
        open_gaps: Evidence gaps inherited from prior steps
        workflow_spec_steps: Optional workflow spec for rules_applicable count

    Returns:
        StepEpistemicState with all computed fields
    """
    primitive = step_output.get("primitive", "")
    step_name = step_output.get("step_name", "")
    output = step_output.get("output", {})
    llm_confidence = output.get("confidence")

    # Layer 1: Mechanical
    evidence_completeness, rule_coverage, citation_rate = compute_mechanical(
        primitive, output, workflow_spec_steps
    )

    # Layer 3: Coherence (cross-step)
    coherence_flags, inherited_gaps, resolved_gaps = compute_coherence_flags(
        step_output, prior_steps, open_gaps
    )

    # Derived: overall and warranted
    mechanical_scores = [s for s in [evidence_completeness, rule_coverage, citation_rate]
                         if s is not None]
    overall, warranted = compute_overall(mechanical_scores, [], coherence_flags)

    # Layer 2: pull judgment fields if LLM reported them
    reasoning_quality = output.get("reasoning_quality")
    outcome_certainty = output.get("outcome_certainty")

    # If judgment scores are present, include them in overall computation
    judgment_scores = [s for s in [reasoning_quality, outcome_certainty] if s is not None]
    if judgment_scores:
        overall, warranted = compute_overall(mechanical_scores, judgment_scores, coherence_flags)

    return StepEpistemicState(
        step_name=step_name,
        primitive=primitive,
        evidence_completeness=evidence_completeness,
        rule_coverage=rule_coverage,
        citation_rate=citation_rate,
        reasoning_quality=reasoning_quality,
        outcome_certainty=outcome_certainty,
        coherence_flags=coherence_flags,
        inherited_gaps=inherited_gaps,
        resolved_gaps=resolved_gaps,
        overall=overall,
        warranted=warranted,
        llm_confidence=llm_confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate trigger DSL
# ─────────────────────────────────────────────────────────────────────────────
#
# gate_triggers is a list of condition strings (YAML) evaluated against
# the workflow's epistemic record. Any condition that fires causes escalation.
#
# Supported syntax:
#   any_step.evidence_completeness < 0.70    — any step below threshold
#   any_step.rule_coverage < 0.80
#   any_step.citation_rate < 0.60
#   any_step.overall < 0.65
#   <primitive>.evidence_completeness < N    — specific primitive type
#   <primitive>.rule_coverage < N
#   <primitive>.citation_rate < N
#   <primitive>.overall < N
#   coherence_flag: VERIFY_DELIBERATE_TENSION  — any step has this flag
#   confidence_drop > N                        — any step dropped > N from prior
#   not_warranted                              — any step warranted=False
#
# Backward compatible: if quality_gates.min_confidence / primitive_floors
# is present (old scalar form), those still work via _evaluate_quality_gate.
# gate_triggers is an additive extension — both can coexist.

import re as _re


def evaluate_gate_triggers(
    triggers: list,
    record: "WorkflowEpistemicRecord",
) -> list[dict]:
    """
    Evaluate gate trigger conditions against a WorkflowEpistemicRecord.

    Accepts triggers as strings or dicts. YAML parses
      - coherence_flag: VERIFY_DELIBERATE_TENSION
    as a dict {"coherence_flag": "VERIFY_DELIBERATE_TENSION"}, so both
    forms are normalised to the string form before evaluation.

    Returns a list of fired trigger dicts (empty = no gates fired).
    Each dict has: trigger, step_name, value, threshold, description.
    """
    fired = []
    for trigger in triggers:
        # Normalise dict form to string
        if isinstance(trigger, dict):
            if "coherence_flag" in trigger:
                trigger_str = f"coherence_flag: {trigger['coherence_flag']}"
            else:
                # Unknown dict form — join as key: value
                k, v = next(iter(trigger.items()))
                trigger_str = f"{k}: {v}"
        else:
            trigger_str = str(trigger).strip()
        result = _evaluate_single_trigger(trigger_str, record)
        if result:
            fired.append(result)
    return fired


def _evaluate_single_trigger(trigger: str, record: "WorkflowEpistemicRecord") -> dict | None:
    """Evaluate a single trigger string. Returns fired dict or None."""

    # ── coherence_flag: FLAG_NAME ─────────────────────────────────
    if trigger.startswith("coherence_flag:"):
        flag_name = trigger.split(":", 1)[1].strip()
        for step_name, flag in record.all_flags:
            if flag.value == flag_name:
                return {
                    "trigger": trigger,
                    "step_name": step_name,
                    "flag": flag_name,
                    "description": f"Coherence flag {flag_name} fired at step '{step_name}'",
                }
        return None

    # ── not_warranted ─────────────────────────────────────────────
    if trigger.strip() == "not_warranted":
        for step in record.steps:
            if not step.warranted:
                return {
                    "trigger": trigger,
                    "step_name": step.step_name,
                    "description": (
                        f"Step '{step.step_name}' ({step.primitive}) is not warranted "
                        f"(overall={step.overall:.2f}, flags={[f.value for f in step.coherence_flags]})"
                    ),
                }
        return None

    # ── confidence_drop > N ───────────────────────────────────────
    m = _re.match(r"confidence_drop\s*>\s*([0-9.]+)", trigger)
    if m:
        threshold = float(m.group(1))
        confs = [
            s.llm_confidence for s in record.steps
            if s.llm_confidence is not None
        ]
        for i in range(1, len(confs)):
            drop = confs[i - 1] - confs[i]
            if drop > threshold:
                return {
                    "trigger": trigger,
                    "step_name": record.steps[i].step_name,
                    "value": round(drop, 3),
                    "threshold": threshold,
                    "description": (
                        f"Confidence dropped {drop:.2f} at '{record.steps[i].step_name}' "
                        f"(from {confs[i-1]:.2f} to {confs[i]:.2f}) > threshold {threshold}"
                    ),
                }
        return None

    # ── [scope].[metric] [op] [threshold] ────────────────────────
    # scope: any_step | <primitive_name>
    # metric: evidence_completeness | rule_coverage | citation_rate | overall
    # op: < | <= | > | >=
    m = _re.match(
        r"(\w+)\.(evidence_completeness|rule_coverage|citation_rate|overall)"
        r"\s*([<>]=?)\s*([0-9.]+)",
        trigger,
    )
    if m:
        scope, metric, op, threshold_str = m.groups()
        threshold = float(threshold_str)

        if scope == "any_step":
            candidates = record.steps
        else:
            # scope is a primitive name
            candidates = [s for s in record.steps if s.primitive == scope]

        for step in candidates:
            value = getattr(step, metric, None)
            if value is None:
                continue
            if _compare(value, op, threshold):
                return {
                    "trigger": trigger,
                    "step_name": step.step_name,
                    "primitive": step.primitive,
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "description": (
                        f"Step '{step.step_name}' ({step.primitive}): "
                        f"{metric}={value:.2f} {op} {threshold}"
                    ),
                }
        return None

    # Unrecognised trigger — log and skip
    return None


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == "<":  return value < threshold
    if op == "<=": return value <= threshold
    if op == ">":  return value > threshold
    if op == ">=": return value >= threshold
    return False