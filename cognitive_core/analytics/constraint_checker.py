"""
Cognitive Core — Deterministic Constraint Checker (TASK 7)

Stage 2 of the two-stage verify primitive. Given evidence characterizations
produced by the LLM in Stage 1, evaluates each rule deterministically.

No LLM is consulted in Stage 2 under any circumstances.
Stage 1 characterizations are immutable once recorded — Stage 2 evaluates
against the recorded characterizations, not a re-run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.analytics.constraint_checker")

CHECKER_VERSION = "constraint_checker_v1"
STAGE1_PROMPT_TEMPLATE = """\
You are performing evidence mapping for compliance verification.
This is Stage 1 of a two-stage verification process. Your task is to
characterize what the evidence shows about each rule variable.
Do NOT render a compliance verdict — that is Stage 2.

CONTEXT:
{context}

EVIDENCE TO MAP:
{subject}

ADDITIONAL INPUT DATA:
{input_data}

RULES AND THEIR VARIABLES:
{rules_json}

For each rule variable listed above, characterize what the available evidence shows.

Respond with JSON matching this structure exactly:
{{
  "characterizations": [
    {{
      "variable": "<variable_name>",
      "characterization": "<what the evidence shows about this variable>",
      "evidence_basis": "<specific evidence cited — quote or reference>",
      "confidence": <0.0 to 1.0>,
      "ambiguity_flags": ["<ambiguity description if any>"]
    }}
  ]
}}

Respond ONLY with the JSON object. No other text.
"""


# ── Stage 1 Output ────────────────────────────────────────────────────────────

@dataclass
class VariableCharacterization:
    variable: str
    characterization: str
    evidence_basis: str
    confidence: float
    ambiguity_flags: list[str] = field(default_factory=list)


@dataclass
class Stage1Result:
    """Immutable record from Stage 1 evidence mapping."""
    characterizations: list[VariableCharacterization]
    llm_version: str
    raw_response: str

    def has_ambiguity(self) -> bool:
        return any(c.ambiguity_flags for c in self.characterizations)

    def ambiguity_summary(self) -> list[str]:
        flags = []
        for c in self.characterizations:
            for f in c.ambiguity_flags:
                flags.append(f"{c.variable}: {f}")
        return flags

    def get_char(self, variable: str) -> VariableCharacterization | None:
        for c in self.characterizations:
            if c.variable == variable:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "characterizations": [
                {
                    "variable": c.variable,
                    "characterization": c.characterization,
                    "evidence_basis": c.evidence_basis,
                    "confidence": c.confidence,
                    "ambiguity_flags": c.ambiguity_flags,
                }
                for c in self.characterizations
            ],
            "llm_version": self.llm_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], raw_response: str = "") -> Stage1Result:
        chars = [
            VariableCharacterization(
                variable=c["variable"],
                characterization=c.get("characterization", ""),
                evidence_basis=c.get("evidence_basis", ""),
                confidence=float(c.get("confidence", 0.5)),
                ambiguity_flags=c.get("ambiguity_flags", []),
            )
            for c in d.get("characterizations", [])
        ]
        return cls(
            characterizations=chars,
            llm_version=d.get("llm_version", "unknown"),
            raw_response=raw_response,
        )


# ── Stage 2 — Deterministic Rule Evaluation ───────────────────────────────────

@dataclass
class RuleResult:
    rule_id: str
    description: str
    verdict: str              # "pass" | "fail" | "insufficient_evidence"
    variables_checked: list[str]
    rationale: str
    ambiguity_flags: list[str] = field(default_factory=list)


@dataclass
class Stage2Result:
    """Deterministic output from Stage 2 constraint checking."""
    rules_evaluated: list[RuleResult]
    overall_conforms: bool
    checker_version: str

    @property
    def violations(self) -> list[RuleResult]:
        return [r for r in self.rules_evaluated if r.verdict == "fail"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules_evaluated": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "verdict": r.verdict,
                    "variables_checked": r.variables_checked,
                    "rationale": r.rationale,
                    "ambiguity_flags": r.ambiguity_flags,
                }
                for r in self.rules_evaluated
            ],
            "overall_conforms": self.overall_conforms,
            "checker_version": self.checker_version,
            "violation_count": len(self.violations),
        }


def evaluate_rules(
    artifact: dict[str, Any],
    stage1: Stage1Result,
    confidence_threshold: float = 0.6,
) -> Stage2Result:
    """
    Deterministically evaluate rules against Stage 1 characterizations.

    A rule PASSES when all its variables have been characterized with
    confidence ≥ threshold AND the characterization does not contain
    explicit failure signals.

    A rule FAILS when at least one variable has a characterization that
    explicitly indicates non-compliance.

    A rule returns INSUFFICIENT_EVIDENCE when not all variables have
    characterizations with confidence ≥ threshold.

    No LLM is consulted here.
    """
    rules = artifact.get("rules", [])
    rule_results = []

    _FAIL_SIGNALS = {
        "violation", "fail", "not met", "non-compliant", "does not comply",
        "missing", "absent", "not provided", "not present", "exceeded",
        "overdue", "late", "past due",
    }

    for rule in rules:
        rule_id = rule.get("rule_id", "unknown")
        description = rule.get("description", "")
        variables = rule.get("variables", [])
        ambiguity_flags: list[str] = []

        chars = [stage1.get_char(v) for v in variables]
        missing_vars = [v for v, c in zip(variables, chars) if c is None]
        low_confidence = [
            c for c in chars
            if c and c.confidence < confidence_threshold
        ]

        if missing_vars:
            rule_results.append(RuleResult(
                rule_id=rule_id,
                description=description,
                verdict="insufficient_evidence",
                variables_checked=variables,
                rationale=f"No characterization for variables: {missing_vars}",
            ))
            continue

        # Collect ambiguity flags from all characterizations
        for c in chars:
            if c:
                ambiguity_flags.extend(c.ambiguity_flags)

        if low_confidence:
            verdict = "insufficient_evidence"
            rationale = (
                f"Low confidence on: "
                + ", ".join(f"{c.variable} ({c.confidence:.2f})" for c in low_confidence)
            )
        else:
            # Check for explicit failure signals in characterizations
            failed_vars = []
            for c in chars:
                if c:
                    text = c.characterization.lower()
                    if any(sig in text for sig in _FAIL_SIGNALS):
                        failed_vars.append(c.variable)

            if failed_vars:
                verdict = "fail"
                rationale = (
                    f"Characterization indicates non-compliance for: {failed_vars}"
                )
            else:
                verdict = "pass"
                rationale = "All variables characterized with sufficient confidence and no violation signals"

        rule_results.append(RuleResult(
            rule_id=rule_id,
            description=description,
            verdict=verdict,
            variables_checked=variables,
            rationale=rationale,
            ambiguity_flags=ambiguity_flags,
        ))

    overall_conforms = all(
        r.verdict == "pass" for r in rule_results
    ) if rule_results else True

    return Stage2Result(
        rules_evaluated=rule_results,
        overall_conforms=overall_conforms,
        checker_version=CHECKER_VERSION,
    )


def build_stage1_prompt(
    artifact: dict[str, Any],
    subject: str,
    context: str,
    input_data: str,
) -> str:
    """Build the Stage 1 evidence mapping prompt."""
    rules = artifact.get("rules", [])
    # Collect all variables across all rules for the evidence mapping
    all_variables: list[dict] = []
    for rule in rules:
        rule_vars = rule.get("variables", [])
        all_variables.append({
            "rule_id": rule.get("rule_id"),
            "description": rule.get("description", ""),
            "variables": rule_vars,
        })

    return STAGE1_PROMPT_TEMPLATE.format(
        context=context or "",
        subject=subject or "No subject provided.",
        input_data=input_data or "",
        rules_json=json.dumps(all_variables, indent=2),
    )
