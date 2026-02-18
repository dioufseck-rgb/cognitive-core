"""
Cognitive Core — Eval Framework

Runs a workflow against a pack of cases with expected results,
scores each output, and produces an acceptance report.

Quality gates:
  - Schema validity: LLM output parses to Pydantic schema
  - Classification accuracy: correct category for classify steps
  - Investigation quality: finding contains expected evidence
  - Confidence calibration: high confidence on clear cases, low on ambiguous
  - Fail-closed: low confidence triggers escalation, not bad decisions
  - Citation coverage: claims reference available evidence

Usage:
    python -m evals.runner --pack evals/packs/product_return.yaml
    python -m evals.runner --pack evals/packs/card_dispute.yaml --verbose
    python -m evals.runner --list

From code:
    from evals.runner import run_eval_pack, EvalResult
    result = run_eval_pack("evals/packs/product_return.yaml", project_root=".")
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CaseExpectation:
    """What we expect from a single eval case."""
    case_id: str
    case_file: str
    category: str                          # classification, adversarial, edge
    description: str

    # Classification expectations (for classify steps)
    expected_classification: str | None = None
    classification_step: str = "classify_return_type"

    # Investigation expectations
    expected_finding_contains: list[str] = field(default_factory=list)
    expected_evidence_flags: list[str] = field(default_factory=list)
    investigation_step: str = "investigate_claim"

    # Decision expectations (for think steps)
    expected_decision_contains: str | None = None
    decision_step: str = "decide_resolution"

    # Confidence expectations
    min_confidence: float | None = None
    max_confidence: float | None = None

    # Escalation expectations
    should_escalate: bool = False

    # Generate expectations
    must_not_contain: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)


@dataclass
class CaseScore:
    """Scored result for a single eval case."""
    case_id: str
    category: str
    description: str
    passed: bool
    checks: list[dict[str, Any]]   # [{name, passed, expected, actual, detail}]
    elapsed_seconds: float = 0.0
    error: str | None = None

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c["passed"])


@dataclass
class AcceptanceCriteria:
    """Quality gates for the eval pack."""
    min_schema_valid_pct: float = 95.0
    min_classification_accuracy_pct: float = 80.0
    min_investigation_quality_pct: float = 70.0
    min_confidence_calibration_pct: float = 75.0
    max_unsupported_claims_pct: float = 15.0
    min_fail_closed_pct: float = 100.0     # MUST escalate on ambiguous cases
    min_generate_compliance_pct: float = 85.0


@dataclass
class EvalResult:
    """Full result for an eval pack."""
    pack_name: str
    workflow: str
    domain: str
    cases: list[CaseScore]
    criteria: AcceptanceCriteria
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def error_cases(self) -> list[CaseScore]:
        return [c for c in self.cases if c.error]

    def gate_results(self) -> dict[str, dict[str, Any]]:
        """Evaluate each quality gate against the criteria."""
        gates = {}

        # Schema validity
        schema_checks = [c for cs in self.cases for c in cs.checks if c["name"] == "schema_valid"]
        if schema_checks:
            pct = sum(1 for c in schema_checks if c["passed"]) / len(schema_checks) * 100
            gates["schema_valid"] = {
                "actual": round(pct, 1),
                "threshold": self.criteria.min_schema_valid_pct,
                "passed": pct >= self.criteria.min_schema_valid_pct,
            }

        # Classification accuracy
        cls_checks = [c for cs in self.cases for c in cs.checks if c["name"] == "classification"]
        if cls_checks:
            pct = sum(1 for c in cls_checks if c["passed"]) / len(cls_checks) * 100
            gates["classification_accuracy"] = {
                "actual": round(pct, 1),
                "threshold": self.criteria.min_classification_accuracy_pct,
                "passed": pct >= self.criteria.min_classification_accuracy_pct,
            }

        # Investigation quality
        inv_checks = [c for cs in self.cases for c in cs.checks if c["name"] == "investigation_finding"]
        if inv_checks:
            pct = sum(1 for c in inv_checks if c["passed"]) / len(inv_checks) * 100
            gates["investigation_quality"] = {
                "actual": round(pct, 1),
                "threshold": self.criteria.min_investigation_quality_pct,
                "passed": pct >= self.criteria.min_investigation_quality_pct,
            }

        # Confidence calibration
        conf_checks = [c for cs in self.cases for c in cs.checks if c["name"] == "confidence_range"]
        if conf_checks:
            pct = sum(1 for c in conf_checks if c["passed"]) / len(conf_checks) * 100
            gates["confidence_calibration"] = {
                "actual": round(pct, 1),
                "threshold": self.criteria.min_confidence_calibration_pct,
                "passed": pct >= self.criteria.min_confidence_calibration_pct,
            }

        # Fail-closed (escalation on ambiguous)
        esc_checks = [c for cs in self.cases for c in cs.checks if c["name"] == "fail_closed"]
        if esc_checks:
            pct = sum(1 for c in esc_checks if c["passed"]) / len(esc_checks) * 100
            gates["fail_closed"] = {
                "actual": round(pct, 1),
                "threshold": self.criteria.min_fail_closed_pct,
                "passed": pct >= self.criteria.min_fail_closed_pct,
            }

        # Generate compliance
        gen_checks = [c for cs in self.cases for c in cs.checks
                      if c["name"] in ("must_contain", "must_not_contain")]
        if gen_checks:
            pct = sum(1 for c in gen_checks if c["passed"]) / len(gen_checks) * 100
            gates["generate_compliance"] = {
                "actual": round(pct, 1),
                "threshold": self.criteria.min_generate_compliance_pct,
                "passed": pct >= self.criteria.min_generate_compliance_pct,
            }

        return gates

    @property
    def all_gates_pass(self) -> bool:
        return all(g["passed"] for g in self.gate_results().values())

    def summary(self, verbose: bool = False) -> str:
        lines = []
        lines.append(f"\n{'=' * 66}")
        lines.append(f"  EVAL: {self.pack_name}")
        lines.append(f"  {self.workflow} / {self.domain}  —  {self.total} cases in {self.elapsed_seconds:.1f}s")
        lines.append(f"{'=' * 66}")

        # Per-case results
        for cs in self.cases:
            icon = "✓" if cs.passed else "✗"
            tag = f"[{cs.category}]"
            lines.append(f"  {icon} {cs.case_id} {tag:15s} {cs.pass_count}/{cs.check_count} checks  ({cs.elapsed_seconds:.1f}s)")
            if verbose or not cs.passed:
                for check in cs.checks:
                    ci = "  ✓" if check["passed"] else "  ✗"
                    lines.append(f"      {ci} {check['name']}: {check.get('detail', '')}")
            if cs.error:
                lines.append(f"      ERROR: {cs.error}")

        # Quality gates
        lines.append(f"\n{'─' * 66}")
        lines.append(f"  QUALITY GATES")
        lines.append(f"{'─' * 66}")
        gates = self.gate_results()
        for name, g in gates.items():
            icon = "✓" if g["passed"] else "✗"
            lines.append(f"  {icon} {name}: {g['actual']}% (threshold: {g['threshold']}%)")

        # Verdict
        lines.append(f"\n{'─' * 66}")
        if self.all_gates_pass:
            lines.append(f"  ✓ ALL GATES PASSED — {self.passed}/{self.total} cases passed")
        else:
            failed_gates = [n for n, g in gates.items() if not g["passed"]]
            lines.append(f"  ✗ GATES FAILED: {', '.join(failed_gates)}")
            lines.append(f"    {self.passed}/{self.total} cases passed, {len(self.error_cases)} errors")
        lines.append(f"{'=' * 66}\n")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Scoring engine
# ═══════════════════════════════════════════════════════════════════

def score_case(
    expectation: CaseExpectation,
    final_state: dict[str, Any],
) -> CaseScore:
    """Score a workflow execution against expectations."""
    checks = []
    steps = final_state.get("steps", [])

    # Helper: find step output by name (last occurrence)
    def get_output(step_name: str) -> dict[str, Any] | None:
        result = None
        for s in steps:
            if s.get("step_name") == step_name:
                result = s.get("output", {})
        return result

    # ── Schema validity ──
    # Every step should have parsed output (no _parse_failed flag)
    parse_failures = sum(1 for s in steps if s.get("output", {}).get("_parse_failed"))
    checks.append({
        "name": "schema_valid",
        "passed": parse_failures == 0,
        "expected": "0 parse failures",
        "actual": f"{parse_failures} parse failures",
        "detail": f"{len(steps)} steps, {parse_failures} parse failures",
    })

    # ── Classification ──
    if expectation.expected_classification:
        cls_output = get_output(expectation.classification_step)
        if cls_output:
            actual = cls_output.get("category", "").lower().strip()
            expected = expectation.expected_classification.lower().strip()
            checks.append({
                "name": "classification",
                "passed": actual == expected,
                "expected": expected,
                "actual": actual,
                "detail": f"expected '{expected}', got '{actual}' (conf: {cls_output.get('confidence', '?')})",
            })
        else:
            checks.append({
                "name": "classification",
                "passed": False,
                "expected": expectation.expected_classification,
                "actual": "step not found",
                "detail": f"step '{expectation.classification_step}' not in output",
            })

    # ── Investigation finding ──
    if expectation.expected_finding_contains:
        inv_output = get_output(expectation.investigation_step)
        if inv_output:
            # Search finding AND reasoning — the model may ground evidence
            # in the reasoning chain even if the finding summary is concise
            finding = str(inv_output.get("finding", "")).lower()
            reasoning = str(inv_output.get("reasoning", "")).lower()
            search_text = finding + " " + reasoning
            found = [kw for kw in expectation.expected_finding_contains if kw.lower() in search_text]
            missing = [kw for kw in expectation.expected_finding_contains if kw.lower() not in search_text]
            checks.append({
                "name": "investigation_finding",
                "passed": len(missing) == 0,
                "expected": expectation.expected_finding_contains,
                "actual": f"found {len(found)}/{len(expectation.expected_finding_contains)}",
                "detail": f"found: {found}, missing: {missing}" if missing else f"all {len(found)} keywords found",
            })
        else:
            checks.append({
                "name": "investigation_finding",
                "passed": False,
                "expected": expectation.expected_finding_contains,
                "actual": "step not found",
                "detail": f"step '{expectation.investigation_step}' not in output",
            })

    # ── Evidence flags ──
    if expectation.expected_evidence_flags:
        inv_output = get_output(expectation.investigation_step)
        if inv_output:
            actual_flags = [f.lower() for f in inv_output.get("evidence_flags", [])]
            found = [f for f in expectation.expected_evidence_flags if f.lower() in actual_flags]
            missing = [f for f in expectation.expected_evidence_flags if f.lower() not in actual_flags]
            checks.append({
                "name": "evidence_flags",
                "passed": len(found) >= len(expectation.expected_evidence_flags) * 0.5,  # 50% threshold
                "expected": expectation.expected_evidence_flags,
                "actual": actual_flags,
                "detail": f"found {len(found)}/{len(expectation.expected_evidence_flags)}: {found}",
            })

    # ── Decision ──
    if expectation.expected_decision_contains:
        dec_output = get_output(expectation.decision_step)
        if dec_output:
            decision = str(dec_output.get("decision", "")).lower()
            thought = str(dec_output.get("thought", "")).lower()
            combined = decision + " " + thought
            expected_lower = expectation.expected_decision_contains.lower()
            checks.append({
                "name": "decision",
                "passed": expected_lower in combined,
                "expected": expectation.expected_decision_contains,
                "actual": decision[:100],
                "detail": f"looking for '{expected_lower}' in decision/thought",
            })

    # ── Confidence calibration ──
    if expectation.min_confidence is not None or expectation.max_confidence is not None:
        # Check confidence on classification step
        cls_output = get_output(expectation.classification_step)
        if cls_output:
            conf = cls_output.get("confidence", 0.0)
            in_range = True
            if expectation.min_confidence is not None and conf < expectation.min_confidence:
                in_range = False
            if expectation.max_confidence is not None and conf > expectation.max_confidence:
                in_range = False
            checks.append({
                "name": "confidence_range",
                "passed": in_range,
                "expected": f"[{expectation.min_confidence}, {expectation.max_confidence}]",
                "actual": conf,
                "detail": f"confidence {conf} {'in' if in_range else 'OUT OF'} range [{expectation.min_confidence}, {expectation.max_confidence}]",
            })

    # ── Fail-closed: escalation on ambiguous cases ──
    if expectation.should_escalate:
        # Check if the workflow reached an escalation step or if
        # the final routing decision was to escalate
        routing_log = final_state.get("routing_log", [])
        escalated = any(
            rd.get("to_step", "").startswith("escalate")
            for rd in routing_log
        )
        # Also check if the last step is an escalation step
        if steps:
            last_step = steps[-1].get("step_name", "")
            if "escalate" in last_step:
                escalated = True
        # Also check if think step decided to escalate
        dec_output = get_output(expectation.decision_step)
        if dec_output:
            decision = str(dec_output.get("decision", "")).lower()
            if "escalat" in decision:
                escalated = True

        checks.append({
            "name": "fail_closed",
            "passed": escalated,
            "expected": "should escalate",
            "actual": "escalated" if escalated else "did not escalate",
            "detail": "ambiguous/adversarial case must route to HITL",
        })

    # ── Generate compliance ──
    # Find the last generate step
    gen_output = None
    for s in reversed(steps):
        if s.get("primitive") == "generate":
            gen_output = s.get("output", {})
            break

    if gen_output:
        artifact = str(gen_output.get("artifact", "")).lower()

        for phrase in expectation.must_contain:
            checks.append({
                "name": "must_contain",
                "passed": phrase.lower() in artifact,
                "expected": f"contains '{phrase}'",
                "actual": f"{'found' if phrase.lower() in artifact else 'NOT FOUND'}",
                "detail": f"looking for '{phrase}' in generated output",
            })

        for phrase in expectation.must_not_contain:
            checks.append({
                "name": "must_not_contain",
                "passed": phrase.lower() not in artifact,
                "expected": f"does NOT contain '{phrase}'",
                "actual": f"{'NOT found (good)' if phrase.lower() not in artifact else 'FOUND (bad)'}",
                "detail": f"checking '{phrase}' not in generated output",
            })

    all_passed = all(c["passed"] for c in checks)
    return CaseScore(
        case_id=expectation.case_id,
        category=expectation.category,
        description=expectation.description,
        passed=all_passed,
        checks=checks,
    )


# ═══════════════════════════════════════════════════════════════════
# Step auto-discovery
# ═══════════════════════════════════════════════════════════════════

# Primitives that map to eval check types
EVAL_PRIMITIVES = {
    "classify": "classification_step",
    "investigate": "investigation_step",
    "think": "decision_step",
    "generate": "generate_step",
    "verify": "verify_step",
    "retrieve": "retrieve_step",
}

def discover_step_names(
    workflow_path: str | Path,
    domain_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Read a workflow YAML and return {eval_role → step_name} mapping.

    For each primitive type relevant to eval, finds the FIRST step in
    the workflow using that primitive. This means packs don't need to
    hardcode step names — the harness discovers them from the workflow.

    Returns e.g.:
        {
            "classification_step": "classify_return_type",
            "investigation_step": "investigate_claim",
            "decision_step": "decide_resolution",
            "generate_step": "generate_response",
            "retrieve_step": "gather_return_data",
        }

    If domain_path is provided, loads three-layer merge to get the
    full step list (in case domain overrides add steps).
    """
    with open(workflow_path) as f:
        wf = yaml.safe_load(f)

    steps = wf.get("steps", [])
    mapping = {}

    for step in steps:
        primitive = step.get("primitive", "")
        role = EVAL_PRIMITIVES.get(primitive)
        if role and role not in mapping:
            mapping[role] = step["name"]

    return mapping


def resolve_step_name(
    explicit: str | None,
    pack_level: str | None,
    discovered: dict[str, str],
    role: str,
    fallback: str,
) -> str:
    """
    Resolve a step name with priority:
      1. Explicit per-case override
      2. Pack-level setting
      3. Auto-discovered from workflow
      4. Hardcoded fallback
    """
    if explicit:
        return explicit
    if pack_level:
        return pack_level
    if role in discovered:
        return discovered[role]
    return fallback


# ═══════════════════════════════════════════════════════════════════
# Governance-aware default criteria
# ═══════════════════════════════════════════════════════════════════

GOVERNANCE_CRITERIA_DEFAULTS = {
    "auto": AcceptanceCriteria(
        min_schema_valid_pct=98.0,
        min_classification_accuracy_pct=90.0,
        min_investigation_quality_pct=80.0,
        min_confidence_calibration_pct=85.0,
        min_fail_closed_pct=100.0,
        min_generate_compliance_pct=90.0,
    ),
    "spot_check": AcceptanceCriteria(
        min_schema_valid_pct=95.0,
        min_classification_accuracy_pct=85.0,
        min_investigation_quality_pct=75.0,
        min_confidence_calibration_pct=80.0,
        min_fail_closed_pct=100.0,
        min_generate_compliance_pct=85.0,
    ),
    "gate": AcceptanceCriteria(
        min_schema_valid_pct=90.0,
        min_classification_accuracy_pct=80.0,
        min_investigation_quality_pct=70.0,
        min_confidence_calibration_pct=75.0,
        min_fail_closed_pct=100.0,
        min_generate_compliance_pct=80.0,
    ),
    "hold": AcceptanceCriteria(
        min_schema_valid_pct=85.0,
        min_classification_accuracy_pct=75.0,
        min_investigation_quality_pct=60.0,
        min_confidence_calibration_pct=70.0,
        min_fail_closed_pct=100.0,
        min_generate_compliance_pct=75.0,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Pack loader
# ═══════════════════════════════════════════════════════════════════

def load_eval_pack(
    pack_path: str,
    project_root: str = ".",
) -> tuple[dict[str, Any], list[CaseExpectation], AcceptanceCriteria]:
    """
    Load an eval pack YAML. Returns (pack_config, expectations, criteria).

    Step name resolution priority:
      1. Per-case override (case.classification_step)
      2. Pack-level setting (pack.classification_step)
      3. Auto-discovered from workflow YAML (first step using that primitive)
      4. Hardcoded fallback

    Acceptance criteria resolution:
      1. Explicit in pack YAML (acceptance_criteria section)
      2. Governance-tier defaults from domain YAML
      3. Generic defaults
    """
    with open(pack_path) as f:
        pack = yaml.safe_load(f)

    root = Path(project_root)
    workflow = pack["workflow"]
    domain = pack["domain"]

    # ── Auto-discover step names from workflow ──
    wf_path = root / "workflows" / f"{workflow}.yaml"
    discovered = {}
    if wf_path.exists():
        discovered = discover_step_names(wf_path)

    pack_cls_step = pack.get("classification_step")
    pack_inv_step = pack.get("investigation_step")
    pack_dec_step = pack.get("decision_step")

    # ── Resolve governance tier for default criteria ──
    dom_path = root / "domains" / f"{domain}.yaml"
    governance_tier = None
    if dom_path.exists():
        with open(dom_path) as f:
            dom = yaml.safe_load(f)
        governance_tier = dom.get("governance")

    # ── Build acceptance criteria ──
    crit_cfg = pack.get("acceptance_criteria", {})
    if crit_cfg:
        # Explicit criteria in pack
        criteria = AcceptanceCriteria(
            min_schema_valid_pct=crit_cfg.get("min_schema_valid_pct", 95.0),
            min_classification_accuracy_pct=crit_cfg.get("min_classification_accuracy_pct", 80.0),
            min_investigation_quality_pct=crit_cfg.get("min_investigation_quality_pct", 70.0),
            min_confidence_calibration_pct=crit_cfg.get("min_confidence_calibration_pct", 75.0),
            max_unsupported_claims_pct=crit_cfg.get("max_unsupported_claims_pct", 15.0),
            min_fail_closed_pct=crit_cfg.get("min_fail_closed_pct", 100.0),
            min_generate_compliance_pct=crit_cfg.get("min_generate_compliance_pct", 85.0),
        )
    elif governance_tier and governance_tier in GOVERNANCE_CRITERIA_DEFAULTS:
        criteria = GOVERNANCE_CRITERIA_DEFAULTS[governance_tier]
    else:
        criteria = AcceptanceCriteria()

    # ── Build expectations ──
    expectations = []
    for case in pack.get("cases", []):
        exp = CaseExpectation(
            case_id=case["id"],
            case_file=case["file"],
            category=case.get("category", "normal"),
            description=case.get("description", ""),
            expected_classification=case.get("expected_classification"),
            classification_step=resolve_step_name(
                case.get("classification_step"), pack_cls_step,
                discovered, "classification_step", "classify"),
            expected_finding_contains=case.get("expected_finding_contains", []),
            expected_evidence_flags=case.get("expected_evidence_flags", []),
            investigation_step=resolve_step_name(
                case.get("investigation_step"), pack_inv_step,
                discovered, "investigation_step", "investigate"),
            expected_decision_contains=case.get("expected_decision_contains"),
            decision_step=resolve_step_name(
                case.get("decision_step"), pack_dec_step,
                discovered, "decision_step", "decide"),
            min_confidence=case.get("min_confidence"),
            max_confidence=case.get("max_confidence"),
            should_escalate=case.get("should_escalate", False),
            must_not_contain=case.get("must_not_contain", []),
            must_contain=case.get("must_contain", []),
        )
        expectations.append(exp)

    return pack, expectations, criteria


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def run_eval_pack(
    pack_path: str,
    project_root: str = ".",
    model: str = "default",
    dry_run: bool = False,
) -> EvalResult:
    """
    Run all cases in an eval pack and score them.

    If dry_run=True, loads cases and scores against pre-recorded outputs
    (for testing the eval framework itself without LLM calls).
    """
    pack, expectations, criteria = load_eval_pack(pack_path, project_root)
    pack_name = pack.get("name", Path(pack_path).stem)
    workflow = pack["workflow"]
    domain = pack["domain"]

    root = Path(project_root)
    scores = []
    t0 = time.time()

    total = len(expectations)
    print(f"\n  EVAL: {pack_name} — {total} cases ({workflow}/{domain})", file=sys.stderr, flush=True)
    print(f"  {'─' * 56}", file=sys.stderr, flush=True)
    for idx, exp in enumerate(expectations, 1):
        case_path = root / exp.case_file
        tag = f"[{idx}/{total}]"

        if not case_path.exists():
            print(f"  {tag} ✗ {exp.case_id} — file not found", file=sys.stderr, flush=True)
            scores.append(CaseScore(
                case_id=exp.case_id,
                category=exp.category,
                description=exp.description,
                passed=False,
                checks=[],
                error=f"Case file not found: {exp.case_file}",
            ))
            continue

        if dry_run:
            # Look for pre-recorded output
            output_path = case_path.with_suffix(".output.json")
            if output_path.exists():
                with open(output_path) as f:
                    final_state = json.load(f)
                score = score_case(exp, final_state)
                scores.append(score)
                print(f"  {tag} {'✓' if score.passed else '✗'} {exp.case_id} (dry-run)", file=sys.stderr, flush=True)
            else:
                scores.append(CaseScore(
                    case_id=exp.case_id,
                    category=exp.category,
                    description=exp.description,
                    passed=False,
                    checks=[],
                    error=f"Dry run: no pre-recorded output at {output_path}",
                ))
                print(f"  {tag} ✗ {exp.case_id} — no recorded output", file=sys.stderr, flush=True)
            continue

        # Live run
        print(f"  {tag} ▶ {exp.case_id} [{exp.category}] ...", end="", file=sys.stderr, flush=True)
        tc = time.time()
        try:
            # Import here to avoid langgraph dependency when just loading packs
            from engine.composer import load_three_layer, run_workflow
            from engine.tools import create_case_registry

            wf_path = root / "workflows" / f"{workflow}.yaml"
            dom_path = root / "domains" / f"{domain}.yaml"

            with open(case_path) as f:
                case_input = json.load(f)

            # Build tool registry from case data (get_* keys become tools)
            tool_registry = create_case_registry(case_input)

            merged, _ = load_three_layer(wf_path, dom_path, case_path)
            final_state = run_workflow(merged, case_input, model=model,
                                       tool_registry=tool_registry)

            score = score_case(exp, final_state)
            score.elapsed_seconds = time.time() - tc
            scores.append(score)

            icon = "✓" if score.passed else "✗"
            print(f" {icon} {score.pass_count}/{score.check_count} checks ({score.elapsed_seconds:.1f}s)", file=sys.stderr, flush=True)

        except Exception as e:
            elapsed = time.time() - tc
            scores.append(CaseScore(
                case_id=exp.case_id,
                category=exp.category,
                description=exp.description,
                passed=False,
                checks=[],
                elapsed_seconds=elapsed,
                error=str(e)[:500],
            ))
            print(f" ERROR ({elapsed:.1f}s): {str(e)[:80]}", file=sys.stderr, flush=True)

    result = EvalResult(
        pack_name=pack_name,
        workflow=workflow,
        domain=domain,
        cases=scores,
        criteria=criteria,
        elapsed_seconds=time.time() - t0,
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive Core Eval Runner")
    parser.add_argument("--pack", required=True, help="Path to eval pack YAML")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--model", default="default", help="Model alias")
    parser.add_argument("--dry-run", action="store_true", help="Score pre-recorded outputs only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all check details")
    parser.add_argument("--discover", action="store_true", help="Show auto-discovered step mapping and exit")
    args = parser.parse_args()

    if args.discover:
        with open(args.pack) as f:
            pack = yaml.safe_load(f)
        root = Path(args.root)
        wf_path = root / "workflows" / f"{pack['workflow']}.yaml"
        dom_path = root / "domains" / f"{pack['domain']}.yaml"

        print(f"\n  Pack: {pack.get('name', args.pack)}")
        print(f"  Workflow: {pack['workflow']}")
        print(f"  Domain: {pack['domain']}")

        if wf_path.exists():
            mapping = discover_step_names(wf_path)
            print(f"\n  Auto-discovered step mapping:")
            for role, step in mapping.items():
                print(f"    {role:25s} → {step}")
        else:
            print(f"\n  ⚠ Workflow not found: {wf_path}")

        if dom_path.exists():
            with open(dom_path) as f:
                dom = yaml.safe_load(f)
            tier = dom.get("governance", "unknown")
            print(f"\n  Governance tier: {tier}")
            if tier in GOVERNANCE_CRITERIA_DEFAULTS:
                c = GOVERNANCE_CRITERIA_DEFAULTS[tier]
                print(f"  Default criteria for '{tier}' tier:")
                print(f"    schema_valid:      ≥{c.min_schema_valid_pct}%")
                print(f"    classification:    ≥{c.min_classification_accuracy_pct}%")
                print(f"    investigation:     ≥{c.min_investigation_quality_pct}%")
                print(f"    confidence:        ≥{c.min_confidence_calibration_pct}%")
                print(f"    fail_closed:       ≥{c.min_fail_closed_pct}%")
                print(f"    generate:          ≥{c.min_generate_compliance_pct}%")

        # Show what the pack overrides
        crit_cfg = pack.get("acceptance_criteria", {})
        if crit_cfg:
            print(f"\n  Pack overrides: {list(crit_cfg.keys())}")
        else:
            print(f"\n  Pack uses governance-tier defaults (no overrides)")

        print()
        return

    result = run_eval_pack(args.pack, args.root, args.model, args.dry_run)
    print(result.summary(verbose=args.verbose))

    if not result.all_gates_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
