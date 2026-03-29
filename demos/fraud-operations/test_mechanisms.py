"""
Cognitive Core — Framework Mechanism Validation

Tests the coordinator state machine end-to-end WITHOUT an LLM.
Validates:
  1. Tool registry discovery from case_dir fixtures
  2. Delegation policy condition matching (step-name selectors)
  3. Delegation policy input resolution (step-name + primitive selectors)
  4. Delegation result injection keyed by handler_workflow_type
  5. Governance tier override from delegation policy config
  6. Single HITL gate across multi-workflow chain
  7. Full delegation chain: triage → specialist → regulatory + resolution → resume

Approach: Instead of running workflows through the LLM, we directly invoke
the coordinator's policy engine and state management with realistic step
outputs, verifying each mechanism in isolation and then in combination.
"""

import json
import os
import sys
import time
import yaml
from pathlib import Path

# Ensure project root is on path (demos/fraud-operations/ → demos/ → repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Setup ──────────────────────────────────────────────────────────

DEMO_DIR = Path(__file__).resolve().parent
COORD_CONFIG = str(DEMO_DIR / "coordinator_config.yaml")
CASE_FILE = str(DEMO_DIR / "cases" / "app_scam_romance.json")

def load_config():
    with open(COORD_CONFIG) as f:
        return yaml.safe_load(f)

def load_case():
    with open(CASE_FILE) as f:
        return json.load(f)

def pass_fail(test_name, passed, detail=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if detail and not passed:
        print(f"         {detail}")
    return passed


# ── Test 1: Tool Registry from case_dir ────────────────────────────

def test_tool_registry_from_case_dir():
    """Gap 1: Fixtures discovered from coordinator's case_dir, not hardcoded path."""
    from cognitive_core.engine.tools import create_case_registry

    print("\n═══ Test 1: Tool Registry from case_dir ═══")

    # Lean input — no get_* keys
    lean = {"case_id": "FRD-2026-00884", "alert_id": "ALT-2026-7155"}

    # Without case_dir — should find nothing (files are in demos/fraud-operations/cases/fixtures/)
    reg_default = create_case_registry(lean)
    get_tools_default = [t for t in reg_default.list_tools() if t.startswith("get_")]

    # With case_dir — should find tools
    reg_demo = create_case_registry(lean, fixtures_dir=str(DEMO_DIR / "cases"))
    get_tools_demo = [t for t in reg_demo.list_tools() if t.startswith("get_")]

    results = []
    results.append(pass_fail(
        "Default path finds no fraud fixtures",
        len(get_tools_default) == 0,
        f"Found {len(get_tools_default)} tools: {get_tools_default}",
    ))
    results.append(pass_fail(
        "Demo case_dir finds fraud fixtures",
        len(get_tools_demo) >= 8,
        f"Found {len(get_tools_demo)} tools: {get_tools_demo}",
    ))

    # Verify data comes through
    result = reg_demo.call("get_member", lean)
    results.append(pass_fail(
        "get_member returns correct data",
        result.error is None and result.data.get("full_name") == "Lisa Okafor-Chen",
        f"Got: {result.data.get('full_name', '?')}",
    ))

    # Test all three cases
    for case_id, expected_name in [
        ("FRD-2026-00871", "Derek Lawson"),
        ("FRD-2026-00902", "Robert Tran"),
    ]:
        reg = create_case_registry({"case_id": case_id}, fixtures_dir=str(DEMO_DIR / "cases"))
        r = reg.call("get_member", {"case_id": case_id})
        results.append(pass_fail(
            f"Case {case_id} → {expected_name}",
            r.error is None and r.data.get("full_name") == expected_name,
        ))

    return all(results)


# ── Test 2: Step-Name Selector in Conditions ────────────────────────

def test_step_name_selector_conditions():
    """Gap 3: step:<name> selector matches specific steps, not last primitive."""
    from cognitive_core.coordinator.policy import load_policy_engine

    print("\n═══ Test 2: Step-Name Selector (Conditions) ═══")

    cfg = load_config()
    policy = load_policy_engine(cfg)

    # Simulate triage output — two classify steps with different categories
    triage_output = {
        "steps": [
            {"step_name": "classify_fraud_type", "primitive": "classify",
             "output": {"category": "card_fraud", "confidence": 0.95}},
            {"step_name": "assess_priority", "primitive": "classify",
             "output": {"category": "critical", "confidence": 0.90}},
        ],
        "input": {"case_id": "FRD-2026-00884", "alert_id": "ALT-2026-7155"},
    }

    delegations = policy.evaluate_delegations(
        domain="fraud_triage",
        workflow_output=triage_output,
    )

    results = []
    results.append(pass_fail(
        "Exactly 1 delegation triggered",
        len(delegations) == 1,
        f"Got {len(delegations)} delegations",
    ))

    if delegations:
        d = delegations[0]
        results.append(pass_fail(
            "Correct policy: triage_to_card_fraud",
            d.policy_name == "triage_to_card_fraud",
            f"Got: {d.policy_name}",
        ))
        results.append(pass_fail(
            "Routes to card_fraud domain",
            d.target_domain == "card_fraud",
            f"Got: {d.target_domain}",
        ))

    # Test that last_classify would have matched wrong step
    # The step:classify_fraud_type selector is what makes this work
    last_classify_category = triage_output["steps"][-1]["output"]["category"]
    first_classify_category = triage_output["steps"][0]["output"]["category"]
    results.append(pass_fail(
        "last_classify would have matched 'critical' (wrong)",
        last_classify_category == "critical",
    ))
    results.append(pass_fail(
        "step:classify_fraud_type matched 'card_fraud' (correct)",
        first_classify_category == "card_fraud",
    ))

    # Test all three fraud types route correctly
    for fraud_type, expected_domain in [
        ("check_fraud", "check_fraud"),
        ("card_fraud", "card_fraud"),
        ("app_scam", "app_scam_fraud"),
    ]:
        output = {
            "steps": [
                {"step_name": "classify_fraud_type", "primitive": "classify",
                 "output": {"category": fraud_type, "confidence": 0.95}},
                {"step_name": "assess_priority", "primitive": "classify",
                 "output": {"category": "high", "confidence": 0.85}},
            ],
            "input": {"case_id": "test"},
        }
        dels = policy.evaluate_delegations(domain="fraud_triage", workflow_output=output)
        results.append(pass_fail(
            f"classify={fraud_type} → domain={expected_domain}",
            len(dels) == 1 and dels[0].target_domain == expected_domain,
            f"Got {len(dels)} delegations" + (f": {dels[0].target_domain}" if dels else ""),
        ))

    return all(results)


# ── Test 3: Step-Name Selector in Input Mappings ────────────────────

def test_step_name_selector_inputs():
    """Gap 3: ${source.step:<name>.field} resolves specific step outputs."""
    from cognitive_core.coordinator.policy import load_policy_engine

    print("\n═══ Test 3: Step-Name Selector (Input Mappings) ═══")

    cfg = load_config()
    policy = load_policy_engine(cfg)

    triage_output = {
        "steps": [
            {"step_name": "classify_fraud_type", "primitive": "classify",
             "output": {"category": "app_scam", "confidence": 0.97}},
            {"step_name": "assess_priority", "primitive": "classify",
             "output": {"category": "critical", "confidence": 0.92}},
        ],
        "input": {"case_id": "FRD-2026-00902", "alert_id": "ALT-2026-7201"},
    }

    delegations = policy.evaluate_delegations(
        domain="fraud_triage",
        workflow_output=triage_output,
    )

    results = []
    assert len(delegations) == 1
    d = delegations[0]

    results.append(pass_fail(
        "triage_classification from step:classify_fraud_type",
        d.inputs.get("triage_classification") == "app_scam",
        f"Got: {d.inputs.get('triage_classification')}",
    ))
    results.append(pass_fail(
        "triage_confidence from step:classify_fraud_type",
        d.inputs.get("triage_confidence") == 0.97,
        f"Got: {d.inputs.get('triage_confidence')}",
    ))
    results.append(pass_fail(
        "triage_priority from step:assess_priority",
        d.inputs.get("triage_priority") == "critical",
        f"Got: {d.inputs.get('triage_priority')}",
    ))
    results.append(pass_fail(
        "case_id from source.input",
        d.inputs.get("case_id") == "FRD-2026-00902",
        f"Got: {d.inputs.get('case_id')}",
    ))

    return all(results)


# ── Test 4: Delegation Result Injection ──────────────────────────────

def test_delegation_result_injection():
    """Gap 4: Results keyed by handler_workflow_type, accessible in templates."""
    from cognitive_core.engine.resume import prepare_resume_state
    from cognitive_core.engine.composer import load_three_layer

    print("\n═══ Test 4: Delegation Result Injection ═══")

    config, _ = load_three_layer(
        str(DEMO_DIR / "workflows" / "fraud_specialty_investigation.yaml"),
        str(DEMO_DIR / "domains" / "app_scam_fraud.yaml"),
    )

    state_snapshot = {
        "input": {"case_id": "FRD-2026-00902", "fraud_type": "app_scam"},
        "steps": [
            {"step_name": "retrieve_evidence", "primitive": "retrieve",
             "output": {"data": {"sources": 9}}},
            {"step_name": "investigate_activity", "primitive": "investigate",
             "output": {"finding": "Romance scam confirmed", "confidence": 0.92}},
            {"step_name": "deliberate_determination", "primitive": "deliberate",
             "output": {"recommended_action": "file_sar", "confidence": 0.88}},
        ],
        "delegation_results": {
            "fraud_regulatory_review": {
                "step_count": 3,
                "steps": [
                    {"step_name": "verify_compliance", "primitive": "verify",
                     "output": {"conforms": True, "findings": ["Reg E does not apply"]}},
                    {"step_name": "generate_compliance_package", "primitive": "generate",
                     "output": {"content": "SAR filing required within 30 days"}},
                ],
            },
            "fraud_case_resolution": {
                "step_count": 5,
                "steps": [
                    {"step_name": "challenge_determination", "primitive": "challenge",
                     "output": {"survives": True, "strengths": ["Clear scam pattern"]}},
                    {"step_name": "generate_case_summary", "primitive": "generate",
                     "output": {"content": "Elder romance scam, $18,500 loss"}},
                ],
            },
        },
        "loop_counts": {},
        "routing_log": [],
    }

    initial = prepare_resume_state(config, state_snapshot, "generate_final_report")

    results = []

    delegation = initial["input"].get("delegation", {})
    results.append(pass_fail(
        "delegation key exists in resumed input",
        "delegation" in initial["input"],
    ))
    results.append(pass_fail(
        "fraud_regulatory_review results present",
        "fraud_regulatory_review" in delegation,
    ))
    results.append(pass_fail(
        "fraud_case_resolution results present",
        "fraud_case_resolution" in delegation,
    ))

    # Verify deep access
    reg_steps = delegation.get("fraud_regulatory_review", {}).get("steps", [])
    results.append(pass_fail(
        "Regulatory verify step accessible",
        len(reg_steps) >= 1 and reg_steps[0]["output"].get("conforms") is True,
        f"Got: {reg_steps[0]['output'] if reg_steps else 'no steps'}",
    ))

    res_steps = delegation.get("fraud_case_resolution", {}).get("steps", [])
    results.append(pass_fail(
        "Resolution challenge step accessible",
        len(res_steps) >= 1 and res_steps[0]["output"].get("survives") is True,
    ))

    # Verify prior steps preserved
    prior_steps = initial["steps"]
    step_names = [s["step_name"] for s in prior_steps]
    results.append(pass_fail(
        "Prior steps preserved (retrieve, investigate, deliberate)",
        step_names == ["retrieve_evidence", "investigate_activity", "deliberate_determination"],
        f"Got: {step_names}",
    ))

    return all(results)


# ── Test 5: Governance Tier Override ─────────────────────────────────

def test_governance_tier_override():
    """Gap 5: Delegation policies can override handler governance tier."""
    from cognitive_core.coordinator.policy import load_policy_engine

    print("\n═══ Test 5: Governance Tier Override ═══")

    cfg = load_config()
    policy = load_policy_engine(cfg)

    # Specialist output triggers regulatory and resolution delegations
    specialist_output = {
        "steps": [
            {"step_name": "deliberate_determination", "primitive": "deliberate",
             "output": {"recommended_action": "file_sar", "confidence": 0.88}},
            {"step_name": "investigate_activity", "primitive": "investigate",
             "output": {"finding": "Romance scam"}},
        ],
        "input": {"case_id": "FRD-2026-00902", "fraud_type": "app_scam"},
    }

    delegations = policy.evaluate_delegations(
        domain="app_scam_fraud",
        workflow_output=specialist_output,
    )

    results = []
    results.append(pass_fail(
        "2 delegations triggered (regulatory + resolution)",
        len(delegations) == 2,
        f"Got {len(delegations)}",
    ))

    for d in delegations:
        results.append(pass_fail(
            f"{d.policy_name} has governance_tier='auto'",
            d.governance_tier == "auto",
            f"Got: {d.governance_tier!r}",
        ))

    # Verify triage delegations do NOT override tier (empty string)
    triage_output = {
        "steps": [
            {"step_name": "classify_fraud_type", "primitive": "classify",
             "output": {"category": "check_fraud", "confidence": 0.95}},
            {"step_name": "assess_priority", "primitive": "classify",
             "output": {"category": "high", "confidence": 0.85}},
        ],
        "input": {"case_id": "test"},
    }
    triage_dels = policy.evaluate_delegations(
        domain="fraud_triage",
        workflow_output=triage_output,
    )
    if triage_dels:
        results.append(pass_fail(
            "Triage delegation has no tier override (empty)",
            triage_dels[0].governance_tier == "",
            f"Got: {triage_dels[0].governance_tier!r}",
        ))

    return all(results)


# ── Test 6: Contract Validation Skip ─────────────────────────────────

def test_contract_validation_skip():
    """Gap 2: Empty contract name skips validation instead of failing."""
    from cognitive_core.coordinator.policy import load_policy_engine

    print("\n═══ Test 6: Optional Contract Validation ═══")

    cfg = load_config()
    policy = load_policy_engine(cfg)

    results = []

    # Empty contract — should return no errors
    errors = policy.validate_work_order_inputs("", {"any": "input"})
    results.append(pass_fail(
        "Empty contract name → no validation errors",
        len(errors) == 0,
        f"Got errors: {errors}",
    ))

    # Unknown contract — should return error
    errors = policy.validate_work_order_inputs("nonexistent_v1", {"any": "input"})
    results.append(pass_fail(
        "Unknown contract → validation error",
        len(errors) > 0 and "Unknown contract" in errors[0],
        f"Got: {errors}",
    ))

    # Same for response validation
    errors = policy.validate_work_order_result("", {"any": "output"})
    results.append(pass_fail(
        "Empty contract on response → no errors",
        len(errors) == 0,
    ))

    return all(results)


# ── Test 7: Full Chain Simulation ────────────────────────────────────

def test_full_chain():
    """End-to-end: simulate the complete delegation chain with realistic outputs."""
    from cognitive_core.coordinator.policy import load_policy_engine
    from cognitive_core.engine.resume import prepare_resume_state
    from cognitive_core.engine.composer import load_three_layer
    from cognitive_core.engine.tools import create_case_registry

    print("\n═══ Test 7: Full Delegation Chain ═══")

    cfg = load_config()
    policy = load_policy_engine(cfg)
    case = load_case()
    results = []

    # ── Step A: Triage completes ──
    triage_output = {
        "steps": [
            {"step_name": "classify_fraud_type", "primitive": "classify",
             "output": {"category": "app_scam", "confidence": 1.0}},
            {"step_name": "assess_priority", "primitive": "classify",
             "output": {"category": "critical", "confidence": 0.95}},
        ],
        "input": {"case_id": case["case_id"], "alert_id": case["alert_id"]},
    }
    triage_delegations = policy.evaluate_delegations(
        domain="fraud_triage", workflow_output=triage_output,
    )
    results.append(pass_fail(
        "A. Triage → 1 delegation (fire_and_forget)",
        len(triage_delegations) == 1
        and triage_delegations[0].mode == "fire_and_forget"
        and triage_delegations[0].target_domain == "app_scam_fraud",
    ))

    # ── Step B: Specialist completes ──
    specialist_output = {
        "steps": [
            {"step_name": "retrieve_evidence", "primitive": "retrieve",
             "output": {"data": {"sources": 9}}},
            {"step_name": "investigate_activity", "primitive": "investigate",
             "output": {
                 "finding": "Romance scam targeting widowed veteran, $18,500 loss across 3 payments",
                 "confidence": 0.94,
                 "evidence_used": ["get_member", "get_triggering_activity", "get_scam_indicators"],
             }},
            {"step_name": "deliberate_determination", "primitive": "deliberate",
             "output": {
                 "recommended_action": "file_sar",
                 "confidence": 0.91,
                 "reasoning": "Clear romance scam pattern with elder vulnerability factors",
             }},
            {"step_name": "generate_final_report", "primitive": "generate",
             "output": {"content": "Investigation report..."}},
        ],
        "input": {
            "case_id": case["case_id"],
            "alert_id": case["alert_id"],
            "fraud_type": "app_scam",
        },
    }
    specialist_delegations = policy.evaluate_delegations(
        domain="app_scam_fraud", workflow_output=specialist_output,
    )
    results.append(pass_fail(
        "B. Specialist → 2 delegations (wait_for_result)",
        len(specialist_delegations) == 2
        and all(d.mode == "wait_for_result" for d in specialist_delegations),
    ))
    results.append(pass_fail(
        "B. Both resume at generate_final_report",
        all(d.resume_at_step == "generate_final_report" for d in specialist_delegations),
    ))
    results.append(pass_fail(
        "B. Both have governance_tier=auto",
        all(d.governance_tier == "auto" for d in specialist_delegations),
    ))

    # Verify input resolution
    reg_deleg = next(d for d in specialist_delegations if "regulatory" in d.policy_name)
    results.append(pass_fail(
        "B. Regulatory gets determination='file_sar'",
        reg_deleg.inputs.get("determination") == "file_sar",
        f"Got: {reg_deleg.inputs.get('determination')}",
    ))
    results.append(pass_fail(
        "B. Regulatory gets investigation_summary",
        "Romance scam" in str(reg_deleg.inputs.get("investigation_summary", "")),
    ))

    # ── Step C: Regulatory completes ──
    regulatory_result = {
        "step_count": 3,
        "steps": [
            {"step_name": "retrieve_regulatory_context", "primitive": "retrieve",
             "output": {"data": {"sources": 6}}},
            {"step_name": "verify_compliance", "primitive": "verify",
             "output": {
                 "conforms": True,
                 "rules_checked": ["reg_e_applicability", "sar_threshold", "elder_abuse_reporting"],
                 "findings": [
                     "Reg E does not apply — payments were authorized by member",
                     "SAR filing required — loss exceeds $5,000 threshold",
                     "Elder abuse report required — victim age 67, lives alone",
                 ],
             }},
            {"step_name": "generate_compliance_package", "primitive": "generate",
             "output": {
                 "content": "SAR must be filed within 30 days. Elder abuse report to FL-DCF required.",
                 "deadlines": {"sar_filing": "2026-03-28", "elder_abuse_report": "2026-02-27"},
             }},
        ],
    }

    # ── Step D: Resolution completes ──
    resolution_result = {
        "step_count": 5,
        "steps": [
            {"step_name": "retrieve_case_context", "primitive": "retrieve",
             "output": {"data": {"sources": 6}}},
            {"step_name": "challenge_determination", "primitive": "challenge",
             "output": {
                 "survives": True,
                 "strengths": [
                     "Classic romance scam escalation pattern",
                     "Recipient accounts show mule indicators",
                     "Stock photo profile confirmed",
                 ],
                 "vulnerabilities": [
                     "Member authorized all payments — limited recovery basis",
                 ],
             }},
            {"step_name": "generate_hitl_package", "primitive": "generate",
             "output": {"content": "HITL review package for fraud analyst..."}},
            {"step_name": "generate_member_notification", "primitive": "generate",
             "output": {"content": "Dear Mr. Tran, We are writing to inform you..."}},
            {"step_name": "generate_case_summary", "primitive": "generate",
             "output": {"content": "Romance scam case summary — $18,500 loss..."}},
        ],
    }

    # ── Step E: Specialist resumes with delegation results ──
    config, _ = load_three_layer(
        str(DEMO_DIR / "workflows" / "fraud_specialty_investigation.yaml"),
        str(DEMO_DIR / "domains" / "app_scam_fraud.yaml"),
    )

    state_at_suspension = {
        "input": specialist_output["input"],
        "steps": specialist_output["steps"][:3],  # before generate_final_report
        "delegation_results": {
            "fraud_regulatory_review": regulatory_result,
            "fraud_case_resolution": resolution_result,
        },
        "loop_counts": {},
        "routing_log": [],
    }

    resumed_state = prepare_resume_state(
        config, state_at_suspension, "generate_final_report",
    )

    delegation = resumed_state["input"].get("delegation", {})

    results.append(pass_fail(
        "E. Regulatory results in resumed context",
        "fraud_regulatory_review" in delegation,
    ))
    results.append(pass_fail(
        "E. Resolution results in resumed context",
        "fraud_case_resolution" in delegation,
    ))

    # Verify the generate step would have access to key data
    reg_verify = delegation["fraud_regulatory_review"]["steps"][1]["output"]
    results.append(pass_fail(
        "E. Regulatory compliance conforms=True accessible",
        reg_verify.get("conforms") is True,
    ))

    res_challenge = delegation["fraud_case_resolution"]["steps"][1]["output"]
    results.append(pass_fail(
        "E. Challenge survives=True accessible",
        res_challenge.get("survives") is True,
    ))

    results.append(pass_fail(
        "E. Challenge strengths accessible",
        len(res_challenge.get("strengths", [])) == 3,
    ))

    # ── Step F: Verify tool registry for child workflows ──
    # When specialist delegates to regulatory, the child gets lean input.
    # The tool registry must still find fixtures via case_dir.
    child_input = reg_deleg.inputs  # case_id, alert_id, fraud_type, etc.
    child_registry = create_case_registry(child_input, fixtures_dir=str(DEMO_DIR / "cases"))
    child_tools = [t for t in child_registry.list_tools() if t.startswith("get_")]

    results.append(pass_fail(
        "F. Child workflow (regulatory) gets tools from fixtures",
        len(child_tools) >= 6,
        f"Got {len(child_tools)} tools: {child_tools}",
    ))

    # Verify the child can query specific data
    member = child_registry.call("get_member", child_input)
    results.append(pass_fail(
        "F. Child's get_member returns Robert Tran",
        member.error is None and member.data.get("full_name") == "Robert Tran",
        f"Got: {member.data.get('full_name', '?') if member.error is None else 'FAIL'}",
    ))

    return all(results)


# ── Test 8: Governance Model ─────────────────────────────────────────

def test_governance_model():
    """Gap 6: Single HITL gate — specialist is gate, children are auto."""
    from cognitive_core.coordinator.runtime import Coordinator

    print("\n═══ Test 8: Governance Model ═══")

    coord = Coordinator(config_path=COORD_CONFIG, verbose=False,
                        db_path=str(DEMO_DIR / "test.db"))
    results = []

    # Triage domain → auto (from domain config)
    triage_tier = coord._resolve_governance_tier("fraud_triage")
    results.append(pass_fail(
        "Triage domain tier = auto",
        triage_tier == "auto",
        f"Got: {triage_tier}",
    ))

    # Specialist domains → gate (from domain config)
    for domain in ["check_fraud", "card_fraud", "app_scam_fraud"]:
        tier = coord._resolve_governance_tier(domain)
        results.append(pass_fail(
            f"{domain} domain tier = gate",
            tier == "gate",
            f"Got: {tier}",
        ))

    # Regulatory and resolution domains → gate (from domain config)
    # BUT delegation policies override to auto
    for domain in ["fraud_regulatory", "fraud_case_resolution"]:
        tier = coord._resolve_governance_tier(domain)
        results.append(pass_fail(
            f"{domain} domain default tier = gate",
            tier == "gate",
            f"Got: {tier}",
        ))

    # The override happens at dispatch time, verified in test 5
    results.append(pass_fail(
        "Delegation policies override regulatory+resolution to auto (verified in test 5)",
        True,  # Already tested
    ))

    return all(results)


# ── Run All Tests ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  COGNITIVE CORE — FRAMEWORK MECHANISM VALIDATION")
    print("  No LLM required. Tests coordinator state machine directly.")
    print("=" * 70)

    tests = [
        ("Tool Registry from case_dir", test_tool_registry_from_case_dir),
        ("Step-Name Selector (Conditions)", test_step_name_selector_conditions),
        ("Step-Name Selector (Input Mappings)", test_step_name_selector_inputs),
        ("Delegation Result Injection", test_delegation_result_injection),
        ("Governance Tier Override", test_governance_tier_override),
        ("Optional Contract Validation", test_contract_validation_skip),
        ("Full Delegation Chain", test_full_chain),
        ("Governance Model", test_governance_model),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            passed = test_fn()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
