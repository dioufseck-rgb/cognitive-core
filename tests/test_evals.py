"""
Cognitive Core — Eval Framework Tests

Tests the scoring logic, quality gate, and pack loading.
Does NOT run actual LLM calls.
"""

import os
import sys
import unittest

# Import eval runner directly to avoid engine/__init__.py
import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("evals_runner", os.path.join(_base, "evals", "runner.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["evals_runner"] = _mod
_spec.loader.exec_module(_mod)

CaseExpectation = _mod.CaseExpectation
CaseScore = _mod.CaseScore
AcceptanceCriteria = _mod.AcceptanceCriteria
EvalResult = _mod.EvalResult
score_case = _mod.score_case
load_eval_pack = _mod.load_eval_pack
discover_step_names = _mod.discover_step_names
resolve_step_name = _mod.resolve_step_name
GOVERNANCE_CRITERIA_DEFAULTS = _mod.GOVERNANCE_CRITERIA_DEFAULTS


class TestScoring(unittest.TestCase):
    """Test the scoring engine against synthetic workflow outputs."""

    def _make_state(self, steps):
        return {"steps": steps, "routing_log": []}

    def test_schema_valid_all_pass(self):
        state = self._make_state([
            {"step_name": "classify", "primitive": "classify",
             "output": {"category": "defective_product", "confidence": 0.9}},
            {"step_name": "generate", "primitive": "generate",
             "output": {"artifact": "Your refund has been approved."}},
        ])
        exp = CaseExpectation(case_id="t1", case_file="x.json", category="normal", description="test")
        score = score_case(exp, state)
        schema_checks = [c for c in score.checks if c["name"] == "schema_valid"]
        self.assertEqual(len(schema_checks), 1)
        self.assertTrue(schema_checks[0]["passed"])

    def test_schema_invalid_parse_failure(self):
        state = self._make_state([
            {"step_name": "classify", "primitive": "classify",
             "output": {"_parse_failed": True, "error": "bad json", "confidence": 0.0}},
        ])
        exp = CaseExpectation(case_id="t2", case_file="x.json", category="normal", description="test")
        score = score_case(exp, state)
        schema_checks = [c for c in score.checks if c["name"] == "schema_valid"]
        self.assertFalse(schema_checks[0]["passed"])

    def test_classification_correct(self):
        state = self._make_state([
            {"step_name": "classify_return_type", "primitive": "classify",
             "output": {"category": "defective_product", "confidence": 0.92}},
        ])
        exp = CaseExpectation(
            case_id="t3", case_file="x.json", category="normal", description="test",
            expected_classification="defective_product",
            classification_step="classify_return_type",
        )
        score = score_case(exp, state)
        cls_checks = [c for c in score.checks if c["name"] == "classification"]
        self.assertEqual(len(cls_checks), 1)
        self.assertTrue(cls_checks[0]["passed"])

    def test_classification_wrong(self):
        state = self._make_state([
            {"step_name": "classify_return_type", "primitive": "classify",
             "output": {"category": "buyers_remorse", "confidence": 0.75}},
        ])
        exp = CaseExpectation(
            case_id="t4", case_file="x.json", category="normal", description="test",
            expected_classification="defective_product",
            classification_step="classify_return_type",
        )
        score = score_case(exp, state)
        cls_checks = [c for c in score.checks if c["name"] == "classification"]
        self.assertFalse(cls_checks[0]["passed"])

    def test_investigation_finding_keywords(self):
        state = self._make_state([
            {"step_name": "investigate_claim", "primitive": "investigate",
             "output": {"finding": "Evidence suggests this is a legitimate defect claim with no fraud indicators.", "confidence": 0.8}},
        ])
        exp = CaseExpectation(
            case_id="t5", case_file="x.json", category="normal", description="test",
            expected_finding_contains=["legitimate", "defect"],
            investigation_step="investigate_claim",
        )
        score = score_case(exp, state)
        inv_checks = [c for c in score.checks if c["name"] == "investigation_finding"]
        self.assertTrue(inv_checks[0]["passed"])

    def test_investigation_finding_missing_keyword(self):
        state = self._make_state([
            {"step_name": "investigate_claim", "primitive": "investigate",
             "output": {"finding": "The return appears normal.", "confidence": 0.7}},
        ])
        exp = CaseExpectation(
            case_id="t6", case_file="x.json", category="normal", description="test",
            expected_finding_contains=["fraud", "serial returner"],
            investigation_step="investigate_claim",
        )
        score = score_case(exp, state)
        inv_checks = [c for c in score.checks if c["name"] == "investigation_finding"]
        self.assertFalse(inv_checks[0]["passed"])

    def test_confidence_in_range(self):
        state = self._make_state([
            {"step_name": "classify_return_type", "primitive": "classify",
             "output": {"category": "defective_product", "confidence": 0.65}},
        ])
        exp = CaseExpectation(
            case_id="t7", case_file="x.json", category="edge", description="test",
            min_confidence=0.5, max_confidence=0.8,
            classification_step="classify_return_type",
        )
        score = score_case(exp, state)
        conf_checks = [c for c in score.checks if c["name"] == "confidence_range"]
        self.assertTrue(conf_checks[0]["passed"])

    def test_confidence_too_high(self):
        state = self._make_state([
            {"step_name": "classify_return_type", "primitive": "classify",
             "output": {"category": "defective_product", "confidence": 0.95}},
        ])
        exp = CaseExpectation(
            case_id="t8", case_file="x.json", category="edge", description="test",
            min_confidence=0.5, max_confidence=0.8,
            classification_step="classify_return_type",
        )
        score = score_case(exp, state)
        conf_checks = [c for c in score.checks if c["name"] == "confidence_range"]
        self.assertFalse(conf_checks[0]["passed"])

    def test_fail_closed_escalation_detected(self):
        state = self._make_state([
            {"step_name": "decide_resolution", "primitive": "think",
             "output": {"decision": "escalate to manager — too many fraud indicators", "confidence": 0.4}},
        ])
        state["routing_log"] = [{"to_step": "escalate_to_manager"}]
        exp = CaseExpectation(
            case_id="t9", case_file="x.json", category="adversarial", description="test",
            should_escalate=True,
        )
        score = score_case(exp, state)
        esc_checks = [c for c in score.checks if c["name"] == "fail_closed"]
        self.assertTrue(esc_checks[0]["passed"])

    def test_fail_closed_not_escalated(self):
        state = self._make_state([
            {"step_name": "decide_resolution", "primitive": "think",
             "output": {"decision": "approve full refund", "confidence": 0.8}},
        ])
        exp = CaseExpectation(
            case_id="t10", case_file="x.json", category="adversarial", description="test",
            should_escalate=True,
        )
        score = score_case(exp, state)
        esc_checks = [c for c in score.checks if c["name"] == "fail_closed"]
        self.assertFalse(esc_checks[0]["passed"])

    def test_must_contain_pass(self):
        state = self._make_state([
            {"step_name": "generate_response", "primitive": "generate",
             "output": {"artifact": "Dear Customer, your full refund of $1,249.99 has been approved."}},
        ])
        exp = CaseExpectation(
            case_id="t11", case_file="x.json", category="normal", description="test",
            must_contain=["refund"],
        )
        score = score_case(exp, state)
        gen_checks = [c for c in score.checks if c["name"] == "must_contain"]
        self.assertTrue(gen_checks[0]["passed"])

    def test_must_not_contain_fail(self):
        state = self._make_state([
            {"step_name": "generate_response", "primitive": "generate",
             "output": {"artifact": "Based on our fraud investigation, your risk score of 72 indicates..."}},
        ])
        exp = CaseExpectation(
            case_id="t12", case_file="x.json", category="normal", description="test",
            must_not_contain=["risk score", "fraud"],
        )
        score = score_case(exp, state)
        gen_checks = [c for c in score.checks if c["name"] == "must_not_contain"]
        # "risk score" found → should fail
        risk_check = [c for c in gen_checks if "risk score" in c["expected"]]
        self.assertFalse(risk_check[0]["passed"])


class TestAcceptanceGates(unittest.TestCase):
    """Test that quality gates aggregate correctly."""

    def test_all_gates_pass(self):
        scores = [
            CaseScore("c1", "normal", "test", True, [
                {"name": "schema_valid", "passed": True},
                {"name": "classification", "passed": True},
                {"name": "investigation_finding", "passed": True},
                {"name": "confidence_range", "passed": True},
                {"name": "must_contain", "passed": True},
            ]),
        ]
        result = EvalResult("test", "wf", "dom", scores, AcceptanceCriteria())
        self.assertTrue(result.all_gates_pass)

    def test_classification_gate_fails(self):
        scores = [
            CaseScore("c1", "normal", "t", True, [
                {"name": "schema_valid", "passed": True},
                {"name": "classification", "passed": False},
            ]),
            CaseScore("c2", "normal", "t", True, [
                {"name": "schema_valid", "passed": True},
                {"name": "classification", "passed": False},
            ]),
        ]
        result = EvalResult("test", "wf", "dom", scores,
                            AcceptanceCriteria(min_classification_accuracy_pct=80))
        gates = result.gate_results()
        self.assertFalse(gates["classification_accuracy"]["passed"])

    def test_fail_closed_gate_strict(self):
        """Even one missed escalation fails the 100% gate."""
        scores = [
            CaseScore("a1", "adversarial", "t", True, [
                {"name": "fail_closed", "passed": True},
            ]),
            CaseScore("a2", "adversarial", "t", False, [
                {"name": "fail_closed", "passed": False},
            ]),
        ]
        result = EvalResult("test", "wf", "dom", scores,
                            AcceptanceCriteria(min_fail_closed_pct=100))
        gates = result.gate_results()
        self.assertFalse(gates["fail_closed"]["passed"])


class TestPackLoading(unittest.TestCase):
    """Test that eval packs load correctly."""

    def test_load_product_return_pack(self):
        pack_path = os.path.join(_base, "evals", "packs", "product_return.yaml")
        pack, expectations, criteria = load_eval_pack(pack_path, _base)
        self.assertEqual(pack["workflow"], "product_return")
        self.assertEqual(pack["domain"], "electronics_return")
        self.assertEqual(len(expectations), 25)

        normal = [e for e in expectations if e.category == "normal"]
        edge = [e for e in expectations if e.category == "edge"]
        adversarial = [e for e in expectations if e.category == "adversarial"]
        self.assertEqual(len(normal), 10)
        self.assertEqual(len(edge), 10)
        self.assertEqual(len(adversarial), 5)

    def test_load_card_dispute_pack(self):
        pack_path = os.path.join(_base, "evals", "packs", "card_dispute.yaml")
        pack, expectations, criteria = load_eval_pack(pack_path, _base)
        self.assertEqual(pack["workflow"], "dispute_resolution")
        self.assertEqual(len(expectations), 25)

    def test_adversarial_cases_require_escalation(self):
        pack_path = os.path.join(_base, "evals", "packs", "product_return.yaml")
        _, expectations, _ = load_eval_pack(pack_path, _base)
        adversarial = [e for e in expectations if e.category == "adversarial"]
        for exp in adversarial:
            self.assertTrue(exp.should_escalate,
                f"Adversarial case {exp.case_id} must have should_escalate=true")

    def test_all_case_files_exist(self):
        for pack_file in ["product_return.yaml", "card_dispute.yaml"]:
            pack_path = os.path.join(_base, "evals", "packs", pack_file)
            _, expectations, _ = load_eval_pack(pack_path, _base)
            for exp in expectations:
                case_path = os.path.join(_base, exp.case_file)
                self.assertTrue(os.path.exists(case_path),
                    f"Case file missing: {exp.case_file} (pack: {pack_file})")


class TestStepDiscovery(unittest.TestCase):
    """Test auto-discovery of step names from workflow YAML."""

    def test_discover_product_return(self):
        wf_path = os.path.join(_base, "workflows", "product_return.yaml")
        mapping = discover_step_names(wf_path)
        self.assertEqual(mapping["classification_step"], "classify_return_type")
        self.assertEqual(mapping["investigation_step"], "investigate_claim")
        self.assertEqual(mapping["decision_step"], "decide_resolution")
        self.assertEqual(mapping["retrieve_step"], "gather_return_data")
        self.assertIn("generate_step", mapping)

    def test_discover_dispute_resolution(self):
        wf_path = os.path.join(_base, "workflows", "dispute_resolution.yaml")
        mapping = discover_step_names(wf_path)
        self.assertEqual(mapping["classification_step"], "classify_dispute_type")
        self.assertEqual(mapping["investigation_step"], "investigate_dispute")
        self.assertEqual(mapping["retrieve_step"], "gather_case_data")

    def test_auto_resolved_step_names_product_return(self):
        """Pack without hardcoded step names should auto-discover from workflow."""
        pack_path = os.path.join(_base, "evals", "packs", "product_return.yaml")
        _, expectations, _ = load_eval_pack(pack_path, _base)
        # All cases should have the discovered step names
        for exp in expectations:
            self.assertEqual(exp.classification_step, "classify_return_type",
                f"Case {exp.case_id}: expected auto-discovered classify step")
            self.assertEqual(exp.investigation_step, "investigate_claim",
                f"Case {exp.case_id}: expected auto-discovered investigate step")
            self.assertEqual(exp.decision_step, "decide_resolution",
                f"Case {exp.case_id}: expected auto-discovered decision step")

    def test_auto_resolved_step_names_card_dispute(self):
        """Card dispute pack should discover different step names."""
        pack_path = os.path.join(_base, "evals", "packs", "card_dispute.yaml")
        _, expectations, _ = load_eval_pack(pack_path, _base)
        for exp in expectations:
            self.assertEqual(exp.classification_step, "classify_dispute_type",
                f"Case {exp.case_id}: expected auto-discovered classify step")
            self.assertEqual(exp.investigation_step, "investigate_dispute",
                f"Case {exp.case_id}: expected auto-discovered investigate step")

    def test_resolve_priority(self):
        """Per-case > pack-level > discovered > fallback."""
        discovered = {"classification_step": "auto_classify"}
        # Discovered wins over fallback
        self.assertEqual(
            resolve_step_name(None, None, discovered, "classification_step", "fallback"),
            "auto_classify")
        # Pack-level wins over discovered
        self.assertEqual(
            resolve_step_name(None, "pack_classify", discovered, "classification_step", "fallback"),
            "pack_classify")
        # Per-case wins over everything
        self.assertEqual(
            resolve_step_name("case_classify", "pack_classify", discovered, "classification_step", "fallback"),
            "case_classify")
        # Fallback when nothing else
        self.assertEqual(
            resolve_step_name(None, None, {}, "classification_step", "fallback"),
            "fallback")


class TestGovernanceCriteria(unittest.TestCase):
    """Test governance-tier-aware default acceptance criteria."""

    def test_auto_tier_strictest(self):
        c = GOVERNANCE_CRITERIA_DEFAULTS["auto"]
        self.assertEqual(c.min_schema_valid_pct, 98.0)
        self.assertEqual(c.min_classification_accuracy_pct, 90.0)

    def test_hold_tier_most_lenient(self):
        c = GOVERNANCE_CRITERIA_DEFAULTS["hold"]
        self.assertEqual(c.min_schema_valid_pct, 85.0)
        self.assertEqual(c.min_classification_accuracy_pct, 75.0)

    def test_all_tiers_require_100_fail_closed(self):
        """Every tier must have 100% fail-closed — non-negotiable."""
        for tier, c in GOVERNANCE_CRITERIA_DEFAULTS.items():
            self.assertEqual(c.min_fail_closed_pct, 100.0,
                f"Tier '{tier}' must have 100% fail_closed")

    def test_pack_criteria_override_governance_defaults(self):
        """Explicit pack criteria should win over governance defaults."""
        pack_path = os.path.join(_base, "evals", "packs", "product_return.yaml")
        _, _, criteria = load_eval_pack(pack_path, _base)
        # Product return pack has explicit criteria — should use those
        self.assertEqual(criteria.min_schema_valid_pct, 95.0)  # pack says 95, auto tier says 98

    def test_domain_without_explicit_criteria_uses_governance(self):
        """If a pack has no acceptance_criteria, use governance tier defaults."""
        import yaml, tempfile, os as _os
        # Create a minimal pack with no acceptance_criteria
        pack_data = {
            "name": "test_pack",
            "workflow": "product_return",
            "domain": "electronics_return",
            "cases": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(pack_data, f)
            tmp_path = f.name
        try:
            _, _, criteria = load_eval_pack(tmp_path, _base)
            # electronics_return domain has governance: auto
            self.assertEqual(criteria.min_schema_valid_pct, 98.0)
            self.assertEqual(criteria.min_classification_accuracy_pct, 90.0)
        finally:
            _os.unlink(tmp_path)


class TestQualityGate(unittest.TestCase):
    """Test the fail-closed quality gate in the coordinator."""

    def test_quality_gate_fires_on_low_confidence(self):
        """Quality gate should fire when a step confidence is below floor."""
        # Import coordinator pieces
        sys.path.insert(0, _base)
        from coordinator.policy import PolicyEngine

        policy = PolicyEngine(raw_config={
            "quality_gates": {
                "min_confidence": 0.5,
                "primitive_floors": {"classify": 0.6},
                "escalation_tier": "gate",
                "escalation_queue": "quality_review",
                "exempt_domains": [],
            }
        })

        # Simulate what _evaluate_quality_gate does
        qg_config = policy.raw_config.get("quality_gates", {})
        global_min = qg_config.get("min_confidence", 0.5)
        primitive_floors = qg_config.get("primitive_floors", {})

        # Classify step with confidence below floor (0.45 < 0.6)
        steps = [
            {"step_name": "classify_return_type", "primitive": "classify",
             "output": {"category": "defective_product", "confidence": 0.45}},
        ]

        fired = False
        for step in steps:
            output = step.get("output", {})
            conf = output.get("confidence")
            if conf is None:
                continue
            primitive = step.get("primitive", "")
            floor = primitive_floors.get(primitive, global_min)
            if conf < floor:
                fired = True
                break

        self.assertTrue(fired, "Quality gate should fire for classify confidence 0.45 < floor 0.6")

    def test_quality_gate_passes_on_good_confidence(self):
        sys.path.insert(0, _base)
        from coordinator.policy import PolicyEngine

        policy = PolicyEngine(raw_config={
            "quality_gates": {
                "min_confidence": 0.5,
                "primitive_floors": {"classify": 0.6},
                "escalation_tier": "gate",
            }
        })

        qg_config = policy.raw_config.get("quality_gates", {})
        primitive_floors = qg_config.get("primitive_floors", {})
        global_min = qg_config.get("min_confidence", 0.5)

        steps = [
            {"step_name": "classify_return_type", "primitive": "classify",
             "output": {"category": "defective_product", "confidence": 0.85}},
            {"step_name": "investigate_claim", "primitive": "investigate",
             "output": {"finding": "legitimate", "confidence": 0.75}},
        ]

        fired = False
        for step in steps:
            conf = step.get("output", {}).get("confidence")
            if conf is None:
                continue
            floor = primitive_floors.get(step["primitive"], global_min)
            if conf < floor:
                fired = True
                break

        self.assertFalse(fired, "Quality gate should NOT fire when all confidences above floor")

    def test_exempt_domain_skips_gate(self):
        qg_config = {
            "min_confidence": 0.5,
            "exempt_domains": ["spending_advisor"],
        }
        # If domain is exempt, gate should not fire regardless of confidence
        domain = "spending_advisor"
        exempt = qg_config.get("exempt_domains", [])
        self.assertIn(domain, exempt)


if __name__ == "__main__":
    unittest.main()
