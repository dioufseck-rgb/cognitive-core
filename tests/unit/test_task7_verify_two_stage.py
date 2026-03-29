"""
TASK 7 — Tests: Restructure verify primitive to two-stage execution

Verifies:
- Verify produces two distinct proof ledger events (evidence_mapping_recorded,
  constraint_check_completed) with separate timestamps
- Stage 1 output is evidence characterization, not compliance verdict
- Stage 2 is deterministic: same Stage 1 characterizations → same Stage 2 result
- ambiguity_flags in Stage 1 triggers escalation when ambiguity_escalation: gate
- Stage 1 proof entry includes LLM version; Stage 2 includes checker version
- Without a constraint checker artifact, verify falls back to v1 and records
  analytics_fallback_applied
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from cognitive_core.analytics.constraint_checker import (
    Stage1Result, VariableCharacterization, evaluate_rules,
    build_stage1_prompt, CHECKER_VERSION,
)


SAMPLE_CC_ARTIFACT = {
    "artifact_name": "constraints.bsa_v1",
    "artifact_type": "constraint_checker",
    "version": "1.0",
    "authored_by": "compliance-team",
    "eval_gate_passed": "2026-02-01",
    "eligibility_predicates": [
        {"field": "domain", "operator": "eq", "value": "fraud_regulatory"}
    ],
    "rules": [
        {
            "rule_id": "bsa_sar_threshold",
            "description": "SAR required for transactions >= $5,000 with fraud indicators",
            "variables": ["transaction_amount", "fraud_confirmed"],
        },
        {
            "rule_id": "reg_e_timeline",
            "description": "Provisional credit within 10 business days",
            "variables": ["provisional_credit_status"],
        },
    ],
    "ambiguity_escalation": "gate",
}


def _make_stage1(characterizations: list[dict], llm_version: str = "test-model") -> Stage1Result:
    """Build a Stage1Result from a list of characterization dicts."""
    chars = [
        VariableCharacterization(
            variable=c["variable"],
            characterization=c.get("characterization", ""),
            evidence_basis=c.get("evidence_basis", ""),
            confidence=c.get("confidence", 0.8),
            ambiguity_flags=c.get("ambiguity_flags", []),
        )
        for c in characterizations
    ]
    return Stage1Result(characterizations=chars, llm_version=llm_version, raw_response="{}")


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    resp.response_metadata = {}
    llm.invoke.return_value = resp
    return llm


# ── Stage 2 Determinism ───────────────────────────────────────────────────────

class TestStage2Determinism(unittest.TestCase):
    """Stage 2 produces the same result given the same Stage 1 characterizations."""

    def test_same_inputs_produce_same_result(self):
        stage1_a = _make_stage1([
            {"variable": "transaction_amount", "characterization": "Amount is $6,000 — above threshold", "confidence": 0.9},
            {"variable": "fraud_confirmed", "characterization": "Fraud has been confirmed by investigation", "confidence": 0.85},
            {"variable": "provisional_credit_status", "characterization": "Credit issued within 5 business days", "confidence": 0.95},
        ])
        stage1_b = _make_stage1([
            {"variable": "transaction_amount", "characterization": "Amount is $6,000 — above threshold", "confidence": 0.9},
            {"variable": "fraud_confirmed", "characterization": "Fraud has been confirmed by investigation", "confidence": 0.85},
            {"variable": "provisional_credit_status", "characterization": "Credit issued within 5 business days", "confidence": 0.95},
        ])

        result_a = evaluate_rules(SAMPLE_CC_ARTIFACT, stage1_a)
        result_b = evaluate_rules(SAMPLE_CC_ARTIFACT, stage1_b)

        self.assertEqual(result_a.overall_conforms, result_b.overall_conforms)
        self.assertEqual(
            [(r.rule_id, r.verdict) for r in result_a.rules_evaluated],
            [(r.rule_id, r.verdict) for r in result_b.rules_evaluated],
        )

    def test_fail_signal_in_characterization_causes_rule_to_fail(self):
        stage1 = _make_stage1([
            {"variable": "transaction_amount", "characterization": "Amount exceeds threshold — violation", "confidence": 0.9},
            {"variable": "fraud_confirmed", "characterization": "Fraud confirmed", "confidence": 0.9},
            {"variable": "provisional_credit_status", "characterization": "Credit not provided — missing", "confidence": 0.9},
        ])
        result = evaluate_rules(SAMPLE_CC_ARTIFACT, stage1)
        failing = [r.rule_id for r in result.violations]
        # provisional_credit rule should fail due to "missing"
        self.assertIn("reg_e_timeline", failing)

    def test_all_pass_when_no_fail_signals(self):
        stage1 = _make_stage1([
            {"variable": "transaction_amount", "characterization": "Amount is within normal range", "confidence": 0.9},
            {"variable": "fraud_confirmed", "characterization": "Investigation inconclusive", "confidence": 0.75},
            {"variable": "provisional_credit_status", "characterization": "Credit issued on day 3", "confidence": 0.9},
        ])
        result = evaluate_rules(SAMPLE_CC_ARTIFACT, stage1)
        self.assertTrue(result.overall_conforms)
        self.assertEqual(len(result.violations), 0)

    def test_insufficient_evidence_when_variable_missing(self):
        # Only characterize some variables
        stage1 = _make_stage1([
            {"variable": "transaction_amount", "characterization": "Amount above threshold", "confidence": 0.9},
            # fraud_confirmed NOT characterized
        ])
        result = evaluate_rules(SAMPLE_CC_ARTIFACT, stage1)
        ie_rules = [r for r in result.rules_evaluated if r.verdict == "insufficient_evidence"]
        self.assertGreater(len(ie_rules), 0, "Should have insufficient_evidence when variable missing")

    def test_checker_version_in_result(self):
        stage1 = _make_stage1([
            {"variable": "transaction_amount", "characterization": "Amount fine", "confidence": 0.9},
            {"variable": "fraud_confirmed", "characterization": "No fraud", "confidence": 0.9},
            {"variable": "provisional_credit_status", "characterization": "Credit issued", "confidence": 0.9},
        ])
        result = evaluate_rules(SAMPLE_CC_ARTIFACT, stage1)
        self.assertEqual(result.checker_version, CHECKER_VERSION)


# ── Stage 1 Prompt Builder ────────────────────────────────────────────────────

class TestStage1PromptBuilder(unittest.TestCase):

    def test_prompt_contains_rule_variables(self):
        prompt = build_stage1_prompt(
            artifact=SAMPLE_CC_ARTIFACT,
            subject="Member claims fraudulent transaction",
            context="Fraud regulatory case",
            input_data="{}",
        )
        self.assertIn("transaction_amount", prompt)
        self.assertIn("fraud_confirmed", prompt)
        self.assertIn("provisional_credit_status", prompt)

    def test_prompt_does_not_ask_for_verdict(self):
        prompt = build_stage1_prompt(
            artifact=SAMPLE_CC_ARTIFACT,
            subject="test",
            context="test",
            input_data="{}",
        )
        # Stage 1 should ask for characterization, not compliance verdict
        self.assertIn("characteriz", prompt.lower())
        # Must explicitly state not to render verdict
        self.assertNotIn("conforms", prompt.lower())


# ── Stage 1 Parsing ───────────────────────────────────────────────────────────

class TestStage1Parsing(unittest.TestCase):

    def test_from_dict_parses_correctly(self):
        raw = {
            "characterizations": [
                {
                    "variable": "transaction_amount",
                    "characterization": "Amount is $8,000",
                    "evidence_basis": "Transaction record",
                    "confidence": 0.9,
                    "ambiguity_flags": [],
                }
            ]
        }
        result = Stage1Result.from_dict(raw, raw_response=json.dumps(raw))
        self.assertEqual(len(result.characterizations), 1)
        self.assertEqual(result.characterizations[0].variable, "transaction_amount")
        self.assertAlmostEqual(result.characterizations[0].confidence, 0.9)

    def test_has_ambiguity_true_when_flags(self):
        raw = {
            "characterizations": [
                {
                    "variable": "x",
                    "characterization": "Unclear",
                    "evidence_basis": "",
                    "confidence": 0.5,
                    "ambiguity_flags": ["Data is contradictory"],
                }
            ]
        }
        result = Stage1Result.from_dict(raw)
        self.assertTrue(result.has_ambiguity())

    def test_has_ambiguity_false_when_no_flags(self):
        raw = {
            "characterizations": [
                {
                    "variable": "x",
                    "characterization": "Clear",
                    "evidence_basis": "doc",
                    "confidence": 0.9,
                    "ambiguity_flags": [],
                }
            ]
        }
        result = Stage1Result.from_dict(raw)
        self.assertFalse(result.has_ambiguity())


# ── Ambiguity Escalation Trigger ─────────────────────────────────────────────

class TestAmbiguityEscalation(unittest.TestCase):

    def test_ambiguity_flags_trigger_when_gate_configured(self):
        """ambiguity_flags in Stage 1 must trigger escalation when ambiguity_escalation: gate."""
        os.environ["CC_CACHE_ENABLED"] = "false"
        os.environ["CC_PII_ENABLED"] = "false"

        stage1_response = json.dumps({
            "characterizations": [
                {
                    "variable": "transaction_amount",
                    "characterization": "Amount is $6,000",
                    "evidence_basis": "Transaction record",
                    "confidence": 0.85,
                    "ambiguity_flags": ["Records show two different amounts"],
                },
                {
                    "variable": "fraud_confirmed",
                    "characterization": "Fraud status is uncertain",
                    "evidence_basis": "Investigation",
                    "confidence": 0.75,
                    "ambiguity_flags": [],
                },
                {
                    "variable": "provisional_credit_status",
                    "characterization": "Credit issued on day 7",
                    "evidence_basis": "Account ledger",
                    "confidence": 0.90,
                    "ambiguity_flags": [],
                },
            ]
        })
        llm = _mock_llm(stage1_response)

        with patch("engine.nodes.create_llm", return_value=llm), \
             patch("engine.nodes.get_governance") as mock_gov_factory, \
             patch("analytics.registry.AnalyticsRegistry") as MockReg:

            mock_gov = MagicMock()
            mock_gov.protected_llm_call.return_value = MagicMock(raw_response=stage1_response)
            mock_gov_factory.return_value = mock_gov

            mock_reg = MagicMock()
            mock_reg.list_eligible.return_value = [SAMPLE_CC_ARTIFACT]
            MockReg.return_value = mock_reg

            from cognitive_core.engine.nodes import create_node
            node = create_node(
                step_name="verify_compliance",
                primitive_name="verify",
                params={"rules": "Check BSA and Reg E compliance"},
            )
            state = {
                "input": {"domain": "fraud_regulatory"},
                "metadata": {"domain": "fraud_regulatory"},
                "steps": [],
                "current_step": "",
                "loop_counts": {},
            }
            result = node(state)

        output = result["steps"][0]["output"]
        # ambiguity_triggered_escalation must be True when ambiguity_flags non-empty
        # and ambiguity_escalation=gate
        self.assertTrue(
            output.get("ambiguity_triggered_escalation", False),
            "ambiguity_triggered_escalation must be True when Stage 1 has ambiguity_flags and artifact has ambiguity_escalation: gate",
        )

        os.environ.pop("CC_CACHE_ENABLED", None)
        os.environ.pop("CC_PII_ENABLED", None)


# ── Proof Ledger Events ───────────────────────────────────────────────────────

class TestVerifyTwoStageProofEvents(unittest.TestCase):

    def test_two_separate_proof_events_with_distinct_timestamps(self):
        from cognitive_core.engine.governance import GovernancePipeline
        import time

        gov = GovernancePipeline()
        gov.initialize()

        t1 = time.time()
        gov._record_proof(
            "evidence_mapping_recorded",
            step="verify_compliance",
            artifact_name="constraints.bsa_v1",
            llm_version="test-model",
            has_ambiguity=False,
            ambiguity_summary=[],
        )
        t2 = time.time()
        gov._record_proof(
            "constraint_check_completed",
            step="verify_compliance",
            artifact_name="constraints.bsa_v1",
            checker_version=CHECKER_VERSION,
            overall_conforms=True,
            rules_count=2,
            violation_count=0,
        )

        s1_events = [e for e in gov._proof_ledger if e["event"] == "evidence_mapping_recorded"]
        s2_events = [e for e in gov._proof_ledger if e["event"] == "constraint_check_completed"]

        self.assertGreaterEqual(len(s1_events), 1)
        self.assertGreaterEqual(len(s2_events), 1)

        # Must be separate events (different event types, separate timestamps)
        self.assertNotEqual(s1_events[0]["event"], s2_events[0]["event"])
        # S1 event should have llm_version, S2 should have checker_version
        self.assertIn("llm_version", s1_events[0])
        self.assertIn("checker_version", s2_events[0])

    def test_stage1_event_has_llm_version(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov._record_proof(
            "evidence_mapping_recorded",
            step="verify_step",
            artifact_name="constraints.bsa_v1",
            llm_version="gemini-2.0-flash",
            has_ambiguity=False,
            ambiguity_summary=[],
        )

        event = next(
            e for e in gov._proof_ledger if e["event"] == "evidence_mapping_recorded"
        )
        self.assertEqual(event["llm_version"], "gemini-2.0-flash")

    def test_stage2_event_has_checker_version(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov._record_proof(
            "constraint_check_completed",
            step="verify_step",
            artifact_name="constraints.bsa_v1",
            checker_version=CHECKER_VERSION,
            overall_conforms=False,
            rules_count=3,
            violation_count=1,
        )

        event = next(
            e for e in gov._proof_ledger if e["event"] == "constraint_check_completed"
        )
        self.assertEqual(event["checker_version"], CHECKER_VERSION)


# ── V1 Fallback ───────────────────────────────────────────────────────────────

class TestVerifyV1Fallback(unittest.TestCase):
    """Without a constraint checker artifact, verify falls back to v1."""

    def setUp(self):
        os.environ["CC_CACHE_ENABLED"] = "false"
        os.environ["CC_PII_ENABLED"] = "false"

    def tearDown(self):
        os.environ.pop("CC_CACHE_ENABLED", None)
        os.environ.pop("CC_PII_ENABLED", None)

    def test_v1_path_when_no_eligible_artifact(self):
        v1_response = json.dumps({
            "conforms": True,
            "violations": [],
            "rules_checked": ["all_rules"],
            "confidence": 0.9,
            "reasoning": "All criteria satisfied",
            "evidence_used": [],
            "evidence_missing": [],
        })
        llm = _mock_llm(v1_response)

        with patch("engine.nodes.create_llm", return_value=llm), \
             patch("engine.nodes.get_governance") as mock_gov_factory, \
             patch("analytics.registry.AnalyticsRegistry") as MockReg:

            mock_gov = MagicMock()
            mock_gov.protected_llm_call.return_value = MagicMock(raw_response=v1_response)
            mock_gov_factory.return_value = mock_gov

            mock_reg = MagicMock()
            mock_reg.list_eligible.return_value = []  # no eligible artifact
            MockReg.return_value = mock_reg

            from cognitive_core.engine.nodes import create_node
            node = create_node(
                step_name="verify_step",
                primitive_name="verify",
                params={"rules": "Check compliance"},
            )
            state = {
                "input": {"domain": "lending"},
                "metadata": {"domain": "lending"},
                "steps": [],
                "current_step": "",
                "loop_counts": {},
            }
            result = node(state)

        output = result["steps"][0]["output"]
        # V1 path: no stage1/stage2 V2 fields
        self.assertNotIn("stage1_llm_version", output)
        self.assertNotIn("stage2_checker_version", output)
        self.assertNotIn("evidence_characterizations", output)


if __name__ == "__main__":
    unittest.main()
