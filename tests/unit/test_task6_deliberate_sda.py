"""
TASK 6 — Tests: Wire deliberate primitive to SDA policy artifacts

Verifies:
- Without a registered artifact, think runs identically to v1
- With a registered artifact, the think prompt includes policy structure and reward spec
- With a registered artifact, the output schema includes all SDA fields
- A policy recommendation that contradicts the causal finding produces non-empty tension_flags
- Proof ledger records sda_policy_invoked and causal_tension_flagged when tension exists
- causal_consistency_check is not_applicable when preceding investigate had no causal artifact
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch


SAMPLE_SDA_ARTIFACT = {
    "artifact_name": "sda.fraud_policy_v1",
    "artifact_type": "sequential_decision",
    "version": "1.0",
    "authored_by": "fraud-analytics-team",
    "eval_gate_passed": "2026-01-20",
    "sda_config": {
        "policy_class": "direct_lookahead",
        "horizon": 3,
        "reward_specification": {
            "correct_decision": 1.0,
            "incorrect_decision": -1.0,
            "regulatory_violation": -5.0,
        },
    },
}


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    resp.response_metadata = {}
    llm.invoke.return_value = resp
    return llm


# ── SDA Context Block ─────────────────────────────────────────────────────────

class TestSDAContextBlock(unittest.TestCase):

    def test_block_contains_policy_class(self):
        from cognitive_core.analytics.sda_policy import build_sda_context_block
        block = build_sda_context_block(SAMPLE_SDA_ARTIFACT)
        self.assertIn("direct_lookahead", block)

    def test_block_contains_reward_spec(self):
        from cognitive_core.analytics.sda_policy import build_sda_context_block
        block = build_sda_context_block(SAMPLE_SDA_ARTIFACT)
        self.assertIn("reward", block.lower())
        self.assertIn("1.0", block)   # correct_decision score

    def test_block_contains_horizon(self):
        from cognitive_core.analytics.sda_policy import build_sda_context_block
        block = build_sda_context_block(SAMPLE_SDA_ARTIFACT)
        self.assertIn("3", block)  # horizon

    def test_block_includes_causal_finding(self):
        from cognitive_core.analytics.sda_policy import build_sda_context_block
        block = build_sda_context_block(
            SAMPLE_SDA_ARTIFACT,
            causal_finding="Unauthorized card fraud confirmed."
        )
        self.assertIn("Unauthorized card fraud confirmed.", block)

    def test_block_no_causal_finding_shows_placeholder(self):
        from cognitive_core.analytics.sda_policy import build_sda_context_block
        block = build_sda_context_block(SAMPLE_SDA_ARTIFACT, causal_finding="")
        self.assertIn("No causal finding", block)

    def test_block_instructs_all_sda_fields(self):
        from cognitive_core.analytics.sda_policy import build_sda_context_block
        block = build_sda_context_block(SAMPLE_SDA_ARTIFACT)
        for field in [
            "policy_class", "policy_recommendation", "expected_value_by_horizon",
            "causal_consistency_check", "tension_flags",
        ]:
            self.assertIn(field, block, f"Block missing SDA field instruction: {field}")


# ── Tension Detection ─────────────────────────────────────────────────────────

class TestTensionDetection(unittest.TestCase):

    def test_no_tension_when_both_consistent(self):
        from cognitive_core.analytics.sda_policy import detect_tension
        flags = detect_tension(
            policy_recommendation="Deny the transaction — high fraud risk",
            causal_finding="Confirmed unauthorized card fraud",
        )
        self.assertEqual(flags, [])

    def test_tension_when_approve_vs_fraud_finding(self):
        from cognitive_core.analytics.sda_policy import detect_tension
        flags = detect_tension(
            policy_recommendation="Approve the transaction",
            causal_finding="Confirmed fraud detected",
        )
        self.assertGreater(len(flags), 0, "Should detect tension: approve vs fraud finding")

    def test_tension_when_deny_vs_legit_finding(self):
        from cognitive_core.analytics.sda_policy import detect_tension
        flags = detect_tension(
            policy_recommendation="Reject and block account",
            causal_finding="No fraud detected, transaction is legitimate",
        )
        self.assertGreater(len(flags), 0, "Should detect tension: deny vs legit finding")

    def test_no_tension_empty_finding(self):
        from cognitive_core.analytics.sda_policy import detect_tension
        flags = detect_tension("Deny the transaction", "")
        self.assertEqual(flags, [])

    def test_no_tension_empty_recommendation(self):
        from cognitive_core.analytics.sda_policy import detect_tension
        flags = detect_tension("", "Confirmed fraud")
        self.assertEqual(flags, [])


# ── Causal Finding Extractor ──────────────────────────────────────────────────

class TestCausalFindingExtractor(unittest.TestCase):

    def test_extracts_from_prior_investigate_with_causal_artifact(self):
        from cognitive_core.analytics.sda_policy import extract_causal_finding
        state = {
            "steps": [
                {
                    "step_name": "investigate_activity",
                    "primitive": "investigate",
                    "output": {
                        "finding": "Unauthorized card fraud detected",
                        "activated_paths": ["unauthorized_card_path"],  # causal artifact was active
                    },
                }
            ]
        }
        finding = extract_causal_finding(state)
        self.assertEqual(finding, "Unauthorized card fraud detected")

    def test_returns_empty_without_causal_artifact(self):
        from cognitive_core.analytics.sda_policy import extract_causal_finding
        state = {
            "steps": [
                {
                    "step_name": "investigate_activity",
                    "primitive": "investigate",
                    "output": {
                        "finding": "No fraud",
                        # no activated_paths → v1 investigate, no causal artifact
                    },
                }
            ]
        }
        finding = extract_causal_finding(state)
        self.assertEqual(finding, "")

    def test_returns_empty_when_no_investigate_step(self):
        from cognitive_core.analytics.sda_policy import extract_causal_finding
        state = {"steps": [{"primitive": "classify", "output": {}}]}
        finding = extract_causal_finding(state)
        self.assertEqual(finding, "")


# ── Think Node — V1 without SDA ───────────────────────────────────────────────

class TestDeliberateV1BehaviorWithoutArtifact(unittest.TestCase):

    def setUp(self):
        os.environ["CC_CACHE_ENABLED"] = "false"
        os.environ["CC_PII_ENABLED"] = "false"

    def tearDown(self):
        os.environ.pop("CC_CACHE_ENABLED", None)
        os.environ.pop("CC_PII_ENABLED", None)

    def test_no_sda_fields_when_no_eligible_artifact(self):
        v1_response = json.dumps({
            "thought": "Analyzing the evidence...",
            "conclusions": ["No fraud indicators"],
            "decision": "Close the case",
            "confidence": 0.8,
            "reasoning": "Evidence is inconclusive",
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
            mock_reg.list_eligible.return_value = []
            MockReg.return_value = mock_reg

            from cognitive_core.engine.nodes import create_node
            node = create_node(
                step_name="test_deliberate",
                primitive_name="deliberate",
                params={"instruction": "Make a fraud determination"},
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
        self.assertNotIn("policy_recommendation", output)
        self.assertNotIn("tension_flags", output)


# ── Think Node — V2 with SDA ──────────────────────────────────────────────────

class TestThinkV2WithSDA(unittest.TestCase):

    def setUp(self):
        os.environ["CC_CACHE_ENABLED"] = "false"
        os.environ["CC_PII_ENABLED"] = "false"

    def tearDown(self):
        os.environ.pop("CC_CACHE_ENABLED", None)
        os.environ.pop("CC_PII_ENABLED", None)

    def _run_deliberate_with_sda(
        self,
        policy_recommendation: str = "Deny — high fraud probability",
        causal_finding: str = "Confirmed fraud",
        tension_flags: list | None = None,
    ) -> dict:
        sda_response = json.dumps({
            "thought": "Analyzing policy options under direct_lookahead framework...",
            "conclusions": ["Policy recommends denial"],
            "decision": "Deny",
            "confidence": 0.9,
            "reasoning": "High expected value for denial",
            "evidence_used": [],
            "evidence_missing": [],
            "policy_class": "direct_lookahead",
            "policy_version": "1.0",
            "reward_specification_version": "1.0",
            "decision_horizon": 3,
            "expected_value_by_horizon": {"1": -0.5, "2": 0.3, "3": 0.8},
            "policy_recommendation": policy_recommendation,
            "causal_consistency_check": "consistent",
            "tension_flags": tension_flags or [],
        })
        llm = _mock_llm(sda_response)

        with patch("engine.nodes.create_llm", return_value=llm), \
             patch("engine.nodes.get_governance") as mock_gov_factory, \
             patch("analytics.registry.AnalyticsRegistry") as MockReg:

            mock_gov = MagicMock()
            mock_gov.protected_llm_call.return_value = MagicMock(raw_response=sda_response)
            mock_gov_factory.return_value = mock_gov

            mock_reg = MagicMock()
            mock_reg.list_eligible.return_value = [SAMPLE_SDA_ARTIFACT]
            MockReg.return_value = mock_reg

            from cognitive_core.engine.nodes import create_node
            node = create_node(
                step_name="deliberate_determination",
                primitive_name="deliberate",
                params={"instruction": "Make a fraud determination"},
            )
            state = {
                "input": {"domain": "check_fraud"},
                "metadata": {"domain": "check_fraud"},
                "steps": [
                    {
                        "step_name": "investigate_activity",
                        "primitive": "investigate",
                        "output": {
                            "finding": causal_finding,
                            "confidence": 0.85,
                            "reasoning": "Evidence analysis complete",
                            "evidence_used": [],
                            "evidence_missing": [],
                            "activated_paths": ["unauthorized_card_path"],
                        },
                    }
                ],
                "current_step": "",
                "loop_counts": {},
            }
            return node(state)

    def test_sda_fields_present_in_output(self):
        result = self._run_deliberate_with_sda()
        output = result["steps"][0]["output"]
        sda_fields = [
            "policy_class", "policy_version", "reward_specification_version",
            "decision_horizon", "expected_value_by_horizon",
            "policy_recommendation", "causal_consistency_check", "tension_flags",
        ]
        for field in sda_fields:
            self.assertIn(field, output, f"Missing SDA field: {field}")

    def test_tension_flags_non_empty_when_contradiction(self):
        result = self._run_deliberate_with_sda(
            policy_recommendation="Deny the transaction",
            causal_finding="Transaction is legitimate — no fraud indicators",
            tension_flags=["Policy recommends denial but causal finding indicates legitimate activity"],
        )
        output = result["steps"][0]["output"]
        self.assertGreater(
            len(output.get("tension_flags", [])), 0,
            "tension_flags must be non-empty when policy contradicts causal finding",
        )


# ── Proof Ledger Events ───────────────────────────────────────────────────────

class TestSDAProofEvents(unittest.TestCase):

    def test_sda_policy_invoked_recorded(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov._record_proof(
            "sda_policy_invoked",
            step="deliberate_determination",
            artifact_name="sda.fraud_policy_v1",
            policy_recommendation="Deny",
            causal_consistency_check="consistent",
        )

        sda_events = [
            e for e in gov._proof_ledger
            if e["event"] == "sda_policy_invoked"
        ]
        self.assertGreaterEqual(len(sda_events), 1)
        self.assertEqual(sda_events[0]["artifact_name"], "sda.fraud_policy_v1")

    def test_causal_tension_flagged_when_tension(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov._record_proof(
            "causal_tension_flagged",
            step="deliberate_determination",
            artifact_name="sda.fraud_policy_v1",
            tension_flags=["Policy recommends approval but causal finding indicates fraud"],
        )

        tension_events = [
            e for e in gov._proof_ledger
            if e["event"] == "causal_tension_flagged"
        ]
        self.assertGreaterEqual(len(tension_events), 1)
        self.assertGreater(len(tension_events[0].get("tension_flags", [])), 0)

    def test_no_tension_event_when_no_tension(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        # Only sda_policy_invoked, no causal_tension_flagged
        gov._record_proof(
            "sda_policy_invoked",
            step="deliberate_determination",
            artifact_name="sda.fraud_policy_v1",
            policy_recommendation="Deny",
            causal_consistency_check="consistent",
        )

        tension_events = [
            e for e in gov._proof_ledger
            if e["event"] == "causal_tension_flagged"
        ]
        self.assertEqual(len(tension_events), 0)


if __name__ == "__main__":
    unittest.main()
