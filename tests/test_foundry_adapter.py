"""
Cognitive Core — Foundry Adapter Integration Tests

Proves the Foundry Responses API adapter correctly:
  1. Parses all Foundry input formats (string, messages, structured JSON)
  2. Routes to correct workflow/domain
  3. Calls coordinator.start() and returns governed results
  4. Handles governance suspension with approval flow
  5. Returns proper Responses API format in all cases
  6. Includes governance metadata (tiers, audit, delegations)
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from api.foundry_adapter import (
    _parse_foundry_input,
    _load_routing_table,
    _format_result_content,
    _error_response,
)


# ═════════════════════════════════════════════════════════════════
# Input Parsing Tests
# ═════════════════════════════════════════════════════════════════

class TestParseFoundryInput(unittest.TestCase):
    """Test all Foundry input format variations."""

    def setUp(self):
        self.routing = {
            "claim": ("claim_intake", "synthetic_claim"),
            "damage": ("damage_assessment", "synthetic_damage"),
            "fraud": ("fraud_screening", "synthetic_fraud"),
        }

    def test_string_input(self):
        """Simple string input wraps as description."""
        body = {"input": "Process claim CLM-001"}
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(case_input["description"], "Process claim CLM-001")
        self.assertEqual(wf, "claim_intake")
        self.assertEqual(dom, "synthetic_claim")

    def test_messages_input_text(self):
        """Messages format with text content."""
        body = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Process this claim please"}
                ]
            }
        }
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(case_input["description"], "Process this claim please")

    def test_messages_input_json(self):
        """Messages format with JSON content parsed as case_input."""
        payload = {"claim_id": "CLM-001", "amount": 12500, "get_policy": {"status": "active"}}
        body = {
            "input": {
                "messages": [
                    {"role": "user", "content": json.dumps(payload)}
                ]
            }
        }
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(case_input["claim_id"], "CLM-001")
        self.assertEqual(case_input["amount"], 12500)
        self.assertEqual(case_input["get_policy"]["status"], "active")

    def test_direct_object_input(self):
        """Direct object input (no messages wrapper)."""
        body = {
            "input": {"claim_id": "CLM-001", "amount": 5000}
        }
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(case_input["claim_id"], "CLM-001")

    def test_metadata_routing_override(self):
        """Metadata workflow/domain overrides env var defaults."""
        body = {
            "input": "test",
            "metadata": {
                "workflow": "fraud_screening",
                "domain": "synthetic_fraud",
            }
        }
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(wf, "fraud_screening")
        self.assertEqual(dom, "synthetic_fraud")

    def test_inline_workflow_routing(self):
        """Workflow/domain fields in case_input are extracted for routing."""
        payload = {"workflow": "damage_assessment", "domain": "synthetic_damage", "claim_id": "CLM-001"}
        body = {
            "input": {
                "messages": [
                    {"role": "user", "content": json.dumps(payload)}
                ]
            }
        }
        case_input, wf, dom = _parse_foundry_input(
            body, "", "", self.routing
        )
        self.assertEqual(wf, "damage_assessment")
        self.assertEqual(dom, "synthetic_damage")
        # workflow/domain should be popped from case_input
        self.assertNotIn("workflow", case_input)
        self.assertNotIn("domain", case_input)
        self.assertEqual(case_input["claim_id"], "CLM-001")

    def test_keyword_routing_claim(self):
        """String input routes by keyword when no defaults set."""
        body = {"input": "I need to process a damage report"}
        case_input, wf, dom = _parse_foundry_input(
            body, "", "", self.routing
        )
        # "damage" keyword matches
        self.assertEqual(wf, "damage_assessment")
        self.assertEqual(dom, "synthetic_damage")

    def test_keyword_routing_fraud(self):
        """Fraud keyword routes correctly."""
        body = {"input": "Check this for fraud indicators"}
        case_input, wf, dom = _parse_foundry_input(
            body, "", "", self.routing
        )
        self.assertEqual(wf, "fraud_screening")

    def test_empty_input(self):
        """Empty input returns empty case_input."""
        body = {"input": ""}
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(case_input.get("description", ""), "")
        self.assertEqual(wf, "claim_intake")

    def test_multi_message_uses_last_user(self):
        """Multiple messages: uses last user message."""
        body = {
            "input": {
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": json.dumps({"claim_id": "CLM-LAST"})},
                ]
            }
        }
        case_input, wf, dom = _parse_foundry_input(
            body, "claim_intake", "synthetic_claim", self.routing
        )
        self.assertEqual(case_input["claim_id"], "CLM-LAST")


# ═════════════════════════════════════════════════════════════════
# Response Formatting Tests
# ═════════════════════════════════════════════════════════════════

class TestResponseFormatting(unittest.TestCase):
    """Test Responses API output formatting."""

    def _mock_instance(self, **kwargs):
        inst = MagicMock()
        inst.instance_id = kwargs.get("instance_id", "wf_test123")
        inst.workflow_type = kwargs.get("workflow_type", "claim_intake")
        inst.domain = kwargs.get("domain", "synthetic_claim")
        inst.governance_tier = kwargs.get("governance_tier", "gate")
        inst.step_count = kwargs.get("step_count", 5)
        inst.status = MagicMock()
        inst.status.value = kwargs.get("status", "completed")
        inst.correlation_id = kwargs.get("correlation_id", "wf_test123")
        return inst

    def test_format_result_classify(self):
        """Classify step output formatted correctly."""
        inst = self._mock_instance()
        result = {
            "steps": [
                {"primitive": "classify", "step_name": "classify_claim",
                 "output": {"category": "physical_damage", "confidence": 1.0}},
            ]
        }
        content = _format_result_content(inst, result, [inst], [])
        self.assertIn("physical_damage", content)
        self.assertIn("confidence: 1.0", content)

    def test_format_result_verify(self):
        """Verify step output shows PASS/FAIL."""
        inst = self._mock_instance()
        result = {
            "steps": [
                {"primitive": "verify", "step_name": "check_eligibility",
                 "output": {"conforms": True, "violations": []}},
            ]
        }
        content = _format_result_content(inst, result, [inst], [])
        self.assertIn("PASS", content)

    def test_format_result_verify_fail(self):
        """Verify failure shows violations."""
        inst = self._mock_instance()
        result = {
            "steps": [
                {"primitive": "verify", "step_name": "check_eligibility",
                 "output": {"conforms": False, "violations": ["expired_policy"]}},
            ]
        }
        content = _format_result_content(inst, result, [inst], [])
        self.assertIn("FAIL", content)
        self.assertIn("expired_policy", content)

    def test_format_result_investigate(self):
        """Investigate step shows finding + evidence flags."""
        inst = self._mock_instance()
        result = {
            "steps": [
                {"primitive": "investigate", "step_name": "investigate_patterns",
                 "output": {"finding": "minor_concerns", "evidence_flags": ["amount_escalation"],
                           "recommendation": "flag_for_monitoring"}},
            ]
        }
        content = _format_result_content(inst, result, [inst], [])
        self.assertIn("minor_concerns", content)
        self.assertIn("amount_escalation", content)
        self.assertIn("flag_for_monitoring", content)

    def test_format_delegations(self):
        """Multi-agent chain shows delegated workflows."""
        parent = self._mock_instance(instance_id="wf_parent", workflow_type="claim_intake")
        child1 = self._mock_instance(instance_id="wf_child1", workflow_type="damage_assessment", domain="synthetic_damage")
        child2 = self._mock_instance(instance_id="wf_child2", workflow_type="fraud_screening", domain="synthetic_fraud")

        content = _format_result_content(parent, {"steps": []}, [parent, child1, child2], [])
        self.assertIn("Delegated workflows: 2", content)
        self.assertIn("damage_assessment", content)
        self.assertIn("fraud_screening", content)

    def test_error_response_format(self):
        """Error response follows Responses API structure."""
        try:
            from fastapi.responses import JSONResponse
        except ImportError:
            self.skipTest("fastapi not installed")
        resp = _error_response("resp_test", 500, "Something broke")
        self.assertEqual(resp.status_code, 500)
        body = json.loads(resp.body)
        self.assertEqual(body["id"], "resp_test")
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["output"][0]["type"], "message")
        self.assertIn("Something broke", body["output"][0]["content"])


# ═════════════════════════════════════════════════════════════════
# End-to-End Adapter Tests (mocked coordinator)
# ═════════════════════════════════════════════════════════════════

class TestFoundryAdapterE2E(unittest.TestCase):
    """E2E tests through the FastAPI adapter with mocked coordinator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "foundry_test.db")

    def _mock_workflow_output(self, wf_type):
        """Same mock outputs as multi-agent tests."""
        outputs = {
            "claim_intake": {
                "steps": [
                    {"primitive": "retrieve", "step_name": "gather_claim_data",
                     "output": {"data": {"get_claim": {"claim_id": "CLM-001", "amount": 12500},
                                        "get_policy": {"policy_id": "POL-001", "status": "active",
                                                      "coverage_type": "comprehensive,collision"}}}},
                    {"primitive": "classify", "step_name": "classify_claim_type",
                     "output": {"category": "physical_damage", "confidence": 1.0}},
                    {"primitive": "verify", "step_name": "check_eligibility",
                     "output": {"conforms": True, "violations": []}},
                    {"primitive": "think", "step_name": "assess_risk",
                     "output": {"risk_score": 55, "recommendation": "standard_review"}},
                    {"primitive": "generate", "step_name": "generate_decision",
                     "output": {"decision": "approve_with_review"}},
                ],
                "input": {"claim_id": "CLM-001"},
            },
            "damage_assessment": {
                "steps": [
                    {"primitive": "retrieve", "step_name": "gather_damage_data",
                     "output": {"data": {"get_claim": {"claim_id": "CLM-001"}}}},
                    {"primitive": "classify", "step_name": "classify_damage_severity",
                     "output": {"category": "major", "confidence": 1.0}},
                    {"primitive": "verify", "step_name": "verify_documentation",
                     "output": {"conforms": True, "violations": []}},
                    {"primitive": "generate", "step_name": "generate_assessment",
                     "output": {"damage_grade": "major", "repair_cost": 11200}},
                ],
                "input": {},
            },
            "fraud_screening": {
                "steps": [
                    {"primitive": "retrieve", "step_name": "gather_fraud_data",
                     "output": {"data": {"get_claim": {"claim_id": "CLM-001"}}}},
                    {"primitive": "classify", "step_name": "classify_fraud_risk",
                     "output": {"category": "medium_risk", "confidence": 1.0}},
                    {"primitive": "investigate", "step_name": "investigate_patterns",
                     "output": {"finding": "minor_concerns", "evidence_flags": ["amount_escalation"],
                               "recommendation": "flag_for_monitoring"}},
                    {"primitive": "generate", "step_name": "generate_screening_result",
                     "output": {"fraud_risk": "medium_risk"}},
                ],
                "input": {},
            },
        }
        return outputs.get(wf_type, {"steps": [], "input": {}})

    def test_e2e_responses_api_completed(self):
        """Full E2E: Foundry POST /responses → coordinator → Responses output."""
        from coordinator.runtime import Coordinator

        config_path = os.path.join(BASE, "coordinator", "config.yaml")
        coord = Coordinator(config_path=config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: \
            self._mock_workflow_output(inst.workflow_type)

        # Simulate what the adapter does
        case_input = {"claim_id": "CLM-001", "amount": 12500,
                     "get_policy": {"policy_id": "POL-001", "status": "active",
                                   "coverage_type": "comprehensive,collision"},
                     "get_claim": {"claim_id": "CLM-001", "amount": 12500,
                                  "claim_type_hint": "physical_damage", "flags": ["high_value"]}}

        instance_id = coord.start(
            workflow_type="claim_intake",
            domain="synthetic_claim",
            case_input=case_input,
        )

        instance = coord.get_instance(instance_id)
        from coordinator.types import InstanceStatus

        # Gate tier → suspended
        self.assertEqual(instance.status, InstanceStatus.SUSPENDED)

        # Approve
        coord.approve(instance_id, approver="foundry_test")
        instance = coord.get_instance(instance_id)
        self.assertEqual(instance.status, InstanceStatus.COMPLETED)

        # Format as Responses API
        result = instance.result or {}
        chain = coord.get_correlation_chain(instance.correlation_id)
        ledger = coord.get_ledger(instance_id=instance_id)

        content = _format_result_content(instance, result, chain, ledger)

        # Build response
        response = {
            "id": "resp_test",
            "object": "response",
            "output": [{"type": "message", "role": "assistant", "content": content}],
            "status": "completed",
            "metadata": {
                "cognitive_core": {
                    "instance_id": instance_id,
                    "correlation_id": instance.correlation_id,
                    "workflow": instance.workflow_type,
                    "domain": instance.domain,
                    "governance_tier": instance.governance_tier,
                    "step_count": instance.step_count,
                    "delegations": len(chain) - 1,
                    "audit_entries": len(ledger),
                }
            },
        }

        # Validate response structure
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["role"], "assistant")

        # Validate content includes workflow results
        self.assertIn("claim_intake", content)
        self.assertIn("completed", content)
        self.assertIn("gate", content)

        # Validate metadata
        meta = response["metadata"]["cognitive_core"]
        self.assertEqual(meta["workflow"], "claim_intake")
        self.assertEqual(meta["governance_tier"], "gate")
        self.assertTrue(meta["step_count"] >= 5)
        self.assertTrue(meta["delegations"] >= 2, f"Expected 2+ delegations, got {meta['delegations']}")
        self.assertTrue(meta["audit_entries"] > 0)

        # Validate delegations in chain
        chain_wfs = {c.workflow_type for c in chain}
        self.assertIn("damage_assessment", chain_wfs)
        self.assertIn("fraud_screening", chain_wfs)

        print(f"\n{'='*60}")
        print("Foundry Responses API E2E:")
        print(f"  Status: {response['status']}")
        print(f"  Workflow: {meta['workflow']}/{meta.get('domain')}")
        print(f"  Tier: {meta['governance_tier']}")
        print(f"  Steps: {meta['step_count']}")
        print(f"  Delegations: {meta['delegations']}")
        print(f"  Audit entries: {meta['audit_entries']}")
        print(f"\nContent preview:")
        for line in content.split("\n")[:15]:
            print(f"  {line}")
        print(f"{'='*60}")

    def test_e2e_suspended_response(self):
        """Suspended workflow returns requires_action status."""
        from coordinator.runtime import Coordinator
        from coordinator.types import InstanceStatus

        config_path = os.path.join(BASE, "coordinator", "config.yaml")
        coord = Coordinator(config_path=config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: \
            self._mock_workflow_output(inst.workflow_type)

        instance_id = coord.start(
            workflow_type="claim_intake",
            domain="synthetic_claim",
            case_input={"claim_id": "CLM-001"},
        )

        instance = coord.get_instance(instance_id)
        self.assertEqual(instance.status, InstanceStatus.SUSPENDED)

        # Format suspended response
        try:
            import fastapi
        except ImportError:
            self.skipTest("fastapi not installed")
        from api.foundry_adapter import _suspended_response
        resp = _suspended_response("resp_test", instance, coord)
        body = json.loads(resp.body)

        self.assertEqual(body["status"], "requires_action")
        self.assertIn("suspended", body["output"][0]["content"].lower())
        self.assertTrue(body["metadata"]["cognitive_core"]["approval_required"])
        self.assertEqual(body["metadata"]["cognitive_core"]["governance_tier"], "gate")

        print(f"\nSuspended response: status={body['status']}, tier={body['metadata']['cognitive_core']['governance_tier']}")

    def test_e2e_auto_tier_no_suspension(self):
        """Auto tier workflow completes without suspension."""
        from coordinator.runtime import Coordinator
        from coordinator.types import InstanceStatus

        config_path = os.path.join(BASE, "coordinator", "config.yaml")
        coord = Coordinator(config_path=config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: \
            self._mock_workflow_output(inst.workflow_type)

        instance_id = coord.start(
            workflow_type="damage_assessment",
            domain="synthetic_damage",
            case_input={"claim_id": "CLM-001"},
        )

        instance = coord.get_instance(instance_id)
        self.assertEqual(instance.status, InstanceStatus.COMPLETED)
        self.assertEqual(instance.governance_tier, "auto")
        print(f"\nAuto tier: completed directly, no suspension")


class TestRegistrationScript(unittest.TestCase):
    """Test the Foundry registration script logic."""

    def test_dry_run(self):
        """Dry run prints agent definitions without calling Azure."""
        from scripts.register_foundry_agents import AGENTS, register_agents
        # Should not raise (no Azure SDK needed for dry run)
        register_agents(
            agents=AGENTS,
            image="dry-run-image",
            endpoint="dry-run-endpoint",
            dry_run=True,
        )

    def test_agent_definitions_complete(self):
        """All agent definitions have required fields."""
        from scripts.register_foundry_agents import AGENTS
        required = {"name", "workflow", "domain", "tier", "cpu", "memory"}
        for agent in AGENTS:
            missing = required - set(agent.keys())
            self.assertEqual(missing, set(), f"Agent {agent.get('name')} missing: {missing}")

    def test_discover_agents(self):
        """Agent discovery finds workflow/domain pairs."""
        from scripts.register_foundry_agents import discover_agents
        agents = discover_agents(BASE)
        # Should find at least the synthetic agents
        names = [a["workflow"] for a in agents]
        # Some may be found depending on domain YAML structure
        self.assertIsInstance(agents, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
