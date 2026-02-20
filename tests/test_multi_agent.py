"""
Cognitive Core — Multi-Agent Coordinator Integration Test

Exercises the FULL multi-agent path:
  claim_intake (orchestrator, tier=gate)
    → damage_assessment (delegated, tier=auto)
    → fraud_screening (delegated, tier=spot_check)

Uses mock LLM responses so we can run without network.
Tests: delegation fan-out, governance tiers per agent,
       audit trail across correlation chain, work orders,
       contract validation, and coordinator store consistency.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Setup paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from coordinator.runtime import Coordinator
from coordinator.types import (
    InstanceState, InstanceStatus, WorkOrder, WorkOrderStatus,
    GovernanceTier,
)
from coordinator.policy import PolicyEngine, load_policy_engine
from coordinator.store import CoordinatorStore
from coordinator.tasks import InMemoryTaskQueue


# ─── Mock LLM Responses ─────────────────────────────────────────────
# These are deterministic responses that the mock LLM returns
# based on step name. They match the YAML domain specs exactly.

CLAIM_INTAKE_RESPONSES = {
    "gather_claim_data": json.dumps({
        "data": {
            "get_policy": {
                "policy_id": "POL-2026-001",
                "status": "active",
                "coverage_type": "comprehensive,collision",
                "tenure_months": 36,
                "prior_claims": 1,
            },
            "get_claim": {
                "claim_id": "CLM-2026-00847",
                "amount": 12500,
                "incident_date": "2026-02-10",
                "claim_type_hint": "physical_damage",
                "flags": ["high_value"],
                "description": "Rear-end collision on I-95. Significant bumper and trunk damage.",
            },
            "get_claimant": {
                "name": "Test Claimant",
                "member_since": "2023-02-01",
                "age": 42,
            },
        }
    }),
    "classify_claim_type": json.dumps({
        "category": "physical_damage",
        "confidence": 1.0,
        "reasoning": "claim_type_hint = physical_damage",
    }),
    "check_eligibility": json.dumps({
        "conforms": True,
        "violations": [],
        "reasoning": "All rules pass: active policy, comprehensive covers physical_damage, amount in range, date not future.",
    }),
    "assess_risk": json.dumps({
        "risk_score": 55,
        "recommendation": "standard_review",
        "reasoning": "Base 0 + 20 (amount > 10K) + 15 (amount > 5K) + 0 (prior_claims < 3) + 20 (high_value flag) = 55",
    }),
    "generate_decision": json.dumps({
        "claim_id": "CLM-2026-00847",
        "decision": "approve_with_review",
        "amount": 12500,
        "risk_score": 55,
        "requires_damage_assessment": True,
        "requires_fraud_screening": True,
    }),
}

DAMAGE_ASSESSMENT_RESPONSES = {
    "gather_damage_data": json.dumps({
        "data": {
            "get_claim": {"claim_id": "CLM-2026-00847", "amount": 12500, "description": "Rear-end collision"},
            "get_photos": {"count": 5, "descriptions": ["front bumper", "trunk", "frame", "interior", "undercarriage"]},
            "get_estimate": {"provided": True, "amount": 11200, "vendor": "AutoBody Pro"},
            "get_police_report": {"filed": True, "report_number": "PR-2026-4421"},
            "get_policy": {"policy_id": "POL-2026-001", "coverage_type": "comprehensive,collision"},
        }
    }),
    "classify_damage_severity": json.dumps({
        "category": "major",
        "confidence": 1.0,
        "reasoning": "Repair cost $11,200 is in range $10,001-$50,000 → major",
    }),
    "verify_documentation": json.dumps({
        "conforms": True,
        "violations": [],
        "reasoning": "D1: 5 photos >= 3 ✓, D2: estimate provided ✓, D3: police report filed for major ✓",
    }),
    "generate_assessment": json.dumps({
        "claim_id": "CLM-2026-00847",
        "damage_grade": "major",
        "repair_cost": 11200,
        "documentation_complete": True,
        "missing_documentation": [],
        "recommendation": "approve_repair",
    }),
}

FRAUD_SCREENING_RESPONSES = {
    "gather_fraud_data": json.dumps({
        "data": {
            "get_claim": {"claim_id": "CLM-2026-00847", "amount": 12500, "incident_date": "2026-02-10"},
            "get_claim_history": {"claims": [{"date": "2025-01-15", "amount": 3200}]},
            "get_flags": {"count": 1, "types": ["high_value"]},
            "get_policy": {"policy_id": "POL-2026-001", "days_active": 1095, "coverage_type": "comprehensive,collision"},
        }
    }),
    "classify_fraud_risk": json.dumps({
        "category": "medium_risk",
        "confidence": 1.0,
        "reasoning": "Flag count = 1 → medium_risk",
    }),
    "investigate_patterns": json.dumps({
        "finding": "minor_concerns",
        "evidence_flags": ["amount_escalation"],
        "recommendation": "flag_for_monitoring",
        "reasoning": "P3 triggered: $12,500 > 2x $3,200 average. No other patterns.",
    }),
    "generate_screening_result": json.dumps({
        "claim_id": "CLM-2026-00847",
        "fraud_risk": "medium_risk",
        "finding": "minor_concerns",
        "evidence_flags": ["amount_escalation"],
        "recommendation": "flag_for_monitoring",
    }),
}

ALL_RESPONSES = {
    "claim_intake": CLAIM_INTAKE_RESPONSES,
    "damage_assessment": DAMAGE_ASSESSMENT_RESPONSES,
    "fraud_screening": FRAUD_SCREENING_RESPONSES,
}


def mock_llm_invoke(prompt, **kwargs):
    """Mock LLM that returns deterministic responses based on step context."""
    # Extract step name from prompt (the engine includes it)
    for wf_name, responses in ALL_RESPONSES.items():
        for step_name, response in responses.items():
            if step_name in str(prompt):
                result = MagicMock()
                result.content = response
                return result
    # Fallback
    result = MagicMock()
    result.content = json.dumps({"status": "ok", "note": "fallback mock response"})
    return result


class TestMultiAgentCoordinator(unittest.TestCase):
    """Integration tests for multi-agent claim processing."""

    def setUp(self):
        """Set up coordinator with test config."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_coordinator.db")

        # Build a coordinator config that includes delegation policies
        # for claim_intake → damage_assessment + fraud_screening
        self.coord_config = {
            "workflow_dir": os.path.join(BASE, "workflows"),
            "domain_dir": os.path.join(BASE, "domains"),
            "case_dir": os.path.join(BASE, "cases"),
            "governance_tiers": {
                "auto": {"hitl": "none", "sample_rate": 0.0},
                "spot_check": {"hitl": "post_completion", "sample_rate": 0.10, "queue": "qa_review", "sla": 7200},
                "gate": {"hitl": "before_act", "queue": "specialist_review", "sla": 14400},
                "hold": {"hitl": "before_finalize", "queue": "compliance_review", "sla": 172800},
            },
            "overrides": {},
            "quality_gates": {
                "min_confidence": 0.5,
                "primitive_floors": {"classify": 0.6, "investigate": 0.5},
                "escalation_tier": "gate",
                "escalation_queue": "quality_review",
                "exempt_domains": [],
            },
            "delegations": [
                {
                    "name": "claim_to_damage_assessment",
                    "mode": "fire_and_forget",
                    "conditions": [
                        {
                            "domain": "synthetic_claim",
                            "selector": "last_classify",
                            "field": "category",
                            "operator": "eq",
                            "value": "physical_damage",
                        }
                    ],
                    "target_workflow": "damage_assessment",
                    "target_domain": "synthetic_damage",
                    "contract": "damage_assessment_v1",
                    "sla": 3600,
                    "inputs": {
                        "claim_id": "${source.last_retrieve.data.get_claim.claim_id}",
                        "policy_id": "${source.last_retrieve.data.get_policy.policy_id}",
                    },
                },
                {
                    "name": "claim_to_fraud_screening",
                    "mode": "fire_and_forget",
                    "conditions": [
                        {
                            "domain": "synthetic_claim",
                            "selector": "last_retrieve",
                            "field": "data.get_claim.amount",
                            "operator": "gte",
                            "value": 5000,
                        }
                    ],
                    "target_workflow": "fraud_screening",
                    "target_domain": "synthetic_fraud",
                    "contract": "fraud_screening_v1",
                    "sla": 3600,
                    "inputs": {
                        "claim_id": "${source.last_retrieve.data.get_claim.claim_id}",
                        "amount": "${source.last_retrieve.data.get_claim.amount}",
                    },
                },
            ],
            "contracts": {
                "damage_assessment_v1": {
                    "version": 1,
                    "request": [
                        {"name": "claim_id", "type": "string", "required": True},
                    ],
                    "response": [
                        {"name": "damage_grade", "type": "string", "required": True},
                    ],
                },
                "fraud_screening_v1": {
                    "version": 1,
                    "request": [
                        {"name": "claim_id", "type": "string", "required": True},
                    ],
                    "response": [
                        {"name": "fraud_risk", "type": "string", "required": True},
                    ],
                },
            },
            "capabilities": [],
        }

    def _make_coordinator(self):
        """Create a coordinator with mock LLM."""
        return Coordinator(
            config=self.coord_config,
            db_path=self.db_path,
            verbose=True,
        )

    def _case_input(self):
        """Build test case input with embedded tool data."""
        return {
            "case_id": "CLM-2026-00847",
            "description": "Rear-end collision, $12,500 claim",
            # Tool data for retrieve steps (case registry pattern)
            "get_policy": {
                "policy_id": "POL-2026-001",
                "status": "active",
                "coverage_type": "comprehensive,collision",
                "tenure_months": 36,
                "prior_claims": 1,
            },
            "get_claim": {
                "claim_id": "CLM-2026-00847",
                "amount": 12500,
                "incident_date": "2026-02-10",
                "claim_type_hint": "physical_damage",
                "flags": ["high_value"],
                "description": "Rear-end collision on I-95.",
            },
            "get_claimant": {
                "name": "Test Claimant",
                "member_since": "2023-02-01",
                "age": 42,
            },
            # Damage assessment tool data
            "get_photos": {"count": 5, "descriptions": ["front", "trunk", "frame", "interior", "undercarriage"]},
            "get_estimate": {"provided": True, "amount": 11200, "vendor": "AutoBody Pro"},
            "get_police_report": {"filed": True, "report_number": "PR-2026-4421"},
            # Fraud screening tool data
            "get_claim_history": {"claims": [{"date": "2025-01-15", "amount": 3200}]},
            "get_flags": {"count": 1, "types": ["high_value"]},
        }

    # ─── Test: Policy Engine Delegation Evaluation ───────────────────

    def test_policy_evaluates_delegations(self):
        """Policy engine correctly identifies both delegations from claim output."""
        policy = load_policy_engine(self.coord_config)

        # Simulate claim_intake workflow output
        workflow_output = {
            "steps": [
                {
                    "primitive": "retrieve",
                    "step_name": "gather_claim_data",
                    "output": json.loads(CLAIM_INTAKE_RESPONSES["gather_claim_data"]),
                },
                {
                    "primitive": "classify",
                    "step_name": "classify_claim_type",
                    "output": json.loads(CLAIM_INTAKE_RESPONSES["classify_claim_type"]),
                },
            ],
        }

        delegations = policy.evaluate_delegations(
            domain="synthetic_claim",
            workflow_output=workflow_output,
        )

        self.assertEqual(len(delegations), 2, f"Expected 2 delegations, got {len(delegations)}")

        names = {d.policy_name for d in delegations}
        self.assertIn("claim_to_damage_assessment", names)
        self.assertIn("claim_to_fraud_screening", names)

        # Check input resolution
        damage_deleg = next(d for d in delegations if d.policy_name == "claim_to_damage_assessment")
        self.assertEqual(damage_deleg.target_workflow, "damage_assessment")
        self.assertEqual(damage_deleg.target_domain, "synthetic_damage")
        self.assertEqual(damage_deleg.inputs.get("claim_id"), "CLM-2026-00847")

        fraud_deleg = next(d for d in delegations if d.policy_name == "claim_to_fraud_screening")
        self.assertEqual(fraud_deleg.target_workflow, "fraud_screening")
        self.assertEqual(fraud_deleg.target_domain, "synthetic_fraud")

    # ─── Test: Governance Tier Resolution Per Domain ─────────────────

    def test_governance_tiers_per_domain(self):
        """Each domain resolves to its declared governance tier."""
        coord = self._make_coordinator()

        tier_claim = coord._resolve_governance_tier("synthetic_claim")
        tier_damage = coord._resolve_governance_tier("synthetic_damage")
        tier_fraud = coord._resolve_governance_tier("synthetic_fraud")

        self.assertEqual(tier_claim, "gate")
        self.assertEqual(tier_damage, "auto")
        self.assertEqual(tier_fraud, "spot_check")

    # ─── Test: Contract Validation ───────────────────────────────────

    def test_contract_validation(self):
        """Contracts validate inputs and catch missing fields."""
        policy = load_policy_engine(self.coord_config)

        # Valid
        errors = policy.validate_work_order_inputs(
            "damage_assessment_v1",
            {"claim_id": "CLM-001"},
        )
        self.assertEqual(errors, [])

        # Missing required field
        errors = policy.validate_work_order_inputs(
            "damage_assessment_v1",
            {},
        )
        self.assertTrue(len(errors) > 0)
        self.assertIn("claim_id", errors[0])

    # ─── Test: Store Persistence ─────────────────────────────────────

    def test_store_instance_lifecycle(self):
        """Coordinator store correctly persists instance state."""
        store = CoordinatorStore(db_path=self.db_path)

        instance = InstanceState.create(
            workflow_type="claim_intake",
            domain="synthetic_claim",
            governance_tier="gate",
        )
        store.save_instance(instance)

        loaded = store.get_instance(instance.instance_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.workflow_type, "claim_intake")
        self.assertEqual(loaded.governance_tier, "gate")
        self.assertEqual(loaded.status, InstanceStatus.CREATED)

    def test_store_work_order_lifecycle(self):
        """Work orders persist and update correctly."""
        store = CoordinatorStore(db_path=self.db_path)

        wo = WorkOrder.create(
            requester_instance_id="wf_test123",
            correlation_id="wf_test123",
            contract_name="damage_assessment_v1",
            inputs={"claim_id": "CLM-001"},
            sla_seconds=3600,
        )
        store.save_work_order(wo)

        loaded = store.get_work_order(wo.work_order_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.contract_name, "damage_assessment_v1")
        self.assertEqual(loaded.status, WorkOrderStatus.CREATED)

    def test_store_action_ledger(self):
        """Action ledger records entries with idempotency."""
        store = CoordinatorStore(db_path=self.db_path)

        store.log_action(
            instance_id="wf_test123",
            correlation_id="wf_test123",
            action_type="start",
            details={"workflow": "claim_intake"},
            idempotency_key="start:wf_test123:123",
        )

        ledger = store.get_ledger(instance_id="wf_test123")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["action_type"], "start")

        # Idempotent: same key doesn't duplicate
        store.log_action(
            instance_id="wf_test123",
            correlation_id="wf_test123",
            action_type="start",
            details={"workflow": "claim_intake"},
            idempotency_key="start:wf_test123:123",
        )
        ledger = store.get_ledger(instance_id="wf_test123")
        self.assertEqual(len(ledger), 1, "Idempotency key should prevent duplicate")

    def test_correlation_chain_query(self):
        """Correlation chain links all instances in a delegation tree."""
        store = CoordinatorStore(db_path=self.db_path)

        # Create parent + 2 children with same correlation_id
        corr_id = "wf_root123"
        for wf, domain, tier in [
            ("claim_intake", "synthetic_claim", "gate"),
            ("damage_assessment", "synthetic_damage", "auto"),
            ("fraud_screening", "synthetic_fraud", "spot_check"),
        ]:
            inst = InstanceState.create(
                workflow_type=wf, domain=domain, governance_tier=tier,
                correlation_id=corr_id,
            )
            store.save_instance(inst)

        chain = store.list_instances(correlation_id=corr_id)
        self.assertEqual(len(chain), 3)
        wf_types = {i.workflow_type for i in chain}
        self.assertEqual(wf_types, {"claim_intake", "damage_assessment", "fraud_screening"})

    # ─── Test: Delegation Depth Guard ────────────────────────────────

    def test_delegation_depth_guard(self):
        """Coordinator rejects excessively deep delegation chains."""
        from coordinator.runtime import DelegationDepthExceeded
        coord = self._make_coordinator()

        deep_lineage = [f"claim_intake:wf_{i}" for i in range(20)]
        with self.assertRaises(DelegationDepthExceeded):
            coord.start(
                workflow_type="claim_intake",
                domain="synthetic_claim",
                case_input=self._case_input(),
                lineage=deep_lineage,
            )

    # ─── Test: Quality Gate Escalation ───────────────────────────────

    def test_quality_gate_escalates_low_confidence(self):
        """Quality gate upgrades governance when confidence is below threshold."""
        coord = self._make_coordinator()

        # Simulate a low-confidence classify output
        final_state = {
            "steps": [
                {"primitive": "classify", "step_name": "classify_claim_type",
                 "output": {"category": "physical_damage", "confidence": 0.45}},
            ],
        }

        decision = coord._evaluate_quality_gate(
            InstanceState.create("claim_intake", "synthetic_claim", "auto"),
            final_state,
        )

        # Should escalate: classify confidence 0.45 < floor 0.6
        self.assertIsNotNone(decision)
        self.assertEqual(decision["escalation_tier"], "gate")

    def test_quality_gate_passes_high_confidence(self):
        """Quality gate does not escalate when confidence is above threshold."""
        coord = self._make_coordinator()

        final_state = {
            "steps": [
                {"primitive": "classify", "step_name": "classify_claim_type",
                 "output": {"category": "physical_damage", "confidence": 0.95}},
            ],
        }

        decision = coord._evaluate_quality_gate(
            InstanceState.create("claim_intake", "synthetic_claim", "auto"),
            final_state,
        )

        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMultiAgentEndToEnd(unittest.TestCase):
    """End-to-end: coordinator.start() through full multi-agent flow with mocked workflow execution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "e2e_coordinator.db")
        self.config_path = os.path.join(BASE, "coordinator", "config.yaml")

        self.case_input = {
            "case_id": "CLM-2026-00847",
            "get_policy": {"policy_id": "POL-2026-001", "status": "active",
                          "coverage_type": "comprehensive,collision", "tenure_months": 36, "prior_claims": 1},
            "get_claim": {"claim_id": "CLM-2026-00847", "amount": 12500, "incident_date": "2026-02-10",
                         "claim_type_hint": "physical_damage", "flags": ["high_value"],
                         "description": "Rear-end collision on I-95."},
            "get_claimant": {"name": "Test Claimant", "member_since": "2023-02-01", "age": 42},
            "get_photos": {"count": 5, "descriptions": ["front", "trunk", "frame", "interior", "undercarriage"]},
            "get_estimate": {"provided": True, "amount": 11200, "vendor": "AutoBody Pro"},
            "get_police_report": {"filed": True, "report_number": "PR-2026-4421"},
            "get_claim_history": {"claims": [{"date": "2025-01-15", "amount": 3200}]},
            "get_flags": {"count": 1, "types": ["high_value"]},
        }

    def _mock_workflow_output(self, workflow_type):
        """Return deterministic workflow output based on workflow type."""
        if workflow_type == "claim_intake":
            return {
                "steps": [
                    {"primitive": "retrieve", "step_name": "gather_claim_data",
                     "output": json.loads(CLAIM_INTAKE_RESPONSES["gather_claim_data"])},
                    {"primitive": "classify", "step_name": "classify_claim_type",
                     "output": json.loads(CLAIM_INTAKE_RESPONSES["classify_claim_type"])},
                    {"primitive": "verify", "step_name": "check_eligibility",
                     "output": json.loads(CLAIM_INTAKE_RESPONSES["check_eligibility"])},
                    {"primitive": "think", "step_name": "assess_risk",
                     "output": json.loads(CLAIM_INTAKE_RESPONSES["assess_risk"])},
                    {"primitive": "generate", "step_name": "generate_decision",
                     "output": json.loads(CLAIM_INTAKE_RESPONSES["generate_decision"])},
                ],
                "input": self.case_input,
            }
        elif workflow_type == "damage_assessment":
            return {
                "steps": [
                    {"primitive": "retrieve", "step_name": "gather_damage_data",
                     "output": json.loads(DAMAGE_ASSESSMENT_RESPONSES["gather_damage_data"])},
                    {"primitive": "classify", "step_name": "classify_damage_severity",
                     "output": json.loads(DAMAGE_ASSESSMENT_RESPONSES["classify_damage_severity"])},
                    {"primitive": "verify", "step_name": "verify_documentation",
                     "output": json.loads(DAMAGE_ASSESSMENT_RESPONSES["verify_documentation"])},
                    {"primitive": "generate", "step_name": "generate_assessment",
                     "output": json.loads(DAMAGE_ASSESSMENT_RESPONSES["generate_assessment"])},
                ],
                "input": self.case_input,
            }
        elif workflow_type == "fraud_screening":
            return {
                "steps": [
                    {"primitive": "retrieve", "step_name": "gather_fraud_data",
                     "output": json.loads(FRAUD_SCREENING_RESPONSES["gather_fraud_data"])},
                    {"primitive": "classify", "step_name": "classify_fraud_risk",
                     "output": json.loads(FRAUD_SCREENING_RESPONSES["classify_fraud_risk"])},
                    {"primitive": "investigate", "step_name": "investigate_patterns",
                     "output": json.loads(FRAUD_SCREENING_RESPONSES["investigate_patterns"])},
                    {"primitive": "generate", "step_name": "generate_screening_result",
                     "output": json.loads(FRAUD_SCREENING_RESPONSES["generate_screening_result"])},
                ],
                "input": self.case_input,
            }
        return {"steps": [], "input": {}}

    def test_e2e_claim_intake_gate_approve_delegations(self):
        """Full E2E: claim_intake → gate suspension → approve → 2 delegations fire."""
        coord = Coordinator(config_path=self.config_path, db_path=self.db_path, verbose=True)

        # Mock _execute_workflow to return deterministic output
        original_execute = coord._execute_workflow
        def mock_execute(instance, case_input, model="default", temperature=0.1):
            return self._mock_workflow_output(instance.workflow_type)
        coord._execute_workflow = mock_execute

        # Start claim_intake
        instance_id = coord.start(
            workflow_type="claim_intake",
            domain="synthetic_claim",
            case_input=self.case_input,
        )

        instance = coord.get_instance(instance_id)
        print(f"\n{'='*60}")
        print(f"claim_intake: {instance_id} → {instance.status}")

        # gate tier, no Act step → governance evaluates as gate → suspend
        self.assertEqual(instance.status, InstanceStatus.SUSPENDED,
                        f"Gate tier should suspend, got {instance.status}")

        # Verify pending approval task
        pending = coord.list_pending_approvals()
        self.assertTrue(len(pending) > 0, "Should have pending approval")
        print(f"Pending approvals: {len(pending)}")

        # Approve
        coord.approve(instance_id, approver="reviewer_01")

        instance = coord.get_instance(instance_id)
        print(f"After approve: {instance.status}")
        self.assertEqual(instance.status, InstanceStatus.COMPLETED)

        # Check delegations fired
        chain = coord.get_correlation_chain(instance.correlation_id)
        wf_types = {i.workflow_type for i in chain}
        print(f"\nCorrelation chain ({len(chain)} instances):")
        for inst in chain:
            print(f"  {inst.instance_id}: {inst.workflow_type}/{inst.domain} [{inst.status}] tier={inst.governance_tier}")

        self.assertIn("damage_assessment", wf_types, "Should have delegated to damage_assessment")
        self.assertIn("fraud_screening", wf_types, "Should have delegated to fraud_screening")
        self.assertEqual(len(chain), 3, f"Expected 3 instances (parent + 2 delegations), got {len(chain)}")

        # Verify governance tiers per agent
        for inst in chain:
            if inst.workflow_type == "claim_intake":
                self.assertEqual(inst.governance_tier, "gate")
            elif inst.workflow_type == "damage_assessment":
                self.assertEqual(inst.governance_tier, "auto")
                self.assertEqual(inst.status, InstanceStatus.COMPLETED, "Auto tier should complete")
            elif inst.workflow_type == "fraud_screening":
                self.assertEqual(inst.governance_tier, "spot_check")
                self.assertEqual(inst.status, InstanceStatus.COMPLETED, "spot_check (0% sample) should complete")

        # Verify ledger has full audit trail across all instances
        full_ledger = coord.get_ledger(correlation_id=instance.correlation_id)
        action_types = [e["action_type"] for e in full_ledger]
        print(f"\nFull ledger ({len(full_ledger)} entries):")
        for entry in full_ledger:
            det = entry.get("details", {})
            if isinstance(det, str):
                try: det = json.loads(det)
                except: pass
            det_str = json.dumps(det)[:80] if isinstance(det, dict) else str(det)[:80]
            print(f"  [{entry['action_type']:30s}] {det_str}")

        self.assertIn("start", action_types)
        self.assertIn("execution_finished", action_types)
        self.assertIn("governance_evaluation", action_types)
        self.assertIn("delegation_evaluation", action_types)
        self.assertIn("delegation_dispatched", action_types)

        # Count delegation dispatches
        deleg_dispatches = [e for e in full_ledger if e["action_type"] == "delegation_dispatched"]
        self.assertEqual(len(deleg_dispatches), 2, f"Expected 2 delegation dispatches, got {len(deleg_dispatches)}")

        print(f"\n✅ PASS: 3 agents, 3 tiers, 2 delegations, full audit trail")
        print(f"{'='*60}")

    def test_e2e_damage_assessment_auto_completes(self):
        """damage_assessment with auto tier completes without HITL."""
        coord = Coordinator(config_path=self.config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: self._mock_workflow_output(inst.workflow_type)

        instance_id = coord.start(
            workflow_type="damage_assessment",
            domain="synthetic_damage",
            case_input=self.case_input,
        )

        instance = coord.get_instance(instance_id)
        self.assertEqual(instance.status, InstanceStatus.COMPLETED)
        self.assertEqual(instance.governance_tier, "auto")
        print(f"\n✅ damage_assessment: auto tier → completed, {instance.step_count} steps")

    def test_e2e_fraud_screening_spot_check_completes(self):
        """fraud_screening with spot_check (0% sample) completes without HITL."""
        coord = Coordinator(config_path=self.config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: self._mock_workflow_output(inst.workflow_type)

        instance_id = coord.start(
            workflow_type="fraud_screening",
            domain="synthetic_fraud",
            case_input=self.case_input,
        )

        instance = coord.get_instance(instance_id)
        self.assertEqual(instance.status, InstanceStatus.COMPLETED)
        self.assertEqual(instance.governance_tier, "spot_check")
        print(f"\n✅ fraud_screening: spot_check → completed, {instance.step_count} steps")

    def test_e2e_lineage_tracks_delegation_chain(self):
        """Delegated instances carry lineage from parent."""
        coord = Coordinator(config_path=self.config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: self._mock_workflow_output(inst.workflow_type)

        instance_id = coord.start(
            workflow_type="claim_intake",
            domain="synthetic_claim",
            case_input=self.case_input,
        )

        # Approve gate
        coord.approve(instance_id, approver="reviewer_01")

        chain = coord.get_correlation_chain(
            coord.get_instance(instance_id).correlation_id
        )

        for inst in chain:
            if inst.workflow_type != "claim_intake":
                # Delegated instances should have parent in lineage
                self.assertTrue(
                    len(inst.lineage) > 0,
                    f"{inst.workflow_type} should have non-empty lineage"
                )
                print(f"  {inst.workflow_type} lineage: {inst.lineage}")

    def test_e2e_work_orders_created_for_delegations(self):
        """Work orders are created and linked for each delegation."""
        coord = Coordinator(config_path=self.config_path, db_path=self.db_path, verbose=True)
        coord._execute_workflow = lambda inst, ci, model="default", temperature=0.1: self._mock_workflow_output(inst.workflow_type)

        instance_id = coord.start(
            workflow_type="claim_intake",
            domain="synthetic_claim",
            case_input=self.case_input,
        )
        coord.approve(instance_id, approver="reviewer_01")

        # Check work orders
        work_orders = coord.get_work_orders(instance_id)
        print(f"\nWork orders for {instance_id}: {len(work_orders)}")
        for wo in work_orders:
            print(f"  {wo.work_order_id}: {wo.contract_name} → {wo.handler_workflow_type}/{wo.handler_domain} [{wo.status}]")

        self.assertEqual(len(work_orders), 2, f"Expected 2 work orders, got {len(work_orders)}")
        contracts = {wo.contract_name for wo in work_orders}
        self.assertIn("damage_assessment_v1", contracts)
        self.assertIn("fraud_screening_v1", contracts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
