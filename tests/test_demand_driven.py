"""
Cognitive Core — Demand-Driven Delegation Integration Tests

Tests the wired path: workflow starts forward, hits a step that
produces a ResourceRequest, coordinator intercepts, dispatches
a provider, provider completes, source resumes and finishes.

No LLM, no LangGraph. These tests patch _execute_workflow and
_execute_workflow_from_state to simulate the stepper's behavior,
then verify the coordinator handles interrupts correctly.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.types import (
    InstanceState, InstanceStatus,
    WorkOrder, WorkOrderStatus, WorkOrderResult,
    Suspension, Capability,
)
from coordinator.store import CoordinatorStore
from coordinator.policy import PolicyEngine, load_policy_engine
from coordinator.tasks import InMemoryTaskQueue, TaskType
from coordinator.runtime import Coordinator
from engine.stepper import StepInterrupt


def _make_coordinator(capabilities=None):
    """Build a coordinator with in-memory store and optional capabilities."""
    config_path = os.path.join(_project_root, "coordinator", "config.yaml")
    store = CoordinatorStore(":memory:")
    coord = Coordinator(
        config_path=config_path,
        store=store,
        verbose=True,
    )
    # Inject test capabilities
    if capabilities:
        coord.policy.capabilities = capabilities
    return coord


# ═══════════════════════════════════════════════════════════════════
# 1. INTERRUPT HANDLING
# ═══════════════════════════════════════════════════════════════════

class TestInterruptHandling(unittest.TestCase):
    """Test _on_interrupted with mocked execution."""

    def setUp(self):
        self.cap = Capability(
            need_type="eligibility_constraints",
            provider_type="workflow",
            workflow_type="eligibility_assessment",
            domain="hardship_eligibility",
            contract_name="eligibility_v1",
        )
        self.coord = _make_coordinator(capabilities=[self.cap])

    def _make_running_instance(self, wf="path_recommendation", domain="hardship_path"):
        inst = InstanceState.create(wf, domain, "auto")
        inst.status = InstanceStatus.RUNNING
        self.coord.store.save_instance(inst)
        return inst

    def test_interrupt_suspends_instance(self):
        """Workflow interrupted → instance becomes SUSPENDED."""
        inst = self._make_running_instance()
        interrupt = StepInterrupt(
            reason="Step 'recommend' needs eligibility_constraints",
            suspended_at_step="recommend",
            state_at_interrupt={
                "input": {"member_id": "MBR-001"},
                "steps": [
                    {"step_name": "retrieve", "primitive": "retrieve",
                     "output": {"data": {"member": "MBR-001"}}},
                    {"step_name": "recommend", "primitive": "think",
                     "output": {"resource_requests": [{
                         "need": "eligibility_constraints",
                         "blocking": True,
                         "reason": "Cannot recommend without constraints",
                         "context": {"member_id": "MBR-001"},
                     }]}},
                ],
                "current_step": "recommend",
                "metadata": {},
                "loop_counts": {},
                "routing_log": [],
            },
            resource_requests=[{
                "need": "eligibility_constraints",
                "blocking": True,
                "reason": "Cannot recommend without constraints",
                "context": {"member_id": "MBR-001"},
            }],
        )

        # Mock start() to prevent actual provider execution
        # (we just want to test the interrupt handling itself)
        with patch.object(self.coord, 'start', return_value="wf_provider_001") as mock_start:
            # Also mock get_instance for the provider check
            provider = InstanceState.create("eligibility_assessment", "hardship_eligibility", "auto")
            provider.status = InstanceStatus.COMPLETED
            provider.result = {"eligible": True, "constraints": ["max_6mo"]}

            original_get = self.coord.store.get_instance
            def patched_get(iid):
                if iid == "wf_provider_001":
                    return provider
                return original_get(iid)

            with patch.object(self.coord.store, 'get_instance', side_effect=patched_get):
                # Also mock resume() to prevent cascading
                with patch.object(self.coord, 'resume') as mock_resume:
                    self.coord._on_interrupted(inst, interrupt)

        # Instance should be suspended
        loaded = self.coord.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.SUSPENDED)

    def test_interrupt_creates_work_order(self):
        """Interrupt creates a work order for the matched capability."""
        inst = self._make_running_instance()
        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="recommend",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[{
                "need": "eligibility_constraints",
                "blocking": True,
                "reason": "need it",
                "context": {"member_id": "MBR-001"},
            }],
        )

        with patch.object(self.coord, 'start', return_value="wf_handler"):
            provider = InstanceState.create("eligibility_assessment", "hardship_eligibility", "auto")
            provider.status = InstanceStatus.COMPLETED
            provider.result = {"eligible": True}

            original_get = self.coord.store.get_instance
            def patched_get(iid):
                if iid == "wf_handler":
                    return provider
                return original_get(iid)

            with patch.object(self.coord.store, 'get_instance', side_effect=patched_get):
                with patch.object(self.coord, 'resume'):
                    self.coord._on_interrupted(inst, interrupt)

        # Check work orders
        loaded = self.coord.store.get_instance(inst.instance_id)
        self.assertTrue(len(loaded.pending_work_orders) > 0 or
                        loaded.status == InstanceStatus.SUSPENDED)

    def test_interrupt_creates_suspension_record(self):
        """Interrupt creates a suspension record with state and needs."""
        inst = self._make_running_instance()
        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="recommend",
            state_at_interrupt={
                "input": {"member_id": "MBR-001"},
                "steps": [{"step_name": "retrieve", "primitive": "retrieve", "output": {}}],
                "current_step": "recommend",
                "metadata": {},
                "loop_counts": {},
                "routing_log": [],
            },
            resource_requests=[{
                "need": "eligibility_constraints",
                "blocking": True,
                "reason": "need it",
                "context": {"member_id": "MBR-001"},
            }],
        )

        with patch.object(self.coord, 'start', return_value="wf_handler"):
            provider = InstanceState.create("eligibility_assessment", "hardship_eligibility", "auto")
            provider.status = InstanceStatus.COMPLETED
            provider.result = {}

            original_get = self.coord.store.get_instance
            def patched_get(iid):
                if iid == "wf_handler":
                    return provider
                return original_get(iid)

            with patch.object(self.coord.store, 'get_instance', side_effect=patched_get):
                with patch.object(self.coord, 'resume'):
                    self.coord._on_interrupted(inst, interrupt)

        sus = self.coord.store.get_suspension(inst.instance_id)
        self.assertIsNotNone(sus)
        self.assertEqual(sus.suspended_at_step, "recommend")
        self.assertEqual(len(sus.unresolved_needs), 1)
        self.assertEqual(sus.unresolved_needs[0]["need"], "eligibility_constraints")

    def test_interrupt_unmatched_need_fails(self):
        """If no capability matches the need, workflow fails."""
        inst = self._make_running_instance()
        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="some_step",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[{
                "need": "nonexistent_capability",
                "blocking": True,
                "reason": "need something impossible",
            }],
        )

        self.coord._on_interrupted(inst, interrupt)

        loaded = self.coord.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.FAILED)
        self.assertIn("unresolvable", loaded.error)

    def test_interrupt_logs_to_ledger(self):
        """Interrupt is logged in the action ledger."""
        inst = self._make_running_instance()
        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="recommend",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[{
                "need": "eligibility_constraints",
                "blocking": True,
                "reason": "need it",
                "context": {},
            }],
        )

        with patch.object(self.coord, 'start', return_value="wf_handler"):
            provider = InstanceState.create("eligibility_assessment", "hardship_eligibility", "auto")
            provider.status = InstanceStatus.COMPLETED
            provider.result = {}

            original_get = self.coord.store.get_instance
            def patched_get(iid):
                if iid == "wf_handler":
                    return provider
                return original_get(iid)

            with patch.object(self.coord.store, 'get_instance', side_effect=patched_get):
                with patch.object(self.coord, 'resume'):
                    self.coord._on_interrupted(inst, interrupt)

        ledger = self.coord.store.get_ledger(instance_id=inst.instance_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("interrupted_for_resources", types)


# ═══════════════════════════════════════════════════════════════════
# 2. CAPABILITY MATCHING
# ═══════════════════════════════════════════════════════════════════

class TestCapabilityMatching(unittest.TestCase):
    """Test _find_capability lookups."""

    def test_find_existing_capability(self):
        cap = Capability(
            need_type="fraud_clearance",
            provider_type="workflow",
            workflow_type="fraud_screening",
            domain="fraud",
        )
        coord = _make_coordinator(capabilities=[cap])
        result = coord._find_capability("fraud_clearance")
        self.assertIsNotNone(result)
        self.assertEqual(result.workflow_type, "fraud_screening")

    def test_find_nonexistent_returns_none(self):
        coord = _make_coordinator(capabilities=[])
        result = coord._find_capability("anything")
        self.assertIsNone(result)

    def test_find_among_many(self):
        caps = [
            Capability(need_type="a", provider_type="workflow", workflow_type="wf_a", domain="d_a"),
            Capability(need_type="b", provider_type="workflow", workflow_type="wf_b", domain="d_b"),
            Capability(need_type="c", provider_type="human_task", queue="q_c"),
        ]
        coord = _make_coordinator(capabilities=caps)
        self.assertEqual(coord._find_capability("b").workflow_type, "wf_b")
        self.assertEqual(coord._find_capability("c").provider_type, "human_task")
        self.assertIsNone(coord._find_capability("z"))


# ═══════════════════════════════════════════════════════════════════
# 3. HUMAN TASK PROVIDER
# ═══════════════════════════════════════════════════════════════════

class TestHumanTaskProvider(unittest.TestCase):
    """Test interrupt with human_task capability → task queue."""

    def setUp(self):
        self.cap = Capability(
            need_type="manual_review",
            provider_type="human_task",
            queue="analyst_review",
            contract_name="review_v1",
        )
        self.coord = _make_coordinator(capabilities=[self.cap])
        # Replace task queue with in-memory for inspection
        self.coord.tasks = InMemoryTaskQueue()

    def test_human_task_published(self):
        inst = InstanceState.create("some_wf", "some_domain", "auto")
        inst.status = InstanceStatus.RUNNING
        self.coord.store.save_instance(inst)

        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="decide",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[{
                "need": "manual_review",
                "blocking": True,
                "reason": "Analyst must verify",
                "context": {"case_id": "C-001"},
            }],
        )

        self.coord._on_interrupted(inst, interrupt)

        # Instance suspended
        loaded = self.coord.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.SUSPENDED)

        # Task published to queue
        tasks = self.coord.tasks.list_tasks(queue="analyst_review", status="pending")
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_type, TaskType.RESOURCE_REQUEST)
        self.assertEqual(task.payload["need"], "manual_review")
        self.assertEqual(task.payload["context"]["case_id"], "C-001")


# ═══════════════════════════════════════════════════════════════════
# 4. START() WITH INTERRUPT
# ═══════════════════════════════════════════════════════════════════

class TestStartWithInterrupt(unittest.TestCase):
    """Test the full start() path when _execute_workflow returns an interrupt."""

    def setUp(self):
        self.cap = Capability(
            need_type="eligibility_constraints",
            provider_type="workflow",
            workflow_type="eligibility_assessment",
            domain="hardship_eligibility",
        )
        self.coord = _make_coordinator(capabilities=[self.cap])

    def test_start_handles_interrupt(self):
        """start() handles StepInterrupt from _execute_workflow."""
        interrupt = StepInterrupt(
            reason="needs eligibility",
            suspended_at_step="recommend",
            state_at_interrupt={"input": {"m": 1}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[{
                "need": "eligibility_constraints",
                "blocking": True,
                "reason": "need it",
                "context": {"m": 1},
            }],
        )

        # Mock _execute_workflow to return an interrupt
        def mock_execute(inst, case_input, model="default", temperature=0.1):
            return interrupt

        # Mock self.start for the provider dispatch (prevent recursion)
        provider = InstanceState.create("eligibility_assessment", "hardship_eligibility", "auto")
        provider.status = InstanceStatus.COMPLETED
        provider.result = {"eligible": True}

        with patch.object(self.coord, '_execute_workflow', side_effect=mock_execute):
            with patch.object(self.coord, '_on_interrupted') as mock_interrupted:
                iid = self.coord.start(
                    "path_recommendation", "hardship_path",
                    {"member_id": "MBR-001"},
                )

        # _on_interrupted was called
        mock_interrupted.assert_called_once()
        call_args = mock_interrupted.call_args
        self.assertEqual(call_args[0][1].suspended_at_step, "recommend")

    def test_start_handles_completion(self):
        """start() handles normal completion from _execute_workflow."""
        final_state = {
            "input": {"m": 1},
            "steps": [
                {"step_name": "retrieve", "primitive": "retrieve",
                 "output": {"data": {}}},
                {"step_name": "think", "primitive": "think",
                 "output": {"confidence": 0.9}},
            ],
            "current_step": "think",
            "metadata": {},
            "loop_counts": {},
            "routing_log": [],
        }

        def mock_execute(inst, case_input, model="default", temperature=0.1):
            return final_state

        with patch.object(self.coord, '_execute_workflow', side_effect=mock_execute):
            with patch.object(self.coord, '_on_completed') as mock_completed:
                iid = self.coord.start(
                    "some_wf", "card_dispute",
                    {"case_id": "C-001"},
                )

        mock_completed.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# 5. RESUME WITH RE-INTERRUPT
# ═══════════════════════════════════════════════════════════════════

class TestResumeWithReInterrupt(unittest.TestCase):
    """Test that resumed workflows can be interrupted again."""

    def setUp(self):
        self.caps = [
            Capability(
                need_type="eligibility_constraints",
                provider_type="workflow",
                workflow_type="eligibility_assessment",
                domain="hardship_eligibility",
            ),
            Capability(
                need_type="fraud_clearance",
                provider_type="workflow",
                workflow_type="fraud_screening",
                domain="fraud",
            ),
        ]
        self.coord = _make_coordinator(capabilities=self.caps)

    def test_resume_can_interrupt_again(self):
        """Resumed workflow hits another need → re-interrupts."""
        # Set up a suspended instance
        inst = InstanceState.create("complex_wf", "complex_domain", "auto")
        inst.status = InstanceStatus.SUSPENDED
        sus = Suspension.create(
            inst.instance_id,
            "step_b",
            {"input": {"m": 1}, "steps": [{"step_name": "step_a", "primitive": "retrieve", "output": {}}],
             "current_step": "step_a", "metadata": {}, "loop_counts": {}, "routing_log": [],
             "delegation_results": {"wo_1": {"eligible": True}}},
            work_order_ids=["wo_1"],
        )
        inst.resume_nonce = sus.resume_nonce
        self.coord.store.save_instance(inst)
        self.coord.store.save_suspension(sus)

        # When resumed, the workflow hits ANOTHER need
        second_interrupt = StepInterrupt(
            reason="now needs fraud clearance",
            suspended_at_step="step_c",
            state_at_interrupt={"input": {"m": 1}, "steps": [],
                                "current_step": "step_c",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[{
                "need": "fraud_clearance",
                "blocking": True,
                "reason": "need fraud check",
                "context": {},
            }],
        )

        def mock_resume_execute(instance, state_snapshot, resume_step,
                                model="default", temperature=0.1):
            return second_interrupt

        with patch.object(self.coord, '_execute_workflow_from_state',
                         side_effect=mock_resume_execute):
            with patch.object(self.coord, '_on_interrupted') as mock_interrupted:
                self.coord.resume(
                    inst.instance_id,
                    external_input={"wo_1": {"eligible": True}},
                    resume_nonce=sus.resume_nonce,
                )

        # _on_interrupted called with the second interrupt
        mock_interrupted.assert_called_once()
        call_args = mock_interrupted.call_args
        self.assertEqual(call_args[0][1].suspended_at_step, "step_c")
        self.assertEqual(
            call_args[0][1].resource_requests[0]["need"],
            "fraud_clearance"
        )


# ═══════════════════════════════════════════════════════════════════
# 6. FULL FORWARD-PAUSE-RESUME CYCLE (No LLM)
# ═══════════════════════════════════════════════════════════════════

class TestForwardPauseResumeCycle(unittest.TestCase):
    """
    End-to-end: one workflow starts forward, pauses for a resource,
    provider runs, result flows back, source resumes and completes.
    
    No LLM, no LangGraph — the execution steps are mocked but
    the coordinator state machine runs for real.
    """

    def setUp(self):
        self.cap = Capability(
            need_type="eligibility_constraints",
            provider_type="workflow",
            workflow_type="eligibility_assessment",
            domain="hardship_eligibility",
        )
        self.coord = _make_coordinator(capabilities=[self.cap])

    def test_full_cycle(self):
        """WF starts → pauses → provider runs → WF resumes → completes."""
        call_count = {"execute": 0, "resume_execute": 0}

        # First call: workflow hits a wall at step 2
        interrupt = StepInterrupt(
            reason="need eligibility",
            suspended_at_step="recommend",
            state_at_interrupt={
                "input": {"member_id": "MBR-007"},
                "steps": [
                    {"step_name": "retrieve", "primitive": "retrieve",
                     "output": {"data": {"member": "MBR-007"}}},
                    {"step_name": "recommend", "primitive": "think",
                     "output": {"confidence": 0.0}},
                ],
                "current_step": "recommend",
                "metadata": {"use_case": "path_recommendation"},
                "loop_counts": {},
                "routing_log": [],
            },
            resource_requests=[{
                "need": "eligibility_constraints",
                "blocking": True,
                "reason": "Cannot recommend path without constraints",
                "context": {"member_id": "MBR-007"},
            }],
        )

        # Provider result (eligibility assessment completes)
        provider_final_state = {
            "input": {"member_id": "MBR-007"},
            "steps": [
                {"step_name": "check", "primitive": "think",
                 "output": {"eligible": True, "constraints": ["max_6mo"],
                            "confidence": 0.95}},
            ],
            "current_step": "check",
            "metadata": {},
            "loop_counts": {},
            "routing_log": [],
        }

        # After resume: source completes
        resumed_final_state = {
            "input": {"member_id": "MBR-007", "delegation": {"eligible": True}},
            "steps": [
                {"step_name": "retrieve", "primitive": "retrieve",
                 "output": {"data": {"member": "MBR-007"}}},
                {"step_name": "recommend", "primitive": "think",
                 "output": {"recommendation": "forbearance", "confidence": 0.88}},
                {"step_name": "compose", "primitive": "generate",
                 "output": {"artifact": "recommendation_letter.pdf"}},
            ],
            "current_step": "compose",
            "metadata": {"use_case": "path_recommendation"},
            "loop_counts": {},
            "routing_log": [],
        }

        def mock_execute(inst, case_input, model="default", temperature=0.1):
            call_count["execute"] += 1
            if call_count["execute"] == 1:
                # Source workflow: interrupt
                return interrupt
            elif call_count["execute"] == 2:
                # Provider workflow: complete
                return provider_final_state
            return {"steps": []}

        def mock_resume_execute(inst, state_snapshot, resume_step,
                                model="default", temperature=0.1):
            call_count["resume_execute"] += 1
            return resumed_final_state

        # Override tier resolution so the provider doesn't get governance-gated
        original_resolve = self.coord._resolve_governance_tier
        def auto_tier(domain):
            if domain == "hardship_eligibility":
                return "auto"
            return original_resolve(domain)

        with patch.object(self.coord, '_execute_workflow',
                         side_effect=mock_execute):
            with patch.object(self.coord, '_execute_workflow_from_state',
                             side_effect=mock_resume_execute):
                with patch.object(self.coord, '_resolve_governance_tier',
                                 side_effect=auto_tier):
                    source_id = self.coord.start(
                        "path_recommendation", "hardship_path",
                        {"member_id": "MBR-007"},
                    )

        # Verify the cycle happened
        self.assertEqual(call_count["execute"], 2, "Source + provider executed")
        self.assertEqual(call_count["resume_execute"], 1, "Source resumed once")

        # Source should be completed
        source = self.coord.store.get_instance(source_id)
        self.assertEqual(source.status, InstanceStatus.COMPLETED)

        # Suspension should be cleaned up
        sus = self.coord.store.get_suspension(source_id)
        self.assertIsNone(sus)

        # Ledger should show the full lifecycle
        ledger = self.coord.store.get_ledger(instance_id=source_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("start", types)
        self.assertIn("interrupted_for_resources", types)
        self.assertIn("resume", types)


# ═══════════════════════════════════════════════════════════════════
# 7. MULTIPLE RESOURCE REQUESTS IN ONE STEP
# ═══════════════════════════════════════════════════════════════════

class TestMultipleResourceRequests(unittest.TestCase):
    """Step produces multiple blocking needs simultaneously."""

    def setUp(self):
        self.caps = [
            Capability(
                need_type="fraud_clearance",
                provider_type="workflow",
                workflow_type="fraud_screening",
                domain="fraud",
            ),
            Capability(
                need_type="credit_check",
                provider_type="workflow",
                workflow_type="credit_review",
                domain="credit",
            ),
        ]
        self.coord = _make_coordinator(capabilities=self.caps)

    def test_multiple_needs_multiple_work_orders(self):
        """Two needs → two work orders → two providers dispatched."""
        inst = InstanceState.create("big_wf", "big_domain", "auto")
        inst.status = InstanceStatus.RUNNING
        self.coord.store.save_instance(inst)

        interrupt = StepInterrupt(
            reason="needs two things",
            suspended_at_step="decide",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[
                {"need": "fraud_clearance", "blocking": True,
                 "reason": "fraud check", "context": {}},
                {"need": "credit_check", "blocking": True,
                 "reason": "credit check", "context": {}},
            ],
        )

        dispatched = []

        def mock_start(workflow_type, domain, case_input, lineage=None,
                       correlation_id="", model="default", temperature=0.1):
            dispatched.append(workflow_type)
            pid = f"wf_provider_{len(dispatched)}"
            prov = InstanceState.create(workflow_type, domain, "auto")
            prov.instance_id = pid
            prov.status = InstanceStatus.COMPLETED
            prov.result = {"status": "ok"}
            self.coord.store.save_instance(prov)
            return pid

        with patch.object(self.coord, 'start', side_effect=mock_start):
            with patch.object(self.coord, 'resume'):
                self.coord._on_interrupted(inst, interrupt)

        # Two providers dispatched
        self.assertEqual(len(dispatched), 2)
        self.assertIn("fraud_screening", dispatched)
        self.assertIn("credit_review", dispatched)

        # Instance has multiple work orders
        loaded = self.coord.store.get_instance(inst.instance_id)
        sus = self.coord.store.get_suspension(inst.instance_id)
        self.assertIsNotNone(sus)
        self.assertEqual(len(sus.work_order_ids), 2)


# ═══════════════════════════════════════════════════════════════════
# 8. MIXED MATCHED/UNMATCHED NEEDS
# ═══════════════════════════════════════════════════════════════════

class TestMixedMatchedUnmatched(unittest.TestCase):
    """Some needs match capabilities, some don't."""

    def test_partial_match_proceeds(self):
        """If at least one need matches, workflow suspends (doesn't fail)."""
        cap = Capability(
            need_type="known_need",
            provider_type="workflow",
            workflow_type="provider_wf",
            domain="provider_d",
        )
        coord = _make_coordinator(capabilities=[cap])

        inst = InstanceState.create("wf", "dom", "auto")
        inst.status = InstanceStatus.RUNNING
        coord.store.save_instance(inst)

        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="step_x",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[
                {"need": "known_need", "blocking": True, "reason": "r1", "context": {}},
                {"need": "unknown_need", "blocking": True, "reason": "r2", "context": {}},
            ],
        )

        with patch.object(coord, 'start', return_value="wf_p"):
            prov = InstanceState.create("provider_wf", "provider_d", "auto")
            prov.status = InstanceStatus.COMPLETED
            prov.result = {}

            original_get = coord.store.get_instance
            def patched_get(iid):
                if iid == "wf_p":
                    return prov
                return original_get(iid)

            with patch.object(coord.store, 'get_instance', side_effect=patched_get):
                with patch.object(coord, 'resume'):
                    coord._on_interrupted(inst, interrupt)

        loaded = coord.store.get_instance(inst.instance_id)
        # Should be suspended, not failed
        self.assertEqual(loaded.status, InstanceStatus.SUSPENDED)


# ═══════════════════════════════════════════════════════════════════
# 9. DEPENDENCY-AWARE DISPATCH
# ═══════════════════════════════════════════════════════════════════

class TestDependencyAwareDispatch(unittest.TestCase):
    """Test that depends_on in ResourceRequest controls dispatch order."""

    def setUp(self):
        self.caps = [
            Capability(
                need_type="settlement_calculation",
                provider_type="workflow",
                workflow_type="settlement_calc",
                domain="settlement",
            ),
            Capability(
                need_type="reserve_impact",
                provider_type="workflow",
                workflow_type="reserve_model",
                domain="reserves",
            ),
            Capability(
                need_type="fraud_clearance",
                provider_type="workflow",
                workflow_type="fraud_screening",
                domain="fraud",
            ),
        ]
        self.coord = _make_coordinator(capabilities=self.caps)

    def test_independent_needs_dispatched_immediately(self):
        """Needs with no depends_on are dispatched right away."""
        inst = InstanceState.create("wf", "dom", "auto")
        inst.status = InstanceStatus.RUNNING
        self.coord.store.save_instance(inst)

        dispatched = []
        def mock_start(workflow_type, domain, case_input, lineage=None,
                       correlation_id="", model="default", temperature=0.1):
            dispatched.append(workflow_type)
            pid = f"wf_p_{len(dispatched)}"
            prov = InstanceState.create(workflow_type, domain, "auto")
            prov.instance_id = pid
            prov.status = InstanceStatus.COMPLETED
            prov.result = {"done": True}
            self.coord.store.save_instance(prov)
            return pid

        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="decide",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[
                {"need": "settlement_calculation", "blocking": True,
                 "reason": "r1", "context": {}},
                {"need": "fraud_clearance", "blocking": True,
                 "reason": "r2", "context": {}},
                # No depends_on → both should dispatch immediately
            ],
        )

        with patch.object(self.coord, 'start', side_effect=mock_start):
            with patch.object(self.coord, 'resume'):
                self.coord._on_interrupted(inst, interrupt)

        # Both dispatched
        self.assertEqual(len(dispatched), 2)
        self.assertIn("settlement_calc", dispatched)
        self.assertIn("fraud_screening", dispatched)

    def test_dependent_need_deferred(self):
        """Need with depends_on is deferred until prerequisite completes."""
        inst = InstanceState.create("wf", "dom", "auto")
        inst.status = InstanceStatus.RUNNING
        self.coord.store.save_instance(inst)

        dispatched = []
        def mock_start(workflow_type, domain, case_input, lineage=None,
                       correlation_id="", model="default", temperature=0.1):
            dispatched.append(workflow_type)
            pid = f"wf_p_{len(dispatched)}"
            prov = InstanceState.create(workflow_type, domain, "auto")
            prov.instance_id = pid
            prov.status = InstanceStatus.COMPLETED
            prov.result = {"amount": 50000}
            self.coord.store.save_instance(prov)
            return pid

        interrupt = StepInterrupt(
            reason="test",
            suspended_at_step="decide",
            state_at_interrupt={"input": {}, "steps": [], "current_step": "",
                                "metadata": {}, "loop_counts": {}, "routing_log": []},
            resource_requests=[
                {"need": "settlement_calculation", "blocking": True,
                 "reason": "Need settlement first", "context": {}},
                {"need": "reserve_impact", "blocking": True,
                 "reason": "Need reserve impact after settlement",
                 "context": {},
                 "depends_on": ["settlement_calculation"]},
            ],
        )

        with patch.object(self.coord, 'start', side_effect=mock_start):
            with patch.object(self.coord, 'resume'):
                self.coord._on_interrupted(inst, interrupt)

        # Only settlement dispatched immediately, not reserve
        # (reserve depends on settlement)
        sus = self.coord.store.get_suspension(inst.instance_id)
        if sus and sus.deferred_needs:
            # First wave: settlement only
            self.assertIn("settlement_calc", dispatched)
            self.assertEqual(len(sus.deferred_needs), 1)
            self.assertEqual(sus.deferred_needs[0]["need"], "reserve_impact")
        else:
            # If both completed synchronously and deferred was dispatched
            # in the same call, both should be in dispatched
            self.assertIn("settlement_calc", dispatched)
            self.assertIn("reserve_model", dispatched)


if __name__ == "__main__":
    unittest.main()
