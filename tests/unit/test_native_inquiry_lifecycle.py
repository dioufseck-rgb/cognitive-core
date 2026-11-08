"""Native Coordinator persistence and human-resource integration regressions.

The lifecycle cases use the real Coordinator, DEVS executor, primitive nodes,
governance pipeline, task queue and SQLite store. Only the model transport and
external research registry are substituted. No paid calls are made.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.coordinator.types import InstanceState, InstanceStatus
from cognitive_core.engine.agentic_devs import InquiryExecutionError
from cognitive_core.engine.governance import get_governance, reset_governance
from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.native_profiles import initial_study


ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "cognitive_core" / "inquiry" / "config" / "coordinator.yaml"
QUESTION = "When is a short conceptual explanation sufficient for a reader?"
CRITERION = "Explain the conditions and assumptions."


def decision(primitive, key, target=None):
    return {
        "action": "invoke", "primitive": primitive, "params_key": key,
        "target": target or {"kind": "study"},
        "reasoning": "Resolve the next material part of the stated question.",
    }


def frame_output(count=0):
    return {
        "confidence": 0.9, "reasoning": "A framing decision requires the reader's intended use.",
        "working_question": QUESTION, "purpose": "Develop a conditional conceptual answer.",
        "scope": ["Ordinary reasoning from explicit premises."], "assumptions": [],
        "subquestions": [{
            "id": "q1", "question": "Which conditions make the explanation useful?",
            "contribution": "Establish the conditions for the answer.",
            "evidence_needs": ["The reader's purpose and explicit reasoning premises."],
            "depends_on": [],
        }],
        "answer_criteria": [CRITERION],
        "clarifications": [{
            "id": f"c{i + 1}", "question": f"What reader requirement matters for choice {i + 1}?",
            "reason": "It changes which conditions the answer should cover.", "blocking": True,
        } for i in range(count)],
    }


def completion_responses(tier="auto"):
    claim = {
        "text": "A concise explanation is useful when its explicit assumptions match the reader's goal.",
        "basis": "inference", "evidence": [],
        "premises": ["Usefulness is assessed against the reader's stated goal."],
        "qualification": "This is a conceptual criterion, not a measured performance claim.",
    }
    return [
        decision("investigate", "investigate_question", {"kind": "question", "question_id": "q1"}),
        {
            "confidence": 0.9, "reasoning": "Relate the stated goal to the explanation's assumptions.",
            "finding": claim["text"], "claims": [claim],
            "question_disposition": "conditional", "material_change": "Made the adequacy condition explicit.",
        },
        decision("deliberate", "synthesize_study"),
        {
            "confidence": 0.9, "reasoning": "Integrate the conceptual finding and its limit.",
            "situation_summary": "The reader needs a conditional conceptual account.",
            "conclusion": claim["text"], "claims": [claim],
            "research_outcome": "conditional", "criteria_coverage": {CRITERION: "Conditional on the stated goal."},
            "limitations": ["No empirical performance comparison was performed."],
        },
        decision("generate", "write_report"),
        {
            "confidence": 0.9, "reasoning": "Render the accepted conditional synthesis.",
            "artifact": "# Answer\n\n" + claim["text"] + "\n\nNo empirical performance comparison was performed.",
            "format": "markdown",
        },
        decision("challenge", "challenge_answer", {"kind": "report"}),
        {
            "confidence": 0.9, "reasoning": "The report preserves its explicit conceptual limitation.",
            "survives": True, "vulnerabilities": [],
            "overall_assessment": "The conditional criterion is supported by its stated premise.",
        },
        decision("govern", "govern_inquiry", {"kind": "report"}),
        {
            "confidence": 0.9, "reasoning": "Apply the configured read-only report policy.",
            "tier_applied": tier, "tier_rationale": "Use the declared review tier.",
            "disposition": "proceed" if tier == "auto" else "suspend",
        },
    ]


class ScriptedModel:
    model_name = "fixture-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, messages, **kwargs):
        text = "\n".join(str(getattr(message, "content", message)) for message in messages)
        self.prompts.append(text)
        if not self.responses:
            raise AssertionError("The native engine made an unexpected extra model request")
        value = self.responses.pop(0)
        return SimpleNamespace(content=json.dumps(value), usage_metadata={}, response_metadata={})


class NativeInquiryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        original_cwd = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, original_cwd)
        self.env = patch.dict(os.environ, {
            "CC_CACHE_ENABLED": "false", "CC_PII_ENABLED": "false",
            "CC_REPLAY_ENABLED": "false", "CC_WEBHOOK_URL": "", "CC_DB_BACKEND": "sqlite",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        reset_governance()
        self.addCleanup(reset_governance)
        self.coordinators = []
        self.addCleanup(self._close_stores)

    def _close_stores(self):
        for coordinator in self.coordinators:
            coordinator.store.close()

    def coordinator(self, *, db="study.db", gate=False):
        coordinator = Coordinator(
            config_path=PACK, db_path=Path(self.tmp.name) / db,
            tool_registry=ToolRegistry(), verbose=False,
            overrides={"overrides": {"reasoning_research": "gate"}} if gate else None,
        )
        self.coordinators.append(coordinator)
        return coordinator

    def run_start(self, coordinator, responses):
        model = ScriptedModel(responses)
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            instance_id = coordinator.start(
                "inquiry", "reasoning_research", {"question": QUESTION, "purpose": "Explain useful conditions."},
            )
        return instance_id, model

    def test_native_clarification_round_trip_across_fresh_coordinator(self):
        governance = get_governance()
        governance.initialize()
        replay = Mock()
        governance._replay_manager = replay
        first = self.coordinator()
        instance_id, start_model = self.run_start(first, [
            decision("frame", "frame_study"), frame_output(2),
        ])
        self.assertEqual(first.get_instance(instance_id).status, InstanceStatus.SUSPENDED)
        self.assertEqual(len(start_model.prompts), 2)
        snapshot = first.checkpoint(instance_id)
        self.assertEqual(snapshot["metadata"]["instance_id"], instance_id)
        self.assertEqual(snapshot["metadata"]["domain"], "reasoning_research")
        self.assertNotIn("_checkpoint_callback", snapshot["metadata"])
        self.assertEqual(len(first.get_work_orders(instance_id)), 2)
        with self.assertRaisesRegex(ValueError, "not approval"):
            first.approve(instance_id, approver="reader")
        with self.assertRaisesRegex(ValueError, "unfulfilled"):
            first.resume(instance_id, {})

        # Recreate both the coordinator and store connection, as the local CLI
        # does on the next invocation. No execution-thread state is retained.
        first.store.close()
        self.coordinators.remove(first)
        resumed = self.coordinator()
        resumed.fulfill_resource(instance_id, "c1", "The reader wants a conceptual overview.")
        self.assertEqual(resumed.get_instance(instance_id).status, InstanceStatus.SUSPENDED)
        self.assertEqual(len(resumed.get_instance(instance_id).pending_work_orders), 1)

        model = ScriptedModel(completion_responses())
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            resumed.fulfill_resource(instance_id, "c2", "The assumptions must be explicit.")
        self.assertEqual(resumed.get_instance(instance_id).status, InstanceStatus.COMPLETED)
        final = resumed.checkpoint(instance_id)
        self.assertEqual(len(final["study"]["frames"]), 1)
        self.assertEqual(final["study"]["clarifications"]["c1"]["answer"], "The reader wants a conceptual overview.")
        self.assertIn("The assumptions must be explicit.", model.prompts[0])
        self.assertTrue(final["study"]["governance"])
        proof = final["metadata"]["inquiry_governance"]
        self.assertEqual(proof["instance_id"], instance_id)
        self.assertEqual(proof["protected_llm_calls"], len(start_model.prompts) + len(model.prompts))
        events = proof["events"]
        self.assertEqual(sum(e["event"] == "circuit_breaker.record" for e in events), len(final["steps"]))
        self.assertTrue(all(
            event["authority"] == "coordinator_store"
            for event in events if event["event"] == "step.checkpoint"
        ))
        self.assertEqual(resumed.get_instance(instance_id).result["study"], final["study"])
        self.assertIsNone(resumed.store.get_suspension(instance_id))
        self.assertEqual(resumed.fulfill_resource(instance_id, "c2", "The assumptions must be explicit."), instance_id)
        with self.assertRaisesRegex(ValueError, "overwritten"):
            resumed.fulfill_resource(instance_id, "c2", "Different answer")
        self.assertTrue(resumed.store.verify_ledger(instance_id)["valid"])
        replay.save_checkpoint.assert_not_called()

    def test_gate_requires_approval_and_binds_to_unchanged_study(self):
        coordinator = self.coordinator(gate=True)
        instance_id, _ = self.run_start(coordinator, [
            decision("frame", "frame_study"), frame_output(), *completion_responses("gate"),
        ])
        self.assertEqual(coordinator.get_instance(instance_id).status, InstanceStatus.SUSPENDED)
        with self.assertRaisesRegex(ValueError, "governance gate"):
            coordinator.resume(instance_id, {})
        original = coordinator.checkpoint(instance_id)
        changed = copy.deepcopy(original)
        changed["study"]["revision"] += 1
        coordinator.store.save_workflow_checkpoint(instance_id, changed)
        with self.assertRaisesRegex(ValueError, "stale"):
            coordinator.approve(instance_id, approver="reader")
        coordinator.store.save_workflow_checkpoint(instance_id, original)
        coordinator.approve(instance_id, approver="reader", notes="Reviewed this exact conditional report.")
        self.assertEqual(coordinator.get_instance(instance_id).status, InstanceStatus.COMPLETED)
        self.assertIsNone(coordinator.store.get_suspension(instance_id))

    def test_unavailable_live_model_fails_without_simulation(self):
        coordinator = self.coordinator()
        with patch("cognitive_core.engine.llm.create_llm", side_effect=RuntimeError("fixture provider unavailable")):
            with self.assertRaises(Exception):
                coordinator.start("inquiry", "reasoning_research", {"question": QUESTION})
        instances = coordinator.store.list_instances()
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].status, InstanceStatus.FAILED)
        state = coordinator.checkpoint(instances[0].instance_id)
        self.assertEqual(state["metadata"]["execution_profile"], "inquiry.v1")
        self.assertEqual(state["steps"], [])
        self.assertGreaterEqual(state["study"]["usage"]["model_calls"], 0)

    def test_failed_run_recovers_accepted_work_and_usage_in_fresh_coordinator(self):
        first = self.coordinator()
        with self.assertRaises(Exception):
            self.run_start(first, [decision("frame", "frame_study"), frame_output()])
        instance = first.store.list_instances()[0]
        self.assertEqual(instance.status, InstanceStatus.FAILED)
        saved = first.checkpoint(instance.instance_id)
        self.assertEqual(len(saved["study"]["frames"]), 1)
        self.assertEqual(len(saved["steps"]), 1)
        used = saved["study"]["usage"]["model_calls"]
        self.assertGreater(used, 2)
        first.store.close()
        self.coordinators.remove(first)
        reset_governance()

        recovered = self.coordinator()
        model = ScriptedModel(completion_responses())
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            recovered.recover_inquiry(instance.instance_id)
        final = recovered.checkpoint(instance.instance_id)
        self.assertEqual(recovered.get_instance(instance.instance_id).status, InstanceStatus.COMPLETED)
        self.assertEqual(final["study"]["frames"], saved["study"]["frames"])
        self.assertEqual(final["steps"][0], saved["steps"][0])
        self.assertGreater(final["study"]["usage"]["model_calls"], used)
        self.assertNotIn("inquiry_termination", final["metadata"])
        self.assertIsNone(recovered.store.get_suspension(instance.instance_id))
        self.assertTrue(recovered.store.verify_ledger(instance.instance_id)["valid"])

    def test_checkpoint_and_ledger_rollback_together(self):
        coordinator = self.coordinator()
        instance = InstanceState.create("inquiry", "reasoning_research")
        instance.status = InstanceStatus.RUNNING
        coordinator.store.save_instance(instance)
        initial = {
            "metadata": {"execution_profile": "inquiry.v1"},
            "study": initial_study(QUESTION), "steps": [], "current_step": "",
        }
        coordinator._persist_inquiry_checkpoint(instance, initial, "fixture_initial")
        changed = copy.deepcopy(initial)
        changed["study"]["revision"] += 1
        changed["current_step"] = "accepted_candidate"
        with patch.object(coordinator.store, "log_action", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "audit"):
                coordinator._persist_inquiry_checkpoint(instance, changed, "step_completed")
        self.assertEqual(coordinator.checkpoint(instance.instance_id), initial)
        self.assertEqual(coordinator.get_instance(instance.instance_id).current_step, "")
        self.assertEqual(len(coordinator.get_ledger(instance.instance_id)), 1)

    def test_post_commit_hook_failure_retains_accepted_step_and_original_error(self):
        coordinator = self.coordinator()
        governance = get_governance()
        governance.initialize()
        reason = "fixture post-commit governance failure"
        with patch.object(governance, "record_step_result", side_effect=RuntimeError(reason)):
            with self.assertRaisesRegex(InquiryExecutionError, reason) as raised:
                self.run_start(coordinator, [decision("frame", "frame_study"), frame_output()])

        instance = coordinator.store.list_instances()[0]
        self.assertEqual(instance.status, InstanceStatus.FAILED)
        saved = coordinator.checkpoint(instance.instance_id)
        self.assertEqual(len(saved["steps"]), 1)
        self.assertEqual(len(saved["study"]["frames"]), 1)
        self.assertEqual(saved["study"]["usage"]["model_calls"], 2)
        self.assertEqual(saved["study"], raised.exception.final_state["study"])
        self.assertEqual(saved["steps"], raised.exception.final_state["steps"])
        self.assertEqual(saved["metadata"]["inquiry_termination"], {"status": "failed", "reason": reason})
        self.assertNotIn("_inquiry_committed_acceptance", saved["metadata"])

        ledger = coordinator.get_ledger(instance.instance_id)
        accepted = [entry for entry in ledger if entry["action_type"] == "step_completed"]
        failed = [entry for entry in ledger if entry["action_type"] == "inquiry_failed"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["details"]["reason"], reason)
        self.assertEqual(accepted[0]["details"]["study_revision"], saved["study"]["revision"])
        self.assertEqual(failed[0]["details"]["study_revision"], saved["study"]["revision"])
        self.assertTrue(coordinator.store.verify_ledger(instance.instance_id)["valid"])

    def test_native_snapshot_preserves_large_source_and_report(self):
        coordinator = self.coordinator()
        text = "A decisive source passage. " * 400
        state = {
            "metadata": {"execution_profile": "inquiry.v1", "_checkpoint_callback": lambda *_: None},
            "study": initial_study(QUESTION),
            "steps": [
                {"step_name": "read", "primitive": "retrieve", "output": {"data": {"source": text}}},
                {"step_name": "report", "primitive": "generate", "output": {"artifact": text}},
            ],
        }
        snapshot = coordinator._compact_state_for_suspension(state)
        self.assertEqual(snapshot["steps"][0]["output"]["data"]["source"], text)
        self.assertEqual(snapshot["steps"][1]["output"]["artifact"], text)
        self.assertNotIn("_checkpoint_callback", snapshot["metadata"])
        self.assertEqual(coordinator._extract_result_summary(state)["steps"], snapshot["steps"])

    def test_checkpoint_rejects_non_json_state(self):
        coordinator = self.coordinator()
        with self.assertRaises(TypeError):
            coordinator.store.save_workflow_checkpoint("fixture", {"callback": lambda: None})
        self.assertIsNone(coordinator.store.get_workflow_checkpoint("fixture"))


if __name__ == "__main__":
    unittest.main()
