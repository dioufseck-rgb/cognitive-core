"""Native inquiry orchestration, protected dispatch, and DEVS recovery tests.

Model and network boundaries use deterministic fixtures. The execution tests
retain the real DEVS model, registered nodes, reducers, and protected-call path.
They establish runtime behavior, not live-model research quality.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import re
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cognitive_core.engine.agentic_devs import (
    AgenticWorkflowExecutor, InquiryExecutionError, OrchestratorStep,
    _AgenticWorkflowStep,
)
from cognitive_core.engine.devs_executor import StepInterrupt, StepState
from cognitive_core.engine.governance import GovernancePipeline
from cognitive_core.engine.retry import RetryPolicy
from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.demo import DEMO_QUESTION, demo_challenge, demo_frame
from cognitive_core.inquiry.native_profiles import apply_output, initial_study


PROFILE_NAMES = {
    "frame": "frame", "retrieve": "retrieve", "investigate": "finding",
    "verify": "support_review", "reflect": "reflection", "deliberate": "synthesis",
    "generate": "report", "challenge": "challenge", "govern": "govern",
}


def config():
    return {
        "name": "native_inquiry_test", "mode": "agentic", "execution_profile": "inquiry.v1",
        "goal": "Answer the supplied question without changing its application.",
        "available_primitives": list(PROFILE_NAMES),
        "constraints": {
            "max_steps": 40, "max_repeat_per_target": 3,
            "must_include": ["frame", "deliberate", "generate", "challenge", "govern"],
            "must_end_with": "govern", "challenge_must_pass": True,
            "budgets": {"model_calls": 100, "searches": 6, "source_reads": 8,
                        "output_repairs_per_operation": 2},
        },
        "orchestrator": {"strategy": "Address the most consequential unresolved issue."},
        "primitive_configs": {
            primitive: {"primitive": primitive, "output_profile": f"inquiry.{name}.v1",
                        "params": {"additional_instructions": "Keep the conclusion conditional."}}
            for primitive, name in PROFILE_NAMES.items()
        },
    }


def state(*, framed=False):
    value = {
        "input": {"question": DEMO_QUESTION, "purpose": "Choose a conceptual architecture."},
        "steps": [], "current_step": "", "loop_counts": {}, "routing_log": [],
        "metadata": {"execution_profile": "inquiry.v1", "instance_id": "native-test",
                     "correlation_id": "native-correlation", "domain": "research"},
        "study": initial_study(DEMO_QUESTION, "Choose a conceptual architecture."),
    }
    if framed:
        value["study"] = apply_output(value, "frame", demo_frame(), {"kind": "study"}, "initial-frame")
    return value


def decision(primitive, kind="study", question_id=None, **overrides):
    target = {"kind": kind}
    if question_id:
        target["question_id"] = question_id
    return {"action": "invoke", "params_key": primitive, "primitive": primitive,
            "target": target, "reasoning": "Resolve the next material issue.", **overrides}


def base(**values):
    return {"confidence": 0.65, "reasoning": "Synthetic fixture rationale.", **values}


def claim():
    return {
        "text": "A fully specified rule can be implemented as a fixed operation.",
        "basis": "inference", "premises": ["The rule and its inputs are fully specified."],
        "qualification": "This conceptual conclusion does not establish measured performance.",
    }


def full_script():
    operations = [
        (decision("frame"), demo_frame()),
        (decision("investigate", "question", "q1"), base(
            finding="Use a fixed operation when the rule is fully specified.",
            claims=[claim()], question_disposition="conditional",
            material_change="The applicable condition is explicit.",
        )),
        (decision("challenge", "question", "q1"), demo_challenge()),
        (decision("deliberate"), base(
            situation_summary="The answer is conceptual and conditional.",
            conclusion="Use a fixed operation for a fully specified rule.", claims=[claim()],
            research_outcome="conditional", limitations=["No performance measurement was made."],
        )),
        (decision("generate", "report"), base(
            artifact="Use a fixed operation for a fully specified rule. No performance measurement was made.",
            format="markdown",
        )),
        (decision("challenge", "report"), demo_challenge()),
        (decision("govern", "report"), base(
            tier_applied="auto", tier_rationale="Read-only, explicitly conditional fixture result.",
            disposition="proceed",
        )),
    ]
    script = []
    for selected, response in operations:
        script.extend([("orchestrate", selected), (selected["primitive"], response)])
    return script


class BoundaryModel:
    model_name = "fixture-model"

    def __init__(self, script):
        self.script = copy.deepcopy(script)
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        operation = re.search(r"OPERATION: (\w+)", prompt).group(1)
        if not self.script:
            raise AssertionError(f"Unexpected extra model operation: {operation}")
        expected, response = self.script.pop(0)
        if operation != expected:
            raise AssertionError(f"Expected {expected}, received {operation}")
        if isinstance(response, Exception):
            raise response
        content = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(content=content, response_metadata={})


class NativeInquiryEngineTests(unittest.TestCase):
    @contextmanager
    def boundary(self, script):
        model = BoundaryModel(script)
        governance = GovernancePipeline()
        governance._initialized = True
        policy = RetryPolicy(max_attempts=2, backoff_base=0, backoff_max=0, jitter=0)
        try:
            with patch("cognitive_core.engine.llm.create_llm", return_value=model), \
                 patch("cognitive_core.engine.llm._load_config", return_value={}), \
                 patch("cognitive_core.engine.governance.get_governance", return_value=governance), \
                 patch("cognitive_core.engine.retry.get_retry_policy", return_value=policy):
                yield model, governance
        finally:
            governance.close()

    def orchestrator(self, current=None, configuration=None):
        return OrchestratorStep(configuration or config(), current or state(), "fixture-model", 0)

    def test_real_nodes_complete_without_branch_challenge_forcing_govern(self):
        checkpoints, completed = [], []

        def checkpoint(current, event, details):
            json.dumps(current)  # Runtime callback objects must never enter a snapshot.
            checkpoints.append((event, copy.deepcopy(current)))

        def accepted(name, output, current):
            self.assertEqual(current["steps"][-1]["step_name"], name)
            completed.append(output["primitive"])

        with self.boundary(full_script()) as (model, gov):
            result = AgenticWorkflowExecutor(config(), ToolRegistry(), model="fixture-model").execute(
                state()["input"], initial_state=state(), step_callback=accepted,
                checkpoint_callback=checkpoint,
            )
        self.assertTrue(result.completed)
        self.assertFalse(model.script)
        self.assertEqual(completed, ["frame", "investigate", "challenge", "deliberate", "generate", "challenge", "govern"])
        self.assertEqual(result.final_state["study"]["usage"]["model_calls"], 14)
        self.assertEqual(result.final_state["study"]["usage"]["steps"], 7)
        self.assertEqual(result.final_state["study"]["governance"]["tier_applied"], "auto")
        self.assertEqual(len([entry for entry in gov._proof_ledger if entry["event"] == "llm.call"]), 14)
        self.assertEqual(len([entry for entry in result.final_state["metadata"]["inquiry_model_calls"]
                              if entry["profile"] == "inquiry.orchestrator.v1"]), 7)
        self.assertTrue(any(event == "invocation_dispatched" for event, _ in checkpoints))
        json.dumps(result.final_state)

    def test_fresh_executor_resumes_without_repeating_accepted_frame(self):
        script = full_script()
        with self.boundary(script[:2]):
            paused = AgenticWorkflowExecutor(config(), ToolRegistry(), model="fixture-model").execute(
                state()["input"], initial_state=state(),
                step_callback=lambda name, output, current: StepInterrupt("test pause", name),
            )
        self.assertFalse(paused.completed)
        self.assertEqual(len(paused.final_state["study"]["frames"]), 1)
        with self.boundary(script[2:]) as (model, _):
            resumed = AgenticWorkflowExecutor(config(), ToolRegistry(), model="fixture-model").resume(paused.final_state)
        self.assertTrue(resumed.completed)
        self.assertFalse(model.script)
        self.assertEqual(len(resumed.final_state["study"]["frames"]), 1)
        self.assertEqual(resumed.final_state["study"]["usage"]["model_calls"], 14)
        self.assertEqual(resumed.final_state["metadata"]["instance_id"], "native-test")

    def test_invalid_decision_is_repaired_through_protected_call_and_counted(self):
        current = state()
        with self.boundary([("orchestrate", {"action": "end"}), ("orchestrate", decision("frame"))]) as (model, gov):
            selected = self.orchestrator(current)._make_decision()
        self.assertEqual(selected["params_key"], "frame")
        self.assertEqual(current["study"]["usage"]["model_calls"], 2)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("previous decision was rejected", model.prompts[1])
        self.assertEqual(len([e for e in gov._proof_ledger if e["event"] == "llm.call"]), 2)

    def test_exhausted_invalid_orchestration_fails_with_no_accepted_step(self):
        snapshots = []
        with self.boundary([("orchestrate", "not json")] * 3):
            with self.assertRaises(InquiryExecutionError):
                AgenticWorkflowExecutor(config(), ToolRegistry(), model="fixture-model").execute(
                    state()["input"], initial_state=state(),
                    checkpoint_callback=lambda current, event, details: snapshots.append(current),
                )
        last = snapshots[-1]
        self.assertFalse(last["steps"])
        self.assertEqual(last["study"]["usage"]["model_calls"], 3)
        self.assertEqual(last["metadata"]["inquiry_termination"]["status"], "failed")

    def test_provider_retry_is_counted_in_orchestration_budget(self):
        current = state()
        with self.boundary([("orchestrate", ConnectionError("temporary connection failure")),
                            ("orchestrate", decision("frame"))]) as (model, _):
            result = self.orchestrator(current)._make_decision()
        self.assertEqual(result["primitive"], "frame")
        self.assertEqual(current["study"]["usage"]["model_calls"], 2)
        self.assertFalse(model.script)

    def test_mismatched_configured_profile_is_rejected_before_dispatch(self):
        wrong = config()
        wrong["primitive_configs"]["frame"]["output_profile"] = "inquiry.finding.v1"
        with self.assertRaisesRegex(InquiryExecutionError, "mismatched"):
            self.orchestrator(configuration=wrong)

    def test_target_repeat_limit_uses_stable_profile_and_question(self):
        current = state(framed=True)
        current["steps"] = [
            {"primitive": "investigate", "params_key": "investigate",
             "step_name": f"invented_label_{index}",
             "target": {"kind": "question", "question_id": "q1", "expected_study_revision": index}}
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "repeat limit"):
            self.orchestrator(current)._validate_inquiry_decision(decision("investigate", "question", "q1"))

    def test_govern_rejects_branch_only_review(self):
        current = state(framed=True)
        current["study"] = apply_output(current, "challenge", demo_challenge(),
                                         {"kind": "question", "question_id": "q1"}, "branch-review")
        with self.assertRaises(ValueError):
            self.orchestrator(current)._validate_inquiry_decision(decision("govern", "report"))

    def test_checkpoint_callback_failure_does_not_accept_candidate_frame(self):
        snapshots = []

        def reject_acceptance(*_):
            raise RuntimeError("atomic acceptance failed")

        with self.boundary(full_script()[:2]):
            with self.assertRaisesRegex(InquiryExecutionError, "atomic acceptance failed"):
                AgenticWorkflowExecutor(config(), ToolRegistry(), model="fixture-model").execute(
                    state()["input"], initial_state=state(), step_callback=reject_acceptance,
                    checkpoint_callback=lambda current, event, details: snapshots.append(current),
                )
        self.assertFalse(snapshots[-1]["steps"])
        self.assertFalse(snapshots[-1]["study"]["frames"])
        self.assertEqual(snapshots[-1]["study"]["usage"]["model_calls"], 2)

    def test_budget_exhaustion_does_not_report_completed_or_call_model(self):
        current = state(framed=True)
        current["study"]["usage"]["model_calls"] = current["study"]["limits"]["max_model_calls"]
        with self.boundary([]) as (model, _):
            with self.assertRaisesRegex(InquiryExecutionError, "budget exhausted"):
                AgenticWorkflowExecutor(config(), ToolRegistry(), model="fixture-model").resume(current)
        self.assertFalse(model.prompts)

    def test_runtime_profile_and_target_metadata_reach_native_dispatch(self):
        current = state(framed=True)
        cfg = config()
        cfg["primitive_configs"]["investigate"]["params"]["scope"] = "A configured scope"
        step = self.orchestrator(current, cfg)._build_step_cfg(decision("investigate", "question", "q1"))
        self.assertEqual(step["output_profile"], "inquiry.finding.v1")
        self.assertEqual(step["params"]["scope"], "A configured scope")
        self.assertEqual(step["target"]["question_id"], "q1")
        self.assertEqual(step["target"]["expected_study_revision"], current["study"]["revision"])
        self.assertEqual(step["params"]["_inquiry_invocation_id"], step["invocation_id"])
        self.assertNotIn("DYNAMIC SPEC FROM REFLECT", json.dumps(step))

    def test_yaml_variant_changes_selected_native_profile_and_applied_frame(self):
        import yaml
        from cognitive_core.engine.composer import load_three_layer

        class ConfigModel:
            model_name = "fixture-model"

            def invoke(self, messages):
                prompt = messages[0].content
                operation = re.search(r"OPERATION: (\w+)", prompt).group(1)
                if operation == "orchestrate":
                    key = re.search(r"CHOOSE_CONFIG: (frame_[ab])", prompt).group(1)
                    response = decision("frame", params_key=key)
                else:
                    variant = re.search(r"FRAME_MODE: ([AB])", prompt).group(1)
                    response = demo_frame()
                    response["scope"] = [f"Configured study scope {variant}"]
                return SimpleNamespace(content=json.dumps(response), response_metadata={})

        workflow = config()
        workflow["orchestrator"]["strategy"] = "${domain.strategy}"
        workflow["primitive_configs"].pop("frame")
        for suffix in ("a", "b"):
            workflow["primitive_configs"][f"frame_{suffix}"] = {
                "primitive": "frame", "output_profile": "inquiry.frame.v1",
                "params": {"additional_instructions": f"FRAME_MODE: {suffix.upper()}"},
            }
        results = []
        with tempfile.TemporaryDirectory() as directory:
            workflow_path = Path(directory) / "workflow.yaml"
            workflow_path.write_text(yaml.safe_dump(workflow), encoding="utf-8")
            for suffix in ("a", "b"):
                domain_path = Path(directory) / f"domain_{suffix}.yaml"
                domain_path.write_text(yaml.safe_dump({
                    "domain_name": "research", "strategy": f"CHOOSE_CONFIG: frame_{suffix}",
                }), encoding="utf-8")
                merged, _ = load_three_layer(workflow_path, domain_path)
                with self.boundary([]), patch("cognitive_core.engine.llm.create_llm", return_value=ConfigModel()):
                    results.append(AgenticWorkflowExecutor(merged, ToolRegistry(), model="fixture-model").execute(
                        state()["input"], initial_state=state(),
                        step_callback=lambda name, output, current: StepInterrupt("inspect frame", name),
                    ))
        self.assertEqual([r.final_state["steps"][0]["params_key"] for r in results], ["frame_a", "frame_b"])
        self.assertEqual([r.final_state["study"]["frames"][0]["scope"] for r in results],
                         [["Configured study scope A"], ["Configured study scope B"]])

    def test_late_timed_out_orchestration_is_not_recorded_as_a_new_decision(self):
        current, selected = state(), config()
        selected["orchestrator"]["sla_seconds"] = 0.02
        orchestrator = self.orchestrator(current, selected)
        injected, timed_out, release, returned = [], threading.Event(), threading.Event(), threading.Event()

        def inject(**kwargs):
            injected.append(kwargs["value"])
            timed_out.set()

        def delayed():
            release.wait(timeout=2)
            returned.set()
            return decision("frame")

        orchestrator._simulator = SimpleNamespace(inject=inject)
        with patch.object(orchestrator, "_make_decision", side_effect=delayed):
            orchestrator._decide_async()
            self.assertTrue(timed_out.wait(timeout=1))
            release.set()
            self.assertTrue(returned.wait(timeout=1))
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["_decision"]["action"], "error")
        self.assertEqual(len(current["routing_log"]), 1)
        self.assertIn("timed out", orchestrator._execution_error)

    def test_late_timed_out_native_node_cannot_mutate_the_study(self):
        current = state()
        component = _AgenticWorkflowStep("frame", current, ToolRegistry(), None, "fixture-model", 0)
        invocation_id = "slow-frame"
        component._inquiry_active_id = invocation_id
        component._inquiry_expected_revision = current["study"]["revision"]
        injected, signaled, release, finished = [], threading.Event(), threading.Event(), threading.Event()

        def inject(**kwargs):
            injected.append(kwargs["value"])
            signaled.set()

        def slow_node(step, execution_state=None):
            release.wait(timeout=2)
            execution_state["study"]["original_question"] = "late corruption"
            finished.set()
            return {"finding": "late output"}, None

        component._simulator = SimpleNamespace(inject=inject)
        step = StepState("running", "slow", "frame", {
            "_inquiry_invocation_id": invocation_id, "sla_seconds": 0.02,
        }, "fixture-model", 0)
        with patch.object(component, "_execute_step", side_effect=slow_node):
            component._run_step_async(step, time.time())
            self.assertTrue(signaled.wait(timeout=1))
            release.set()
            self.assertTrue(finished.wait(timeout=1))
        self.assertEqual(current["study"]["original_question"], DEMO_QUESTION)
        self.assertEqual(len(injected), 1)
        self.assertTrue(injected[0]["output"]["_inquiry_failure"])


if __name__ == "__main__":
    unittest.main()
