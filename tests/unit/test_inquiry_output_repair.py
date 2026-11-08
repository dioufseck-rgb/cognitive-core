"""Budgeted correction of malformed model output without weakening evidence rules."""

from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.demo import (
    DEMO_QUESTION, DEMO_TEXT, DEMO_URL, ScriptedReasoner, demo_script,
)
from cognitive_core.inquiry.models import InquiryLimits
from cognitive_core.inquiry.reasoning import CoreReasoner
from cognitive_core.inquiry.runner import InquiryRunner
from cognitive_core.inquiry.store import InquiryStore


RAW_SENTINEL = "malformed-output-private-value"
UNKNOWN_FIELD = "malformed-output-private-field"


def invalid_read_plan():
    plan = deepcopy(demo_script()[2][1])
    plan["calls"][0]["arguments"][UNKNOWN_FIELD] = RAW_SENTINEL
    return plan


class FixtureGovernance:
    """Return synthetic provider output through CoreReasoner's real parser."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def protected_llm_call(self, **kwargs):
        operation = re.search(r"OPERATION: (\w+)", kwargs["prompt"]).group(1)
        self.calls.append({"operation": operation, **deepcopy(kwargs)})
        if not self.script:
            raise AssertionError("The fixture output sequence was exhausted")
        expected, output = self.script.pop(0)
        if operation != expected:
            raise AssertionError(f"Expected {expected}, got {operation}")
        if isinstance(output, BaseException):
            raise output
        return SimpleNamespace(raw_response=output if isinstance(output, str) else json.dumps(output))


class InquiryOutputRepairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = InquiryStore(Path(self.directory.name) / "study")
        self.searches = []
        self.reads = []
        self.tools = ToolRegistry()

        def search(args):
            self.searches.append(deepcopy(args))
            return {"results": [{
                "url": DEMO_URL, "title": "Synthetic comparison", "snippet": "A candidate, not read evidence.",
            }]}

        def read(args):
            self.reads.append(deepcopy(args))
            return {
                "url": args["url"], "title": "Synthetic comparison", "text": DEMO_TEXT,
                "content_type": "text/plain", "truncated": False,
            }

        self.tools.register("web_search", search)
        self.tools.register("read_source", read)

    def runner(self, script):
        governance = FixtureGovernance(script)
        reasoner = CoreReasoner(llm=object(), governance=governance)
        return InquiryRunner(reasoner, self.tools, self.store), governance

    def assert_no_raw_output_leak(self, state, *governances):
        stored = self.store.state_path.read_text(encoding="utf-8")
        report = self.store.report_path.read_text(encoding="utf-8")
        for sentinel in (RAW_SENTINEL, UNKNOWN_FIELD):
            self.assertNotIn(sentinel, state.model_dump_json())
            self.assertNotIn(sentinel, stored)
            self.assertNotIn(sentinel, report)
            for governance in governances:
                for call in governance.calls:
                    self.assertNotIn(sentinel, call["prompt"])
                    self.assertNotIn(sentinel, json.dumps(call["case_input"]))

    def assert_repair_feedback_present(self, repair, prompt):
        # The correction may serialize issue records rather than repeat the
        # human-facing summary. Assert the actionable content is preserved.
        self.assertTrue(repair.issues)
        for issue in repair.issues:
            for field in ("path", "code", "message"):
                self.assertIn(issue[field], prompt)

    def test_malformed_read_plan_is_corrected_before_any_tool_execution(self):
        script = demo_script()
        script.insert(2, ("retrieve", invalid_read_plan()))
        runner, governance = self.runner(script)

        state = runner.start(DEMO_QUESTION)

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 17)  # The original demo takes 16 steps.
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assertEqual(self.reads[0]["url"], DEMO_URL)
        self.assertEqual(sum(e.kind == "output_rejected" for e in state.events), 1)
        self.assertEqual(sum(e.kind == "schema_repair" for e in state.events), 1)
        self.assertEqual(sum(e.kind == "retrieve" for e in state.events), 3)
        self.assertIsNone(state.output_repair)
        self.assertFalse(governance.script)
        self.assertIn("Rejected model outputs: 1; correction attempts started: 1.",
                      self.store.report_path.read_text(encoding="utf-8"))
        self.assert_no_raw_output_leak(state, governance)

    def test_five_call_plan_is_rejected_in_full_then_replanned(self):
        script = demo_script()
        overfull = deepcopy(script[2][1])
        overfull["calls"] *= 5
        script.insert(2, ("retrieve", overfull))
        runner, governance = self.runner(script)

        state = runner.start(DEMO_QUESTION)

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 17)
        self.assertEqual(len(self.reads), 1)
        rejected = next(e for e in state.events if e.kind == "output_rejected")
        self.assertIn("calls", json.dumps(rejected.details))
        self.assertTrue(any(issue["code"] == "too_long" for issue in rejected.details["issues"]))
        self.assertEqual(len(governance.calls), len(demo_script()) + 1)

    def test_malformed_json_can_be_regenerated_without_replaying_raw_text(self):
        script = demo_script()
        script.insert(2, ("retrieve", "{ invalid JSON " + RAW_SENTINEL))
        runner, governance = self.runner(script)

        state = runner.start(DEMO_QUESTION)

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 17)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assert_no_raw_output_leak(state, governance)

    def test_repeated_invalid_output_stops_after_two_corrections(self):
        invalid = [("retrieve", invalid_read_plan()) for _ in range(4)]
        runner, governance = self.runner(demo_script()[:2] + invalid)

        state = runner.start(DEMO_QUESTION, limits=InquiryLimits(max_steps=40))

        self.assertEqual(state.status, "failed")
        self.assertIn("calls", state.error)
        self.assertEqual(state.usage.steps, 6)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))
        self.assertEqual(len(governance.calls), 5)  # Frame, search plan, three rejected read plans.
        self.assertEqual(len(governance.script), 1)
        self.assertEqual(sum(e.kind == "output_rejected" for e in state.events), 3)
        self.assertEqual(sum(e.kind == "schema_repair" for e in state.events), 2)
        self.assertEqual(state.output_repair.failures, 3)
        self.assertIsNone(state.in_flight)
        self.assertEqual(state.questions["q1"].phase, "plan")
        self.assertFalse(state.questions["q1"].pending_calls)
        self.assertNotEqual(governance.calls[-1]["prompt"], governance.calls[-2]["prompt"])
        self.assertEqual(
            [call["case_input"]["remaining_budget"]["steps"] for call in governance.calls[-3:]],
            [36, 35, 34],
        )
        self.assert_no_raw_output_leak(state, governance)

    def test_automatic_correction_can_be_disabled(self):
        runner, governance = self.runner(demo_script()[:2] + [("retrieve", invalid_read_plan())])

        state = runner.start(DEMO_QUESTION, limits=InquiryLimits(max_output_repairs=0))

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.usage.steps, 4)
        self.assertEqual(len(governance.calls), 3)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))
        self.assertFalse(any(e.kind == "schema_repair" for e in state.events))

    def test_budget_pause_preserves_feedback_and_resumes_without_repeating_search(self):
        first, before = self.runner(demo_script()[:2] + [("retrieve", invalid_read_plan())])

        paused = first.start(DEMO_QUESTION, limits=InquiryLimits(max_steps=4))

        self.assertEqual(paused.status, "budget_exhausted", paused.error)
        self.assertEqual(paused.usage.steps, 4)
        self.assertIsNone(paused.in_flight)
        self.assertEqual(paused.output_repair.failures, 1)
        self.assertEqual(paused.output_repair.primitive, "retrieve")
        self.assertEqual(paused.output_repair.question_id, "q1")
        self.assertEqual(paused.sources["S1"].status, "discovered")
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))
        self.assertFalse(any(e.kind == "schema_repair" for e in paused.events))
        self.assertIn("Pending output correction:", self.store.report_path.read_text(encoding="utf-8"))

        second, after = self.runner(demo_script()[2:])
        state = second.resume(additional_steps=20)

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 17)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assert_repair_feedback_present(paused.output_repair, after.calls[0]["prompt"])
        self.assertEqual(len(state.frames), 1)
        self.assertIsNone(state.output_repair)
        self.assert_no_raw_output_leak(state, before, after)

    def test_budget_resume_does_not_reset_the_remaining_correction_allowance(self):
        first, _ = self.runner(demo_script()[:2] + [("retrieve", invalid_read_plan())])
        paused = first.start(DEMO_QUESTION, limits=InquiryLimits(max_steps=4))
        self.assertEqual(paused.output_repair.failures, 1)
        second, governance = self.runner([("retrieve", invalid_read_plan()) for _ in range(3)])

        state = second.resume(additional_steps=10)

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.usage.steps, 6)
        self.assertEqual(len(governance.calls), 2)
        self.assertEqual(len(governance.script), 1)
        self.assertEqual(state.output_repair.failures, 3)
        self.assertEqual(sum(e.kind == "schema_repair" for e in state.events), 2)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))

    def test_exhausted_repair_checkpoint_does_not_gain_an_extra_attempt(self):
        first, _ = self.runner(demo_script()[:2] + [("retrieve", invalid_read_plan()) for _ in range(3)])
        checkpoint = first.start(DEMO_QUESTION)
        self.assertEqual(checkpoint.output_repair.failures, 3)
        # Model an interruption after the last rejection was saved, before the
        # bounded failure reached _run's normal failed-status checkpoint.
        checkpoint.status = "ready"
        checkpoint.error = None
        checkpoint.events = [event for event in checkpoint.events if event.kind != "failed"]
        self.store.save(checkpoint)
        second, governance = self.runner(demo_script()[2:])

        state = second.resume()

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.usage.steps, checkpoint.usage.steps)
        self.assertEqual(state.output_repair.failures, 3)
        self.assertFalse(governance.calls)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))
        self.assertEqual(sum(e.kind == "output_rejected" for e in state.events), 3)
        self.assertFalse(any(e.kind == "repair_restarted" for e in state.events))

    def test_keyboard_interrupt_during_correction_preserves_feedback(self):
        first, before = self.runner(demo_script()[:2] + [
            ("retrieve", invalid_read_plan()), ("retrieve", KeyboardInterrupt()),
        ])
        with self.assertRaises(KeyboardInterrupt):
            first.start(DEMO_QUESTION)
        paused = self.store.load()
        self.assertEqual(paused.status, "ready")
        self.assertEqual(paused.usage.steps, 5)
        self.assertEqual(paused.output_repair.failures, 1)

        second, after = self.runner(demo_script()[2:])
        state = second.resume()

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 18)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assert_repair_feedback_present(paused.output_repair, after.calls[0]["prompt"])
        self.assertIsNone(state.output_repair)
        self.assert_no_raw_output_leak(state, before, after)

    def test_explicit_resume_after_failure_starts_a_new_bounded_batch(self):
        first, before = self.runner(demo_script()[:2] + [("retrieve", invalid_read_plan()) for _ in range(3)])
        failed = first.start(DEMO_QUESTION)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.output_repair.failures, 3)

        second, after = self.runner(demo_script()[2:])
        state = second.resume()

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 19)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assertTrue(any(e.kind == "repair_restarted" for e in state.events))
        self.assert_repair_feedback_present(failed.output_repair, after.calls[0]["prompt"])
        self.assertIsNone(state.output_repair)
        self.assert_no_raw_output_leak(state, before, after)

    def test_provider_failure_is_not_treated_as_a_formatting_error(self):
        runner, governance = self.runner(demo_script()[:2] + [
            ("retrieve", RuntimeError(RAW_SENTINEL)),
            ("retrieve", deepcopy(demo_script()[2][1])),
        ])

        state = runner.start(DEMO_QUESTION)

        self.assertEqual(state.status, "failed")
        self.assertIn("protected inquiry model call failed", state.error)
        self.assertEqual(state.usage.steps, 4)
        self.assertEqual(len(governance.calls), 3)
        self.assertEqual(len(governance.script), 1)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))
        self.assertFalse(any(e.kind in {"schema_repair", "output_rejected"} for e in state.events))
        self.assertIsNone(state.output_repair)
        self.assert_no_raw_output_leak(state, governance)

    def test_correct_shape_does_not_allow_fabricated_evidence_quotes(self):
        script = demo_script()
        script[4][1]["claims"][0]["evidence"][0]["quote"] = "This quotation is absent from the read source."
        runner, governance = self.runner(script)

        state = runner.start(DEMO_QUESTION)

        self.assertEqual(state.status, "failed")
        self.assertIn("does not occur", state.error)
        self.assertEqual(len(governance.calls), 5)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assertFalse(state.questions["q1"].findings)
        self.assertFalse(any(e.kind == "schema_repair" for e in state.events))

    def test_correct_shape_does_not_allow_an_invented_source_url(self):
        script = demo_script()
        script[2][1]["calls"][0]["arguments"]["url"] = "https://example.org/not-discovered"
        runner, governance = self.runner(script)

        state = runner.start(DEMO_QUESTION)

        self.assertEqual(state.status, "failed")
        self.assertIn("must come from search results", state.error)
        self.assertEqual(len(governance.calls), 3)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 0))
        self.assertFalse(any(e.kind == "schema_repair" for e in state.events))

    def test_adapter_validation_error_uses_the_same_bounded_correction_path(self):
        script = demo_script()
        script.insert(2, ("retrieve", invalid_read_plan()))
        model = ScriptedReasoner(script)

        state = InquiryRunner(model, self.tools, self.store).start(DEMO_QUESTION)

        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 17)
        self.assertEqual((len(self.searches), len(self.reads)), (1, 1))
        self.assertEqual(sum(e.kind == "schema_repair" for e in state.events), 1)
        self.assert_no_raw_output_leak(state)
        for _, _, instructions in model.contexts:
            self.assertNotIn(UNKNOWN_FIELD, instructions)
            self.assertNotIn(RAW_SENTINEL, instructions)


if __name__ == "__main__":
    unittest.main()
