"""Behavioral tests of complete inquiries, recovery, and evidence boundaries."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.demo import (
    DEMO_QUESTION, DEMO_TEXT, DEMO_URL, ScriptedReasoner, demo_challenge,
    demo_finding, demo_frame, demo_reflection, demo_script, demo_synthesis, demo_tools,
)
from cognitive_core.inquiry.models import InquiryLimits
from cognitive_core.inquiry.runner import InquiryRunner
from cognitive_core.inquiry.store import InquiryStore


def base(**kwargs):
    return {"confidence": 0.6, "reasoning": "Test rationale.", **kwargs}


def no_search():
    return ("retrieve", {"calls": [], "rationale": "Use supplied reasoning context."})


def conceptual_finding():
    return base(finding="The conclusion depends on the stated design objective.",
                claims=[{"text": "The design objective determines the tradeoff.", "basis": "inference"}])


def conceptual_synthesis():
    return base(situation_summary="The branches are coherent.", conclusion="Choose according to the design objective.",
                claims=[{"text": "The choice follows from the stated objective.", "basis": "inference"}])


def ending(synthesis=None, answer="The recommendation is conditional on the stated objective."):
    return [("deliberate", synthesis or conceptual_synthesis()),
            ("challenge", demo_challenge()), ("reflect", demo_reflection()),
            ("generate", base(artifact=answer, format="markdown"))]


class InquiryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = InquiryStore(self.tmp.name)

    def runner(self, script, tools=None):
        reasoner = ScriptedReasoner(script)
        return InquiryRunner(reasoner, tools or demo_tools(), self.store), reasoner

    def test_full_loop_revises_then_generates_cited_answer(self):
        runner, model = self.runner(demo_script())
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.steps, 16)
        self.assertEqual(len(state.questions["q1"].findings), 2)
        self.assertFalse(state.questions["q1"].challenges[0].survives)
        self.assertTrue(state.questions["q1"].challenges[1].survives)
        self.assertEqual(state.questions["q1"].status, "conditional")
        supplied = [ctx for primitive, ctx, _ in model.contexts if primitive == "investigate"][0]
        self.assertIn(DEMO_TEXT, [p["text"] for p in supplied["evidence"]])
        self.assertGreater(len(supplied["evidence"][0]["text"]), 120)
        self.assertEqual(self.store.load().answer, state.answer)
        self.assertIn("[S1:P1](<" + DEMO_URL + ">)", self.store.report_path.read_text())
        self.assertEqual(model.script, [])

    def test_core_reasoner_and_protected_call_path_complete_the_inquiry(self):
        from cognitive_core.engine.governance import GovernancePipeline
        from cognitive_core.inquiry.reasoning import CoreReasoner
        import re

        class FixtureLLM:
            def __init__(self):
                self.remaining = demo_script()

            def invoke(self, messages):
                operation = re.search(r"OPERATION: (\w+)", messages[0].content).group(1)
                expected, response = self.remaining.pop(0)
                if operation != expected:
                    raise AssertionError("Unexpected fixture operation")
                return SimpleNamespace(content=json.dumps(response), response_metadata={})

        llm = FixtureLLM()
        governance = GovernancePipeline()
        # Exercise the real protected-call and retry path with optional modules
        # disabled; no external model, cache, or account is used by this test.
        governance._initialized = True
        reasoner = CoreReasoner(llm=llm, governance=governance)
        state = InquiryRunner(reasoner, demo_tools(), self.store).start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)
        self.assertFalse(llm.remaining)

    def test_budget_stop_and_fresh_process_resume_do_not_repeat_tools(self):
        runner, first_model = self.runner(demo_script())
        stopped = runner.start(DEMO_QUESTION, limits=InquiryLimits(max_steps=9))
        self.assertEqual(stopped.status, "budget_exhausted")
        self.assertEqual(stopped.questions["q1"].phase, "investigate")
        self.assertIsNone(stopped.answer)
        second, _ = self.runner(first_model.script)
        completed = second.resume(additional_steps=20)
        self.assertEqual(completed.status, "completed", completed.error)
        self.assertEqual((completed.usage.searches, completed.usage.reads), (1, 1))
        self.assertEqual(completed.usage.steps, 16)
        self.assertEqual(len(completed.frames), 1)

    def test_blocking_frame_clarification_suspends_without_more_calls(self):
        frame = demo_frame()
        frame["clarifications"] = [{"id": "goal", "question": "Which objective matters?", "reason": "It changes the answer."}]
        runner, model = self.runner([("frame", frame)])
        paused = runner.start(DEMO_QUESTION)
        self.assertEqual(paused.status, "awaiting_user")
        self.assertEqual(paused.usage.steps, 1)
        self.assertEqual(paused.pending_clarifications[0].id, "C1")
        waiting, unused = self.runner([])
        self.assertEqual(waiting.resume().status, "awaiting_user")
        self.assertFalse(unused.contexts)
        remaining = demo_script()
        continued, model = self.runner(remaining)
        done = continued.resume(answers={"C1": "Interpretability"})
        self.assertEqual(done.status, "completed", done.error)
        self.assertEqual(len(done.frames), 2)
        self.assertEqual(done.clarifications[0].answer, "Interpretability")
        self.assertEqual(model.contexts[0][0], "frame")

    def test_nonblocking_clarification_does_not_suspend(self):
        script = demo_script()
        script[0][1]["clarifications"] = [{"id": "optional", "question": "Any examples?", "reason": "May help presentation.", "blocking": False}]
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)

    def test_dependency_order_and_conditional_propagation(self):
        frame = demo_frame()
        first = frame["subquestions"][0]
        second = {"id": "q2", "question": "Which design follows?", "contribution": "Apply the distinction.", "depends_on": ["q1"]}
        frame["subquestions"] = [second, first]
        script = [("frame", frame)]
        for _ in range(2):
            script += [no_search(), ("investigate", conceptual_finding()),
                       ("challenge", demo_challenge()), ("reflect", demo_reflection())]
        script += ending()
        runner, model = self.runner(script, ToolRegistry())
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)
        order = [c["focus"]["id"] for op, c, _ in model.contexts if op == "investigate"]
        self.assertEqual(order, ["q1", "q2"])
        self.assertEqual(state.questions["q2"].status, "conditional")
        self.assertEqual(state.usage.searches, 0)

    def test_search_snippet_cannot_be_cited_as_read_evidence(self):
        script = demo_script()[:2] + [no_search(), ("investigate", demo_finding())]
        runner, _ = self.runner(script)
        # No source read is permitted, including the controller's bounded
        # discovery follow-through; a snippet must still fail as evidence.
        state = runner.start(DEMO_QUESTION, limits=InquiryLimits(max_reads=0))
        self.assertEqual(state.status, "failed")
        self.assertIn("not read", state.error)
        self.assertEqual(state.sources["S1"].status, "discovered")

    def test_wrong_quote_is_rejected(self):
        script = demo_script()
        script[4][1]["claims"][0]["evidence"][0]["quote"] = "This fabricated measurement is not in the source."
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "failed")
        self.assertIn("does not occur", state.error)
        self.assertFalse(state.questions["q1"].findings)

    def test_unknown_final_citation_is_rejected(self):
        script = demo_script()
        script[-1][1]["artifact"] += " Another result [S999:P1]."
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "failed")
        self.assertIn("citations", state.error)
        self.assertIsNone(state.answer)

    def test_malformed_final_citation_is_rejected(self):
        script = demo_script()
        script[-1][1]["artifact"] += " Another result [S1:P0]."
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "failed")

    def test_final_answer_cannot_invent_source_url(self):
        script = demo_script()
        script[-1][1]["artifact"] += " See https://example.org/invented-report."
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "failed")
        self.assertIn("URL", state.error)

    def test_accepting_failed_challenge_requires_response(self):
        script = demo_script()[:6] + [("reflect", demo_reflection())]
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "failed")
        self.assertIn("response to the objection", state.error)

    def test_weak_objection_can_be_explicitly_rejected(self):
        script = demo_script()[:6]
        reflection = demo_reflection()
        reflection["objection_response"] = "The objection is outside the expressly conceptual scope; no efficiency claim is retained."
        script[4] = ("investigate", demo_finding(True))
        script += [("reflect", reflection)] + ending(demo_synthesis(), "A conditional design argument. [S1:P1]")
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(len(state.questions["q1"].findings), 1)

    def test_round_limit_preserves_unresolved_branch(self):
        script = demo_script()[:7] + ending(demo_synthesis(), "A provisional design argument. [S1:P1]")
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION, limits=InquiryLimits(max_rounds_per_question=1))
        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.questions["q1"].status, "unresolved")
        self.assertIn("Revision budget reached", self.store.report_path.read_text())

    def test_research_after_challenge_runs_a_new_query(self):
        script = demo_script()[:6]
        reflection = demo_reflection()
        reflection.update(next_action="research", trajectory="revise", next_question="Look for contrary evidence.")
        plan = {"calls": [{"tool": "web_search", "arguments": {"query": "contrary evidence"}, "purpose": "Test the challenged premise"}], "rationale": "Resolve the objection."}
        script += [("reflect", reflection), ("retrieve", plan), no_search()]
        script += demo_script()[7:]
        runner, _ = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(state.usage.searches, 2)
        self.assertEqual(state.usage.reads, 1)
        self.assertEqual(len(state.sources), 1)

    def test_synthesis_can_reopen_research_even_when_frame_is_unchanged(self):
        script = demo_script()[:12]
        reflection = demo_reflection()
        reflection.update(next_action="research", trajectory="revise", next_question="Recheck evidence supporting the integrated answer.")
        script += [("reflect", reflection), ("frame", demo_frame()), no_search(),
                   ("investigate", demo_finding(True)), ("challenge", demo_challenge()), ("reflect", demo_reflection())]
        script += ending(demo_synthesis(), "A revised integration. [S1:P1]")
        runner, model = self.runner(script)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(state.status, "completed", state.error)
        self.assertEqual(len(state.frames), 2)
        self.assertEqual(state.usage.reframes, 1)
        self.assertEqual(len([op for op, _, _ in model.contexts if op == "investigate"]), 3)
        self.assertTrue(state.archived_questions)

    def test_failed_model_output_can_resume_from_its_phase(self):
        script = demo_script()
        invalid = copy.deepcopy(script[:5])
        invalid[4][1]["claims"][0]["evidence"][0]["quote"] = "Wrong quote."
        runner, _ = self.runner(invalid)
        failed = runner.start(DEMO_QUESTION)
        self.assertEqual(failed.status, "failed")
        resumed, _ = self.runner(script[4:])
        done = resumed.resume()
        self.assertEqual(done.status, "completed", done.error)
        self.assertEqual(done.usage.searches, 1)
        self.assertEqual(done.usage.reads, 1)
        self.assertEqual(done.usage.steps, 17)

    def test_zero_research_budget_still_allows_conceptual_inquiry(self):
        script = [("frame", demo_frame()), ("investigate", conceptual_finding()),
                  ("challenge", demo_challenge()), ("reflect", demo_reflection())] + ending()
        runner, _ = self.runner(script, ToolRegistry())
        state = runner.start(DEMO_QUESTION, limits=InquiryLimits(max_searches=0, max_reads=0))
        self.assertEqual(state.status, "completed", state.error)
        self.assertFalse(state.searches)

    def test_single_writer_lock_and_atomic_checkpoint(self):
        runner, _ = self.runner(demo_script())
        with self.store.locked():
            with self.assertRaisesRegex(RuntimeError, "already running"):
                runner.start(DEMO_QUESTION)
        state = runner.start(DEMO_QUESTION)
        self.assertEqual(self.store.load(), state)
        self.assertFalse(list(Path(self.tmp.name).glob(".state.json.*")))
        self.assertEqual(json.loads(self.store.state_path.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
