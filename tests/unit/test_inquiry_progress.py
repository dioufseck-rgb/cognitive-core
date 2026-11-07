"""Behavioral regressions from a live inquiry that planned without progressing."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.models import (
    FindingOutput, InquiryEvent, InquiryLimits, InquiryReflection, InquiryState,
    Passage, QuestionRecord, ReadArguments, ReadCall, SearchArguments, SearchCall,
    SearchRecord, SourceRecord,
)
from cognitive_core.inquiry.runner import InquiryRunner
from cognitive_core.inquiry.store import InquiryStore
from cognitive_core.primitives.schemas import ChallengeOutput, FrameOutput, FrameQuestion


def base(**kwargs):
    return {"confidence": 0.5, "reasoning": "Synthetic regression fixture.", **kwargs}


def frame(count=1, dependencies=None):
    dependencies = dependencies or {}
    return FrameOutput(**base(
        working_question="When is A preferable to B?", purpose="Choose a design.",
        subquestions=[FrameQuestion(
            id=f"q{i}", question=f"What evidence informs design dimension {i}?",
            contribution=f"Compare the alternatives on dimension {i}.",
            depends_on=dependencies.get(f"q{i}", []),
        ) for i in range(1, count + 1)],
        answer_criteria=["Explain the conditions and limits."],
    ))


def state_for(plan=None, **kwargs):
    plan = plan or frame()
    return InquiryState(
        original_question=plan.working_question, frames=[plan], phase="work",
        questions={q.id: QuestionRecord(specification=q, frame_version=1)
                   for q in plan.subquestions}, **kwargs,
    )


def finding(text="A may help under stated assumptions."):
    return FindingOutput(**base(
        finding=text,
        claims=[{"text": "The effect depends on the operating conditions.", "basis": "inference"}],
        unresolved_issues=["The effect is not yet measured."],
    ))


def reflection(**kwargs):
    values = base(
        next_action="research", what_changed="The empirical gap remains.",
        next_question="Find a measured comparison.", trajectory="continue",
    )
    values.update(kwargs)
    return InquiryReflection(**values)


class AdaptiveFixtureReasoner:
    """Choose responses by operation and branch, without fixing execution order."""

    def __init__(self, plan=None, research="none", accept_after=2):
        self.plan = plan or frame()
        self.research = research
        self.accept_after = accept_after
        self.rounds = defaultdict(int)
        self.calls = []
        self.contexts = []

    def invoke(self, primitive, *, context, instructions, response_model):
        qid = context["focus"]["id"] if context.get("focus") else None
        self.calls.append((primitive, qid, self.rounds[qid]))
        self.contexts.append((primitive, deepcopy(context), instructions))
        if primitive == "frame":
            output = self.plan
        elif primitive == "retrieve":
            search = self.research == "repeat_search" or (
                self.research == "search_then_stop" and not context.get("search_history")
            )
            output = {
                "calls": [{"tool": "web_search", "arguments": {"query": "shared comparison evidence", "num": 3},
                           "purpose": "Find a measured comparison."}] if search else [],
                "rationale": "The fixture requests evidence." if search else "Continue with the remaining gap explicit.",
            }
        elif primitive == "investigate":
            self.rounds[qid] += 1
            output = finding(
                "The condition needs to be stated explicitly."
                if self.rounds[qid] == 1 else
                "The design depends on the stated operating condition; its effect is unmeasured."
            )
        elif primitive == "challenge":
            output = base(
                survives=qid is None or self.accept_after is not None and self.rounds[qid] >= self.accept_after,
                overall_assessment="Retain the empirical limitation and examine the stated condition.",
            )
        elif primitive == "reflect":
            accepted = qid is None or self.accept_after is not None and self.rounds[qid] >= self.accept_after
            output = reflection(
                next_action="accept" if accepted else "research" if self.research != "none" else "revise",
                resolution="conditional",
                what_changed="The condition is now explicit." if accepted else "The empirical gap remains.",
            )
        elif primitive == "deliberate":
            output = base(
                situation_summary="The branches retain their qualifications.",
                conclusion="The comparison remains conditional on the operating conditions.",
                claims=[{"text": "The operating conditions determine the appropriate comparison.", "basis": "inference"}],
                limitations=["The fixture establishes no measured performance benefit."],
            )
        elif primitive == "generate":
            output = base(artifact="The comparison remains conditional; no measured benefit is established.", format="markdown")
        else:
            raise AssertionError(f"Unexpected operation: {primitive}")
        return response_model.model_validate(output)


class InquiryProgressTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = InquiryStore(Path(self.directory.name) / "study")

    def tools(self, read_fails=False):
        searches, reads = [], []
        tools = ToolRegistry()

        def search(args):
            searches.append(args["query"])
            return {"results": [
                {"url": f"https://example.org/ranked/{i}", "title": f"Ranked source {i}",
                 "snippet": "A candidate comparison; not read evidence."}
                for i in range(1, 4)
            ]}

        def read(args):
            reads.append(args["url"])
            if read_fails:
                raise RuntimeError("Synthetic source is unavailable")
            return {"url": args["url"], "title": "Candidate comparison",
                    "text": "The fixture describes conditions but establishes no measured advantage.",
                    "content_type": "text/plain", "truncated": False}

        tools.register("web_search", search)
        tools.register("read_source", read)
        return tools, searches, reads

    def test_independent_branches_are_reviewed_before_a_second_round(self):
        model = AdaptiveFixtureReasoner(frame(3))
        runner = InquiryRunner(model, ToolRegistry(), self.store)
        result = runner.start("Compare A and B.", limits=InquiryLimits(max_steps=40, max_searches=0, max_reads=0))

        self.assertEqual(result.status, "completed", result.error)
        investigations = [qid for op, qid, _ in model.calls if op == "investigate"]
        self.assertEqual(set(investigations[:3]), {"q1", "q2", "q3"})
        first_second_round = next(i for i, (op, _, rounds) in enumerate(model.calls)
                                  if op == "investigate" and rounds == 1)
        for qid in ("q1", "q2", "q3"):
            investigated = next(i for i, call in enumerate(model.calls) if call == ("investigate", qid, 0))
            challenged = next(i for i, call in enumerate(model.calls) if call == ("challenge", qid, 1))
            reflected = next(i for i, call in enumerate(model.calls) if call == ("reflect", qid, 1))
            self.assertLess(investigated, challenged)
            self.assertLess(challenged, reflected)
            self.assertLess(reflected, first_second_round)

    def test_fairness_preserves_real_dependencies(self):
        model = AdaptiveFixtureReasoner(frame(3, dependencies={"q2": ["q1"]}))
        runner = InquiryRunner(model, ToolRegistry(), self.store)
        result = runner.start("Compare A and B.", limits=InquiryLimits(max_steps=40, max_searches=0, max_reads=0))

        self.assertEqual(result.status, "completed", result.error)
        dependency_closed = model.calls.index(("reflect", "q1", 2))
        dependent_started = model.calls.index(("investigate", "q2", 0))
        self.assertLess(dependency_closed, dependent_started)
        self.assertEqual(result.questions["q2"].status, "conditional")

    def test_cached_plans_are_bounded_and_do_not_repeat_network_calls(self):
        model = AdaptiveFixtureReasoner(research="repeat_search", accept_after=None)
        tools, searches, reads = self.tools()
        runner = InquiryRunner(model, tools, self.store)
        result = runner.start("Compare A and B.", limits=InquiryLimits(max_steps=40, max_searches=6, max_reads=8))

        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(searches, ["shared comparison evidence"])
        self.assertEqual(reads, ["https://example.org/ranked/1"])
        self.assertEqual((result.usage.searches, result.usage.reads), (1, 1))
        plans = Counter(rounds for op, qid, rounds in model.calls if op == "retrieve" and qid == "q1")
        self.assertLessEqual(plans[0], 3)  # One useful plan and at most two stalled plans.
        self.assertTrue(all(count <= 2 for rounds, count in plans.items() if rounds > 0))
        self.assertEqual(result.questions["q1"].status, "unresolved")
        self.assertIn("not yet measured", result.questions["q1"].findings[-1].unresolved_issues[0])
        retrieval_contexts = [context for op, context, _ in model.contexts if op == "retrieve"]
        self.assertTrue(any(context["branch_progress"]["retrieval_feedback"] for context in retrieval_contexts))

    def test_initial_discovery_is_read_once_when_planner_stops(self):
        model = AdaptiveFixtureReasoner(research="search_then_stop", accept_after=1)
        tools, searches, reads = self.tools()
        result = InquiryRunner(model, tools, self.store).start("Compare A and B.")

        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(len(searches), 1)
        self.assertEqual(reads, ["https://example.org/ranked/1"])
        first_investigation = next(context for op, context, _ in model.contexts if op == "investigate")
        self.assertTrue(first_investigation["evidence"])
        self.assertEqual(first_investigation["evidence"][0]["source_id"], "S1")
        self.assertEqual(sum(event.kind == "initial_source_read" for event in result.events), 1)

    def test_cached_discovery_can_supply_an_initial_read_to_another_branch(self):
        state = state_for(frame(2))
        state.questions["q1"].status = "unresolved"
        state.questions["q1"].phase = "done"
        for i in range(1, 4):
            state.sources[f"S{i}"] = SourceRecord(id=f"S{i}", url=f"https://example.org/ranked/{i}",
                                                  question_ids=["q1"])
        state.searches.append(SearchRecord(query="shared comparison evidence", num=3, question_id="q1",
                                           purpose="Find a measured comparison.", source_ids=["S1", "S2", "S3"]))
        state.usage.searches = 1
        self.store.save(state)
        model = AdaptiveFixtureReasoner(research="repeat_search", accept_after=1)
        tools, searches, reads = self.tools()

        result = InquiryRunner(model, tools, self.store).resume()

        self.assertEqual(result.status, "completed", result.error)
        self.assertFalse(searches)
        self.assertEqual(reads, ["https://example.org/ranked/1"])
        self.assertEqual((result.usage.searches, result.usage.reads), (1, 1))
        self.assertIn("q2", result.sources["S1"].question_ids)
        self.assertEqual(result.sources["S1"].status, "read")

    def test_mixed_plan_retains_feasible_read_and_filters_redundant_work(self):
        state = state_for(frame(2), limits=InquiryLimits(max_searches=1, max_reads=1))
        state.usage.searches = 1
        state.sources["S1"] = SourceRecord(id="S1", url="https://example.org/ranked/1", question_ids=["q2"])
        state.searches.append(SearchRecord(query="cached comparison", num=3, question_id="q2",
                                           purpose="Find a comparison.", source_ids=["S1"]))
        calls = [
            SearchCall(tool="web_search", arguments=SearchArguments(query="cached comparison", num=3),
                       purpose="Use available discovery results."),
            ReadCall(tool="read_source", arguments=ReadArguments(url="https://example.org/ranked/1"),
                     purpose="Read the candidate evidence."),
            ReadCall(tool="read_source", arguments=ReadArguments(url="https://example.org/ranked/1"),
                     purpose="Duplicate request for the same candidate."),
            SearchCall(tool="web_search", arguments=SearchArguments(query="new comparison", num=3),
                       purpose="Request a search after the allowance is used."),
        ]
        tools, searches, reads = self.tools()
        runner = InquiryRunner(AdaptiveFixtureReasoner(), tools, self.store)
        q = state.questions["q1"]

        pending = runner._new_retrieval_calls(state, q, calls)

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].tool, "read_source")
        self.assertEqual(pending[0].arguments.url, "https://example.org/ranked/1")
        self.assertIn("q1", state.sources["S1"].question_ids)
        self.assertFalse(searches)
        self.assertFalse(reads)
        for skipped in ("cached", "duplicate", "budget"):
            self.assertIn(skipped, q.research_feedback.casefold())
        q.pending_calls = pending
        q.phase = "retrieve"
        runner._question_step(state, q)
        self.assertEqual(reads, ["https://example.org/ranked/1"])
        self.assertFalse(searches)
        self.assertEqual((state.usage.searches, state.usage.reads), (1, 1))
        self.assertFalse(q.pending_calls)

    def test_failed_initial_read_is_not_retried_by_fallback(self):
        model = AdaptiveFixtureReasoner(research="search_then_stop", accept_after=None)
        tools, _, reads = self.tools(read_fails=True)
        result = InquiryRunner(model, tools, self.store).start("Compare A and B.")

        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(reads, ["https://example.org/ranked/1"])
        self.assertEqual(result.usage.reads, 1)
        self.assertEqual(result.sources["S1"].status, "failed")
        self.assertEqual(result.questions["q1"].status, "unresolved")

    def test_initial_read_fallback_respects_scope_budget_and_prior_attempt(self):
        for scenario in ("unrelated_branch", "no_read_budget", "already_attempted", "later_round"):
            with self.subTest(scenario=scenario):
                state = state_for(frame(2), limits=InquiryLimits(max_reads=0 if scenario == "no_read_budget" else 8))
                q = state.questions["q1"]
                owner = "q2" if scenario == "unrelated_branch" else "q1"
                state.sources["S1"] = SourceRecord(id="S1", url="https://example.org/candidate", question_ids=[owner])
                state.searches.append(SearchRecord(query="comparison evidence", num=1, question_id=owner,
                                                   purpose="Find a comparison.", source_ids=["S1"]))
                if scenario == "already_attempted":
                    state.events.append(InquiryEvent(sequence=1, step=1, kind="read_source",
                                                     question_id="q1", summary="Source could not be read"))
                if scenario == "later_round":
                    q.rounds = 1
                runner = InquiryRunner(AdaptiveFixtureReasoner(), ToolRegistry(), self.store)

                runner._question_step(state, q)

                self.assertEqual(q.phase, "investigate")
                self.assertFalse(q.pending_calls)
                self.assertFalse(any(event.kind == "initial_source_read" for event in state.events))
                self.assertEqual(state.usage.reads, 0)

    def test_catalog_keeps_search_rank_within_newest_relevant_batch(self):
        state = state_for(frame(2))
        for i in range(1, 7):
            state.sources[f"S{i}"] = SourceRecord(id=f"S{i}", url=f"https://example.org/source/{i}",
                                                  question_ids=["q2" if i == 6 else "q1"])
        for index, (owner, ids) in enumerate((
            ("q1", ["S1", "S2", "S3"]), ("q1", ["S4", "S5"]), ("q2", ["S6"]),
        )):
            state.searches.append(SearchRecord(query=f"query {index}", num=len(ids), question_id=owner,
                                               purpose="Resolve this branch.", source_ids=ids))
        runner = InquiryRunner(AdaptiveFixtureReasoner(), ToolRegistry(), self.store)

        catalog = runner._context(state, state.questions["q1"])["source_catalog"]

        self.assertEqual([source["id"] for source in catalog], ["S4", "S5", "S1", "S2", "S3", "S6"])

    def test_context_exposes_recent_progress_and_actionable_guidance(self):
        state = state_for(limits=InquiryLimits(max_rounds_per_question=5))
        q = state.questions["q1"]
        q.findings = [finding(f"Tentative finding {i}.") for i in range(1, 4)]
        q.rounds = 3
        q.research_plans = 1
        q.research_feedback = "The last requested source was already read."
        q.reflections = [reflection(what_changed=f"Recorded change {i}.") for i in range(1, 4)]
        q.reflections[-1] = reflection(
            what_changed="Recorded change 3.", open_questions=["Which comparison measured the effect?"],
            what_was_assumed=["The workload is representative."], revision_target="The claimed performance benefit",
            established_facts_to_skip=["Both designs can return citations."],
        )
        state.sources["S1"] = SourceRecord(id="S1", url="https://example.org/read", status="read",
                                           question_ids=["q1"], passages=[Passage(id="S1:P1", text="A bounded source passage.")])
        state.sources["S2"] = SourceRecord(id="S2", url="https://example.org/unread", question_ids=["q1"])
        for i in range(1, 4):
            state.events.append(InquiryEvent(sequence=i, step=i, kind="retrieval_plan", question_id="q1",
                                             summary="Plan recorded", details={"rationale": f"Plan {i}", "calls": []}))
        runner = InquiryRunner(AdaptiveFixtureReasoner(), ToolRegistry(), self.store)
        runner._apply_reflection(state, q, q.reflections[-1])

        context = runner._context(state, q)

        progress = context["branch_progress"]
        self.assertEqual((progress["completed_rounds"], progress["round_limit"]), (3, 5))
        self.assertEqual((progress["research_plans_this_round"], progress["research_plan_limit"]), (1, 4))
        self.assertEqual((progress["read_sources"], progress["unread_sources"]), (1, 1))
        self.assertEqual(progress["retrieval_feedback"], q.research_feedback)
        self.assertEqual([entry["finding"] for entry in context["recent_findings"]],
                         ["Tentative finding 2.", "Tentative finding 3."])
        self.assertEqual([entry["what_changed"] for entry in context["recent_reflections"]],
                         ["Recorded change 2.", "Recorded change 3."])
        self.assertEqual([entry["rationale"] for entry in context["recent_retrieval_plans"]], ["Plan 2", "Plan 3"])
        for required in ("Which comparison measured the effect?", "The workload is representative.",
                         "The claimed performance benefit", "Both designs can return citations."):
            self.assertIn(required, context["instructions"])

    def test_small_context_keeps_current_work_and_read_passage(self):
        plan = FrameOutput(**base(
            working_question="When is A preferable to B?", purpose="Choose a design.",
            subquestions=[FrameQuestion(id="q1", question="What evidence favors A or B?",
                                        contribution="Compare the alternatives.")],
            answer_criteria=["Explain the conditions and limits."],
        ))
        state = state_for(plan, limits=InquiryLimits(max_context_chars=4000))
        q = state.questions["q1"]
        q.rounds = 2
        q.findings = [finding(), finding()]
        q.challenges = [ChallengeOutput(**base(survives=False, overall_assessment="The effect has no measured support."))
                        for _ in range(2)]
        q.reflections = [reflection(), reflection()]
        state.sources["S1"] = SourceRecord(id="S1", url="https://example.org/read", status="read", question_ids=["q1"],
                                           passages=[Passage(id="S1:P1", text="No comparative measurement is reported.")])
        runner = InquiryRunner(AdaptiveFixtureReasoner(), ToolRegistry(), self.store)

        context = runner._context(state, q)

        self.assertLessEqual(len(json.dumps(context)), 4000)
        self.assertEqual(context["focus"]["id"], "q1")
        self.assertEqual(context["branch_progress"]["completed_rounds"], 2)
        self.assertFalse(context["current_challenge"]["survives"])
        self.assertIn("S1:P1", [passage["id"] for passage in context["evidence"]])

    def test_existing_checkpoint_without_progress_fields_loads(self):
        state = state_for()
        serialized = state.model_dump(mode="json")
        serialized["questions"]["q1"].pop("stalled_research_plans")
        serialized["questions"]["q1"].pop("research_feedback")
        self.store.directory.mkdir(parents=True)
        self.store.state_path.write_text(json.dumps(serialized), encoding="utf-8")

        loaded = self.store.load()

        self.assertEqual(loaded.questions["q1"].stalled_research_plans, 0)
        self.assertEqual(loaded.questions["q1"].research_feedback, "")
        self.assertEqual(loaded.original_question, state.original_question)


if __name__ == "__main__":
    unittest.main()
