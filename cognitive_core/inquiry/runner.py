"""Adaptive inquiry coordination, separate from institutional decision closure.

LLMs propose frames, findings, objections and next steps. This controller owns
the lifecycle, budgets, dependency scheduling, actual tools and evidence checks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib.parse import urldefrag

from pydantic import BaseModel, ValidationError

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.models import (
    FindingOutput,
    InquiryEvent,
    InquiryLimits,
    InquiryReflection,
    InquiryState,
    OutputRepairState,
    Passage,
    QuestionRecord,
    ReadArguments,
    ReadCall,
    RetrievalPlan,
    SearchRecord,
    SourceRecord,
    SynthesisOutput,
    UserClarification,
)
from cognitive_core.inquiry.store import InquiryStore
from cognitive_core.inquiry.reasoning import ReasoningError, StructuredOutputError, schema_error
from cognitive_core.primitives.schemas import ChallengeOutput, FrameOutput, GenerateOutput


DONE = {"answered", "conditional", "unresolved"}
CITATION = re.compile(r"\[(S[1-9][0-9]*:P[1-9][0-9]*)\]")
URL = re.compile(r"https?://[^\s<>\]\)\"']+")


class InquiryError(ValueError):
    """A recoverable, safe-to-display inquiry contract failure."""


class _BudgetExhausted(Exception):
    pass


class InquiryRunner:
    def __init__(
        self,
        reasoner,
        tools: ToolRegistry,
        store: InquiryStore,
        on_event: Callable[[InquiryEvent], None] | None = None,
    ):
        self.reasoner = reasoner
        self.tools = tools
        self.store = store
        self.on_event = on_event

    def start(
        self,
        question: str,
        *,
        purpose: str = "",
        context: str = "",
        limits: InquiryLimits | None = None,
    ) -> InquiryState:
        with self.store.locked():
            if self.store.state_path.exists():
                raise InquiryError("An inquiry already exists here; use resume or a new output directory")
            state = InquiryState(
                original_question=question,
                purpose=purpose,
                user_context=context,
                limits=limits or InquiryLimits(),
            )
            for url in URL.findall(question + "\n" + context):
                self._source(state, url.rstrip(".,;"), "User-supplied source", "", "user")
            self.store.save(state)
            return self._run(state)

    def resume(
        self,
        *,
        answers: dict[str, str] | None = None,
        additional_steps: int = 0,
        additional_searches: int = 0,
        additional_reads: int = 0,
    ) -> InquiryState:
        with self.store.locked():
            state = self.store.load()
            additions = {
                "max_steps": additional_steps,
                "max_searches": additional_searches,
                "max_reads": additional_reads,
            }
            if any(v < 0 for v in additions.values()):
                raise InquiryError("Additional budgets must be nonnegative")
            limits = state.limits.model_dump()
            for key, value in additions.items():
                limits[key] += value
            state.limits = InquiryLimits.model_validate(limits)
            known = {c.id: c for c in state.clarifications if c.active and c.answer is None}
            for key, value in (answers or {}).items():
                if key not in known:
                    raise InquiryError("Unknown or already answered clarification ID; use status to list pending questions")
                if not value.strip():
                    raise InquiryError("A clarification answer cannot be empty")
            for key, value in (answers or {}).items():
                clarification = known[key]
                clarification.answer = value.strip()
                for url in URL.findall(value):
                    self._source(state, url.rstrip(".,;"), "User-supplied source", "", clarification.question_id or "user")
                self._event(state, "user_answer", "User supplied a clarification", clarification.question_id)
                if clarification.question_id is None:
                    for q in list(state.questions.values()):
                        state.archived_questions.append(q.model_copy(deep=True))
                        state.questions[q.specification.id] = QuestionRecord(
                            specification=q.specification, frame_version=q.frame_version,
                            instructions="Reconsider using the user's new clarification.",
                        )
                    state.synthesis = None
                    state.synthesis_challenge = None
                    state.synthesis_reflection = None
                    state.synthesis_rounds = 0
                    state.answer = None
                    state.phase = "frame"
                elif clarification.question_id in state.questions:
                    self._invalidate(state, clarification.question_id)
            if any(known[key].question_id is None for key in (answers or {})):
                state.phase = "frame"
            if state.status == "completed" and not answers:
                return state
            if state.pending_clarifications:
                state.status = "awaiting_user"
                self._checkpoint(state)
                return state
            if answers:
                state.output_repair = None
            elif state.status == "failed" and state.output_repair is not None:
                # An explicit retry after failure can use a changed model or
                # configuration. Budget/interrupt pauses keep the original count.
                state.output_repair.failures = 0
                self._event(state, "repair_restarted", "Explicit resume starts a new bounded correction batch",
                            state.output_repair.question_id)
            state.status = "ready"
            state.error = None
            state.stop_reason = ""
            return self._run(state)

    def _run(self, state: InquiryState) -> InquiryState:
        if state.in_flight:
            self._event(
                state, "recovery",
                "Resuming an interrupted step; the read-only operation may be repeated",
                details=state.in_flight,
            )
            state.in_flight = None
        state.status = "running"
        try:
            while state.status == "running":
                if state.pending_clarifications:
                    state.status = "awaiting_user"
                    break
                if state.usage.steps >= state.limits.max_steps:
                    raise _BudgetExhausted()
                self._advance(state)
                self._checkpoint(state)
        except _BudgetExhausted:
            state.status = "budget_exhausted"
            state.stop_reason = "Step budget exhausted; the saved findings remain provisional"
            self._event(state, "budget_exhausted", state.stop_reason)
        except KeyboardInterrupt:
            state.status = "ready"
            state.stop_reason = "Interrupted; resume continues from the last completed checkpoint"
            self._checkpoint(state)
            raise
        except Exception as exc:
            state.status = "failed"
            if isinstance(exc, InquiryError):
                state.error = str(exc)
            elif isinstance(exc, ValidationError):
                fields = [".".join(map(str, e["loc"])) for e in exc.errors(include_input=False)[:5]]
                state.error = "Invalid structured output at: " + ", ".join(fields)
            elif isinstance(exc, ReasoningError):
                state.error = str(exc)
            else:
                state.error = f"{type(exc).__name__} during the current step; check configuration and resume"
            state.in_flight = None
            self._event(state, "failed", state.error)
        self._checkpoint(state)
        return state

    def _advance(self, state: InquiryState) -> None:
        if state.phase == "frame":
            self._frame(state)
        elif state.phase == "work":
            ready = [
                q for q in state.questions.values()
                if q.status not in DONE
                and all(state.questions[d].status in DONE for d in q.specification.depends_on)
            ]
            if not ready:
                if all(q.status in DONE for q in state.questions.values()):
                    state.phase = "synthesize"
                else:
                    raise InquiryError("Question dependencies cannot progress; reframe the inquiry")
                return
            # Finish reviewing a finding before switching branches. Between
            # reviewed rounds, give independent branches their first pass before
            # spending another round on a branch that has already had one.
            reviewing = [q for q in ready if q.phase in {"challenge", "reflect"}]
            self._question_step(state, reviewing[0] if reviewing else min(ready, key=lambda q: q.rounds))
        elif state.phase == "synthesize":
            context = self._context(state)
            output = self._invoke(state, "deliberate", SynthesisOutput, context)
            self._validate_claims(output, context)
            state.synthesis = output
            state.synthesis_rounds += 1
            state.phase = "review_synthesis"
        elif state.phase == "review_synthesis":
            state.synthesis_challenge = self._invoke(
                state, "challenge", ChallengeOutput, self._context(state),
                "Review the whole synthesis: interactions, contradictions, missing branches, and citations.",
            )
            state.phase = "reflect_synthesis"
        elif state.phase == "reflect_synthesis":
            reflection = self._invoke(state, "reflect", InquiryReflection, self._context(state))
            self._validate_acceptance(reflection, state.synthesis_challenge)
            state.synthesis_reflection = reflection
            self._apply_synthesis_reflection(state, reflection)
        elif state.phase == "generate":
            context = self._context(state)
            output = self._invoke(state, "generate", GenerateOutput, context)
            if not isinstance(output.artifact, str) or not output.artifact.strip():
                raise InquiryError("Generate must return a nonempty Markdown artifact")
            self._validate_answer(output.artifact, state, context)
            state.answer = output.artifact.strip()
            state.phase = "done"
            state.status = "completed"
            if not state.stop_reason:
                state.stop_reason = "Reviewed synthesis completed within the configured budget"
            self._event(state, "completed", state.stop_reason)
        elif state.phase == "done":
            state.status = "completed"

    def _frame(self, state: InquiryState) -> None:
        frame = self._invoke(state, "frame", FrameOutput, self._context(state))
        previous = state.frame
        version = len(state.frames) + 1
        old = state.questions
        unchanged = set()
        if previous and (previous.working_question, previous.purpose, previous.scope, previous.assumptions, previous.answer_criteria) == (
            frame.working_question, frame.purpose, frame.scope, frame.assumptions, frame.answer_criteria
        ):
            unchanged = {
                spec.id for spec in frame.subquestions
                if spec.id in old and old[spec.id].specification == spec
            }
            # A changed prerequisite invalidates all dependent findings.
            while True:
                reduced = {qid for qid in unchanged if set(old[qid].specification.depends_on) <= unchanged}
                if reduced == unchanged:
                    break
                unchanged = reduced
        state.frames.append(frame)
        state.questions = {}
        for spec in frame.subquestions:
            if spec.id in unchanged:
                record = old[spec.id]
                record.frame_version = version
            else:
                record = QuestionRecord(specification=spec, frame_version=version)
            state.questions[spec.id] = record
        state.archived_questions.extend(q for key, q in old.items() if key not in unchanged)
        state.synthesis = None
        state.synthesis_challenge = None
        state.synthesis_reflection = None
        state.synthesis_rounds = 0
        state.answer = None
        # Unanswered questions from an old frame no longer block a new one.
        for c in state.clarifications:
            if c.answer is None:
                c.active = False
        for c in frame.clarifications:
            self._ask(state, c.question, c.reason, None, c.blocking)
        state.phase = "work"
        self._event(state, "framed", f"Frame {version}: {len(frame.subquestions)} subquestions")

    def _question_step(self, state: InquiryState, q: QuestionRecord) -> None:
        qid = q.specification.id
        q.status = "active"
        context = self._context(state, q)
        if q.phase == "plan":
            if (
                q.research_plans >= state.limits.max_research_plans_per_round
                or (state.usage.searches >= state.limits.max_searches and state.usage.reads >= state.limits.max_reads)
            ):
                q.instructions += "\nResearch allowance reached. Preserve any remaining evidence gaps."
                self._finish_research(state, q)
                return
            plan = self._invoke(state, "retrieve", RetrievalPlan, context, q.instructions, qid)
            q.research_plans += 1
            self._event(state, "retrieval_plan", "Retrieval plan recorded", qid, {
                "round": q.rounds + 1, "rationale": plan.rationale,
                "calls": [call.model_dump(mode="json") for call in plan.calls],
            })
            q.pending_calls = self._new_retrieval_calls(state, q, plan.calls)
            if q.pending_calls:
                q.stalled_research_plans = 0
                q.phase = "retrieve"
            elif not plan.calls:
                self._finish_research(state, q)
            else:
                q.stalled_research_plans += 1
                self._event(state, "research_stalled", q.research_feedback, qid)
                if q.stalled_research_plans >= 2:
                    q.instructions += "\nRepeated retrieval plans fetched no new evidence. Preserve unresolved gaps."
                    self._finish_research(state, q)
        elif q.phase == "retrieve":
            if not q.pending_calls:
                q.phase = "plan"
                return
            call = q.pending_calls[0]
            self._retrieve(state, q, call)
            q.pending_calls.pop(0)
            if not q.pending_calls:
                q.phase = "plan"
        elif q.phase == "investigate":
            if q.rounds >= state.limits.max_rounds_per_question:
                self._defer(q, "Revision budget reached before the question was resolved")
                return
            finding = self._invoke(state, "investigate", FindingOutput, context, q.instructions, qid)
            self._validate_claims(finding, context)
            q.findings.append(finding)
            q.rounds += 1
            q.research_plans = 0
            q.stalled_research_plans = 0
            q.phase = "challenge"
        elif q.phase == "challenge":
            q.challenges.append(self._invoke(state, "challenge", ChallengeOutput, context, "", qid))
            q.phase = "reflect"
        elif q.phase == "reflect":
            reflection = self._invoke(state, "reflect", InquiryReflection, context, "", qid)
            self._validate_acceptance(reflection, q.challenges[-1] if q.challenges else None)
            q.reflections.append(reflection)
            self._apply_reflection(state, q, reflection)

    def _apply_reflection(self, state: InquiryState, q: QuestionRecord, reflection: InquiryReflection) -> None:
        action = reflection.next_action
        q.instructions = self._guidance(reflection)
        if action == "accept":
            q.status = reflection.resolution
            # Material unresolved dependencies remain visible in downstream conclusions.
            if any(state.questions[d].status in {"conditional", "unresolved"} for d in q.specification.depends_on):
                q.status = "conditional"
            q.phase = "done"
            q.unresolved_reason = "; ".join(reflection.open_questions)
        elif action == "defer":
            self._defer(q, reflection.escalation_reason or reflection.what_changed)
        elif action == "ask_user":
            c = reflection.clarification
            if not self._ask(state, c.question, c.reason, q.specification.id):
                self._defer(q, "The requested clarification was already answered; the branch remains unresolved")
            else:
                q.phase = "plan"
        elif action == "reframe":
            self._request_reframe(state, q.instructions, q)
        elif q.rounds >= state.limits.max_rounds_per_question:
            self._defer(q, "Revision budget reached before the remaining concerns were resolved")
        else:
            q.phase = "plan" if action == "research" else "investigate"

    def _apply_synthesis_reflection(self, state: InquiryState, reflection: InquiryReflection) -> None:
        state.instructions = self._guidance(reflection)
        action = reflection.next_action
        if action == "accept":
            state.phase = "generate"
        elif action == "ask_user":
            c = reflection.clarification
            if not self._ask(state, c.question, c.reason, None):
                self._finish_conditionally(state, "The synthesis requested an already answered clarification")
        elif action in {"reframe", "research"}:
            # New evidence needs in an integrated answer reopen the question plan.
            self._request_reframe(state, state.instructions)
        elif action == "revise" and state.synthesis_rounds < state.limits.max_rounds_per_question:
            state.phase = "synthesize"
        else:
            self._finish_conditionally(state, reflection.escalation_reason or reflection.what_changed)

    def _request_reframe(self, state: InquiryState, guidance: str, q: QuestionRecord | None = None) -> None:
        if state.usage.reframes >= state.limits.max_reframes:
            if q:
                self._defer(q, "Reframing budget reached; " + guidance)
            else:
                self._finish_conditionally(state, "Reframing budget reached; " + guidance)
            return
        state.usage.reframes += 1
        state.instructions = guidance
        if q:
            q.phase = "plan"
            q.research_plans = 0
        else:
            # A synthesis that needs new research must actually reopen work,
            # even if Frame preserves the same question IDs and wording.
            for old in list(state.questions.values()):
                state.archived_questions.append(old.model_copy(deep=True))
                state.questions[old.specification.id] = QuestionRecord(
                    specification=old.specification, frame_version=old.frame_version,
                    instructions=guidance,
                )
        state.phase = "frame"

    def _finish_conditionally(self, state: InquiryState, reason: str) -> None:
        state.stop_reason = "Completed with unresolved synthesis concerns: " + reason
        if state.synthesis and reason not in state.synthesis.limitations:
            state.synthesis.limitations.append(reason)
        state.phase = "generate"

    @staticmethod
    def _guidance(reflection: InquiryReflection) -> str:
        guidance = list(filter(None, [
            reflection.what_changed, reflection.escalation_reason,
            reflection.next_question, reflection.hypothesis,
            reflection.template_guidance, reflection.objection_response,
        ]))
        if reflection.revision_target:
            guidance.append("Revision target: " + reflection.revision_target)
        for label, values in (
            ("Open questions", reflection.open_questions),
            ("Assumptions to examine", reflection.what_was_assumed),
            ("Established facts to avoid repeating", reflection.established_facts_to_skip),
        ):
            if values:
                guidance.append(label + ": " + "; ".join(values))
        return "\n".join(guidance)

    def _new_retrieval_calls(self, state, q, calls):
        """Keep useful tool work and explain cached or unavailable requests.

        Reusing a cached result can associate it with this branch, but it is not
        new evidence and must not occupy an entire round of planning calls.
        """
        pending = []
        seen = set()
        skipped = []
        remaining = {
            "web_search": state.limits.max_searches - state.usage.searches,
            "read_source": state.limits.max_reads - state.usage.reads,
        }
        for call in calls:
            args = call.arguments.model_dump()
            src = self._find_source(state, args["url"]) if call.tool == "read_source" else None
            key = (call.tool, src.id if src else json.dumps(args, sort_keys=True).casefold())
            if key in seen:
                skipped.append("Duplicate call in this plan")
                continue
            seen.add(key)
            cached = None
            if call.tool == "web_search":
                cached = self._cached_search(state, args)
            elif src is None:
                raise InquiryError("read_source URL must come from search results or user-supplied context")
            if cached or (src is not None and src.status == "read"):
                # This path only associates known data; it cannot perform IO.
                self._retrieve(state, q, call)
                skipped.append("Requested search or source is already cached")
                continue
            if remaining[call.tool] <= 0:
                skipped.append("No remaining budget for " + call.tool)
                continue
            remaining[call.tool] -= 1
            pending.append(call)
        q.research_feedback = (
            "; ".join(dict.fromkeys(skipped))
            + ". These calls supply no new evidence. Read a different useful discovered source, "
              "make a different targeted search, or proceed with the gap explicit."
        ) if skipped else ""
        return pending

    def _finish_research(self, state: InquiryState, q: QuestionRecord) -> None:
        """Follow through on initial discovery with at most one fallback read.

        Search rank is only a selection fallback, never an authority judgment.
        The normal read path still accounts for budget, errors and checkpoints.
        Conceptual branches and branches that already attempted a read are left
        to the planner's judgment.
        """
        qid = q.specification.id
        discovered_ids = {sid for search in state.searches if not search.error for sid in search.source_ids}
        has_discovery = any(s.id in discovered_ids and qid in s.question_ids for s in state.sources.values())
        has_read = any(s.status == "read" and qid in s.question_ids for s in state.sources.values())
        attempted_read = any(e.kind == "read_source" and e.question_id == qid for e in state.events)
        if (
            q.rounds == 0 and has_discovery and not has_read and not attempted_read
            and state.usage.reads < state.limits.max_reads
        ):
            candidates = [s for s in self._ordered_sources(state, q)
                          if s.status == "discovered" and qid in s.question_ids]
            if candidates:
                q.pending_calls = [ReadCall(
                    tool="read_source", arguments=ReadArguments(url=candidates[0].url),
                    purpose="Inspect a discovered candidate before the initial finding; assess its relevance and limitations.",
                )]
                q.research_plans = state.limits.max_research_plans_per_round
                q.phase = "retrieve"
                self._event(state, "initial_source_read", "Initial discovery has no read evidence; inspect one ranked candidate", qid,
                            {"source_id": candidates[0].id})
                return
        q.phase = "investigate"

    @staticmethod
    def _cached_search(state: InquiryState, args: dict):
        return next((s for s in state.searches if s.error is None
                     and s.query.casefold() == args["query"].casefold() and s.num == args["num"]), None)

    @staticmethod
    def _ordered_sources(state: InquiryState, focus: QuestionRecord | None = None, required=()):
        # Prefer recent relevant searches, preserving each provider result list's
        # ranking instead of reversing S1..S5 into S5..S1.
        ranks = {}
        for index in range(len(state.searches) - 1, -1, -1):
            for rank, sid in enumerate(state.searches[index].source_ids):
                ranks.setdefault(sid, (-index, rank))
        return sorted(state.sources.values(), key=lambda s: (
            bool(focus and focus.specification.id not in s.question_ids),
            s.id not in required, ranks.get(s.id, (1, int(s.id[1:]))),
        ))

    @staticmethod
    def _defer(q: QuestionRecord, reason: str) -> None:
        q.status = "unresolved"
        q.phase = "done"
        q.unresolved_reason = reason

    def _ask(self, state: InquiryState, question: str, reason: str, qid: str | None, blocking: bool = True) -> bool:
        for c in state.clarifications:
            if c.question.casefold() == question.casefold() and c.question_id == qid:
                if c.answer is not None:
                    return False
                c.active = True
                c.blocking = blocking
                return True
        state.clarifications.append(UserClarification(
            id=f"C{len(state.clarifications) + 1}", question=question, reason=reason,
            question_id=qid, blocking=blocking,
        ))
        self._event(state, "clarification", "A material clarification is needed", qid)
        return True

    def _invalidate(self, state: InquiryState, qid: str) -> None:
        affected = {qid}
        while True:
            expanded = affected | {
                key for key, q in state.questions.items()
                if set(q.specification.depends_on) & affected
            }
            if expanded == affected:
                break
            affected = expanded
        for key in affected:
            q = state.questions[key]
            state.archived_questions.append(q.model_copy(deep=True))
            state.questions[key] = QuestionRecord(
                specification=q.specification, frame_version=q.frame_version,
                instructions="Reconsider this question using the user's new clarification.",
            )
        state.synthesis = None
        state.synthesis_challenge = None
        state.synthesis_reflection = None
        state.synthesis_rounds = 0
        state.answer = None
        state.phase = "work"

    def _retrieve(self, state: InquiryState, q: QuestionRecord, call) -> None:
        qid = q.specification.id
        args = call.arguments.model_dump()
        if call.tool == "web_search":
            cached = self._cached_search(state, args)
            if cached:
                for sid in cached.source_ids:
                    if qid not in state.sources[sid].question_ids:
                        state.sources[sid].question_ids.append(qid)
                self._event(state, "cache_hit", "Reused an earlier search", qid)
                return
            if state.usage.searches >= state.limits.max_searches:
                self._event(state, "research_limit", "Search budget reached; retain the evidence gap", qid)
                return
            self._begin(state, call.tool, qid)
            state.usage.searches += 1
            self.store.save(state)
            result = self.tools.call(call.tool, args)
            record = SearchRecord(query=args["query"], num=args["num"], question_id=qid, purpose=call.purpose)
            if result.status != "success":
                record.error = result.error or "Search failed"
            else:
                for hit in (result.data or {}).get("results", []):
                    src = self._source(state, hit["url"], hit.get("title", ""), hit.get("snippet", ""), qid)
                    if src.id not in record.source_ids:
                        record.source_ids.append(src.id)
            state.searches.append(record)
            self._complete(state, call.tool, qid, "Search failed" if record.error else "Search results recorded")
        else:
            src = self._find_source(state, args["url"])
            if src is None:
                raise InquiryError("read_source URL must come from search results or user-supplied context")
            if src.status == "read":
                if qid not in src.question_ids:
                    src.question_ids.append(qid)
                self._event(state, "cache_hit", "Reused a previously read source", qid)
                return
            if state.usage.reads >= state.limits.max_reads:
                self._event(state, "research_limit", "Source-reading budget reached; retain the evidence gap", qid)
                return
            self._begin(state, call.tool, qid)
            state.usage.reads += 1
            self.store.save(state)
            result = self.tools.call(call.tool, args)
            if result.status != "success":
                src.status = "failed"
                src.error = result.error or "Source reading failed"
            else:
                doc = result.data or {}
                text = str(doc.get("text", "")).strip()
                if not text:
                    src.status = "failed"
                    src.error = "Source returned no readable text"
                else:
                    old_url = src.url
                    src.url = doc.get("url", src.url)
                    if old_url != src.url and old_url not in src.aliases:
                        src.aliases.append(old_url)
                    src.title = doc.get("title") or src.title
                    src.status = "read"
                    src.error = None
                    src.content_type = doc.get("content_type", "")
                    src.retrieved_at = doc.get("retrieved_at")
                    src.truncated = bool(doc.get("truncated", False))
                    src.passages = self._split_passages(src.id, text)
                    if qid not in src.question_ids:
                        src.question_ids.append(qid)
            self._complete(state, call.tool, qid, "Source read" if src.status == "read" else "Source could not be read")

    @staticmethod
    def _split_passages(source_id: str, text: str) -> list[Passage]:
        chunks = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            while paragraph:
                split = min(len(paragraph), 1400)
                if split < len(paragraph):
                    boundary = paragraph.rfind(" ", 700, split)
                    if boundary != -1:
                        split = boundary
                chunks.append(paragraph[:split])
                paragraph = paragraph[split:].strip()
        return [Passage(id=f"{source_id}:P{i + 1}", text=t) for i, t in enumerate(chunks)]

    @staticmethod
    def _find_source(state: InquiryState, url: str) -> SourceRecord | None:
        clean = urldefrag(url).url
        return next((s for s in state.sources.values() if clean in [s.url, *s.aliases]), None)

    def _source(self, state: InquiryState, url: str, title: str, snippet: str, qid: str) -> SourceRecord:
        url = urldefrag(url).url
        src = self._find_source(state, url)
        if src:
            if qid not in src.question_ids:
                src.question_ids.append(qid)
            return src
        src = SourceRecord(id=f"S{len(state.sources) + 1}", url=url, title=title, snippet=snippet, question_ids=[qid])
        state.sources[src.id] = src
        return src

    def _invoke(self, state, primitive, response_model, context, instructions="", qid=None):
        repair = state.output_repair
        if repair is not None and (
            repair.primitive, repair.question_id, repair.response_model
        ) != (primitive, qid, response_model.__name__):
            state.output_repair = repair = None
        while True:
            if repair is not None and repair.failures > state.limits.max_output_repairs:
                # Also covers an interruption immediately after persisting the
                # last rejected response but before raising the bounded failure.
                raise StructuredOutputError(
                    f"{repair.message} Correction allowance exhausted ({state.limits.max_output_repairs} repairs).",
                    issues=repair.issues,
                )
            self._begin(state, primitive, qid)
            attempt_context = context
            if isinstance(context.get("remaining_budget"), dict):
                attempt_context = {**context, "remaining_budget": {
                    **context["remaining_budget"],
                    "steps": state.limits.max_steps - state.usage.steps,
                }}
            guidance = [state.instructions, instructions]
            if repair is not None:
                # Include a unique scheduled step so an exact response cache
                # cannot replay a previously rejected corrective response.
                guidance.append(
                    f"OUTPUT CORRECTION at scheduled step {state.usage.steps}; "
                    f"prior rejected outputs in this batch: {repair.failures}. "
                    "The previous response was rejected before any of its proposed tool calls or findings were accepted. "
                    "Regenerate a response for the same task and authoritative response schema. "
                    "Correct the listed format/field errors; do not invent missing evidence, tool results, or URLs. "
                    "No rejected response is being supplied as trusted content. "
                    "Validation feedback: " + json.dumps(repair.issues, ensure_ascii=True)
                )
                self._event(state, "schema_repair", f"Requesting a corrected {primitive} response", qid,
                            {"primitive": primitive, "prior_rejections": repair.failures})
                self.store.save(state)
            try:
                try:
                    output = self.reasoner.invoke(
                        primitive, context=attempt_context,
                        instructions="\n".join(filter(None, guidance)),
                        response_model=response_model,
                    )
                    if not isinstance(output, response_model):
                        output = response_model.model_validate(output)
                except ValidationError as exc:
                    # Protocol-compatible adapters can expose Pydantic errors
                    # directly. Use the same safe diagnostics as CoreReasoner.
                    raise schema_error(primitive, response_model, exc) from None
            except StructuredOutputError as exc:
                failures = repair.failures + 1 if repair is not None else 1
                state.in_flight = None
                repair = OutputRepairState(
                    primitive=primitive, question_id=qid, response_model=response_model.__name__,
                    failures=failures, message=str(exc), issues=exc.issues,
                )
                state.output_repair = repair
                self._event(state, "output_rejected", str(exc), qid, {
                    "primitive": primitive, "failures": failures, "issues": exc.issues,
                })
                self.store.save(state)
                if failures > state.limits.max_output_repairs:
                    raise StructuredOutputError(
                        f"{exc} Correction allowance exhausted ({state.limits.max_output_repairs} repairs).",
                        issues=exc.issues,
                    ) from None
                continue
            state.output_repair = None
            self._complete(state, primitive, qid, f"{primitive.capitalize()} completed")
            return output

    def _begin(self, state: InquiryState, operation: str, qid: str | None) -> None:
        if state.usage.steps >= state.limits.max_steps:
            raise _BudgetExhausted()
        state.usage.steps += 1
        state.in_flight = {"operation": operation, "question_id": qid or ""}
        self._event(state, "step_started", operation, qid)
        self.store.save(state)

    def _complete(self, state: InquiryState, operation: str, qid: str | None, summary: str) -> None:
        state.in_flight = None
        self._event(state, operation, summary, qid)

    def _event(self, state, kind, summary, qid=None, details=None):
        event = InquiryEvent(sequence=len(state.events) + 1, step=state.usage.steps, kind=kind,
                             question_id=qid, summary=summary, details=details or {})
        state.events.append(event)
        if self.on_event:
            self.on_event(event)

    def _checkpoint(self, state: InquiryState) -> None:
        from cognitive_core.inquiry.report import render_report

        self.store.save(state)
        self.store.write_report(render_report(state))

    @staticmethod
    def _validate_acceptance(reflection: InquiryReflection, challenge: ChallengeOutput | None) -> None:
        if reflection.next_action == "accept" and challenge and not challenge.survives:
            if not reflection.objection_response.strip():
                raise InquiryError("Accepting a challenged conclusion requires an explicit response to the objection")

    @staticmethod
    def _validate_claims(output: BaseModel, context: dict) -> None:
        available = {p["id"]: p["text"] for p in context.get("evidence", [])}
        for claim in output.claims:
            for ref in claim.evidence:
                if ref.passage_id not in available:
                    raise InquiryError("A claim cited a passage that was not read and supplied to this step")
                quote = " ".join(ref.quote.split())
                passage = " ".join(available[ref.passage_id].split())
                if quote not in passage:
                    raise InquiryError("A claim's quoted evidence does not occur in the cited passage")

    @staticmethod
    def _validate_answer(answer: str, state: InquiryState, context: dict) -> None:
        known = {p["id"] for p in context.get("evidence", [])}
        cited = set(re.findall(r"\[(S[0-9]+(?::P[0-9]+)?)\]", answer, re.IGNORECASE))
        if cited - known:
            raise InquiryError("The answer contains unknown or unavailable passage citations")
        if state.synthesis and any(c.evidence for c in state.synthesis.claims) and not cited:
            raise InquiryError("The answer omitted citations for its source-supported synthesis")
        known_urls = {s.url for s in state.sources.values() if s.status == "read"}
        if any(url.rstrip(".,;") not in known_urls for url in URL.findall(answer)):
            raise InquiryError("The answer contains a source URL that was not read")

    def _context(self, state: InquiryState, focus: QuestionRecord | None = None) -> dict[str, Any]:
        def summary(q):
            finding = q.findings[-1] if q.findings else None
            return {
                "id": q.specification.id, "question": q.specification.question,
                "depends_on": q.specification.depends_on, "status": q.status,
                "finding": finding.finding[:2500] if finding else None,
                "claims": [{"text": c.text[:1000], "basis": c.basis,
                            "passage_ids": [r.passage_id for r in c.evidence],
                            "qualification": c.qualification[:500]}
                           for c in finding.claims[:10]] if finding else [],
                "assumptions": finding.assumptions[:8] if finding else [],
                "unresolved_issues": finding.unresolved_issues[:8] if finding else [],
                "unresolved_reason": q.unresolved_reason,
            }

        required = set()
        for q in state.questions.values():
            if q.findings:
                required.update(r.passage_id for c in q.findings[-1].claims for r in c.evidence)
        if state.synthesis:
            required.update(r.passage_id for c in state.synthesis.claims for r in c.evidence)
        required_sources = {pid.split(":")[0] for pid in required}
        catalog = self._ordered_sources(state, focus, required_sources)
        context = {
            "original_question": state.original_question,
            "purpose": state.purpose,
            "user_context": state.user_context[:state.limits.max_context_chars // 4],
            "user_context_truncated": len(state.user_context) > state.limits.max_context_chars // 4,
            "frame": state.frame.model_dump(mode="json") if state.frame else None,
            "branches": [summary(q) for q in state.questions.values()],
            "clarifications": [c.model_dump() for c in state.clarifications if c.active or c.answer is not None],
            "focus": focus.specification.model_dump() if focus else None,
            "current_finding": summary(focus) if focus else None,
            "branch_progress": {
                "completed_rounds": focus.rounds,
                "round_limit": state.limits.max_rounds_per_question,
                "research_plans_this_round": focus.research_plans,
                "research_plan_limit": state.limits.max_research_plans_per_round,
                "read_sources": sum(s.status == "read" and focus.specification.id in s.question_ids for s in state.sources.values()),
                "unread_sources": sum(s.status == "discovered" and focus.specification.id in s.question_ids for s in state.sources.values()),
                "retrieval_feedback": focus.research_feedback,
            } if focus else None,
            "recent_findings": [{
                "round": i + 1, "finding": f.finding[:2000],
                "claims": [{"text": c.text[:800], "basis": c.basis,
                            "passage_ids": [r.passage_id for r in c.evidence]} for c in f.claims[:8]],
                "unresolved_issues": f.unresolved_issues[:8],
            } for i, f in enumerate(focus.findings) if i >= len(focus.findings) - 2] if focus else [],
            "recent_reflections": [{
                "next_action": r.next_action, "what_changed": r.what_changed,
                "next_question": r.next_question, "open_questions": r.open_questions[:8],
                "objection_response": r.objection_response,
            } for r in focus.reflections[-2:]] if focus else [],
            "recent_retrieval_plans": [e.details for e in state.events
                                       if e.kind == "retrieval_plan" and focus and e.question_id == focus.specification.id][-2:],
            "current_challenge": (
                focus.challenges[-1].model_dump() if focus and focus.challenges else
                state.synthesis_challenge.model_dump() if not focus and state.synthesis_challenge else None
            ),
            "synthesis": state.synthesis.model_dump() if state.synthesis and not focus else None,
            "synthesis_reflection": state.synthesis_reflection.model_dump() if state.synthesis_reflection and not focus else None,
            "search_history": [s.model_dump() for s in reversed(state.searches[-12:])],
            "source_catalog": [{"id": s.id, "url": s.url, "title": s.title,
                                "snippet": s.snippet[:700], "status": s.status,
                                "error": s.error, "truncated": s.truncated,
                                "retrieved_at": s.retrieved_at}
                               for s in catalog[:12]],
            "source_catalog_omitted": max(0, len(catalog) - 12),
            "remaining_budget": {
                "steps": state.limits.max_steps - state.usage.steps,
                "searches": state.limits.max_searches - state.usage.searches,
                "reads": state.limits.max_reads - state.usage.reads,
            },
            "instructions": state.instructions + ("\n" + focus.instructions if focus else ""),
            "evidence": [],
        }
        # Bound repetitive metadata, while preserving evidence quotes verbatim below.
        def compact(value, limit, key=""):
            if isinstance(value, str):
                if key in {"url", "id", "passage_id"}:
                    return value
                return value[:limit] + (" [truncated]" if len(value) > limit else "")
            if isinstance(value, list):
                return [compact(v, limit) for v in value[:12]]
            if isinstance(value, dict):
                return {k: compact(v, limit, k) for k, v in value.items()}
            return value

        metadata_limit = state.limits.max_context_chars // 2
        for limit in (1800, 700, 300, 120):
            if len(json.dumps(context)) <= metadata_limit:
                break
            context = compact(context, limit)
            context["metadata_truncated"] = True
        # Reserve space for actual passages even with large histories. Remove
        # redundant summaries before sacrificing the active question or sources.
        for key in ("current_finding", "synthesis_reflection", "search_history", "recent_retrieval_plans",
                    "recent_findings", "recent_reflections", "user_context"):
            if len(json.dumps(context)) <= metadata_limit:
                break
            context.pop(key, None)
            context["metadata_truncated"] = True
        while len(json.dumps(context)) > metadata_limit and len(context["source_catalog"]) > 1:
            context["source_catalog"].pop()
            context["source_catalog_omitted"] += 1
        if len(json.dumps(context)) > metadata_limit:
            # Extremely small context allowances still retain every branch's
            # status. Detailed state remains intact in the checkpoint.
            context["branches"] = [
                {"id": b["id"], "status": b["status"], "finding": (b["finding"] or "")[:120]}
                for b in context["branches"]
            ]
            context["frame"] = {
                "working_question": state.frame.working_question[:200],
                "answer_criteria": [v[:120] for v in state.frame.answer_criteria[:4]],
            } if state.frame else None
            if state.synthesis:
                context["synthesis"] = {
                    "conclusion": state.synthesis.conclusion[:300],
                    "limitations": [v[:120] for v in state.synthesis.limitations[:4]],
                }
            context["metadata_truncated"] = True
        if len(json.dumps(context)) > metadata_limit:
            raise InquiryError("The context allowance cannot hold this inquiry's essential state; increase max_context_chars")
        query = focus.specification.question + " " + focus.instructions if focus else state.original_question
        terms = {t.casefold() for t in re.findall(r"\w{4,}", query)}
        candidates = []
        for source in state.sources.values():
            if source.status != "read":
                continue
            for passage in source.passages:
                score = len(terms & set(re.findall(r"\w{4,}", passage.text.casefold())))
                priority = (passage.id not in required,
                            bool(focus and focus.specification.id not in source.question_ids), -score)
                candidates.append((priority, source, passage))
        candidates.sort(key=lambda item: item[0])
        remaining = max(0, state.limits.max_context_chars - len(json.dumps(context)) - 120)
        for _, source, passage in candidates:
            item = {"id": passage.id, "source_id": source.id, "title": source.title,
                    "url": source.url, "text": passage.text, "truncated_source": source.truncated,
                    "retrieved_at": source.retrieved_at}
            length = len(json.dumps(item))
            if length <= remaining:
                context["evidence"].append(item)
                remaining -= length + 2
        context["evidence_passages_omitted"] = len(candidates) - len(context["evidence"])
        return context
