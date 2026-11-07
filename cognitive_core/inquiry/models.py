"""Typed inquiry state and the research profiles of existing primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cognitive_core.primitives.schemas import (
    ChallengeOutput,
    DeliberateOutput,
    FrameOutput,
    FrameQuestion,
    InvestigateOutput,
    ReflectOutput,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InquiryLimits(Record):
    max_steps: int = Field(default=80, ge=1, le=1000)
    max_searches: int = Field(default=12, ge=0, le=200)
    max_reads: int = Field(default=18, ge=0, le=300)
    max_rounds_per_question: int = Field(default=3, ge=1, le=10)
    max_reframes: int = Field(default=2, ge=0, le=10)
    max_research_plans_per_round: int = Field(default=4, ge=1, le=10)
    max_context_chars: int = Field(default=36000, ge=4000, le=200000)


class Usage(Record):
    steps: int = 0
    searches: int = 0
    reads: int = 0
    reframes: int = 0


class SearchArguments(Record):
    query: str = Field(min_length=1, max_length=500)
    num: int = Field(default=5, ge=1, le=10)


class ReadArguments(Record):
    url: str = Field(min_length=1, max_length=4000)


class SearchCall(Record):
    tool: Literal["web_search"]
    arguments: SearchArguments
    purpose: str = Field(min_length=1, max_length=1000)


class ReadCall(Record):
    tool: Literal["read_source"]
    arguments: ReadArguments
    purpose: str = Field(min_length=1, max_length=1000)


RetrievalCall = Annotated[Union[SearchCall, ReadCall], Field(discriminator="tool")]


class RetrievalPlan(Record):
    """A typed Retrieve tool plan; empty calls means evidence is sufficient to reason."""

    calls: list[RetrievalCall] = Field(default_factory=list, max_length=4)
    rationale: str = Field(min_length=1)


class EvidenceReference(Record):
    passage_id: str = Field(pattern=r"^S[1-9][0-9]*:P[1-9][0-9]*$")
    quote: str = Field(min_length=1, max_length=1500)


class Claim(Record):
    text: str = Field(min_length=1)
    basis: Literal["source", "inference", "assumption", "user"]
    evidence: list[EvidenceReference] = Field(default_factory=list)
    qualification: str = ""

    @model_validator(mode="after")
    def source_claim_requires_evidence(self) -> "Claim":
        if self.basis == "source" and not self.evidence:
            raise ValueError("Source claims require a reference to a read passage")
        return self


class FindingOutput(InvestigateOutput):
    """Investigate output with explicit claim/evidence links for research."""

    claims: list[Claim] = Field(min_length=1, max_length=24)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)


class ClarificationRequest(Record):
    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class InquiryReflection(ReflectOutput):
    """Reflect controls a branch without closing the entire inquiry."""

    next_action: Literal["accept", "revise", "research", "reframe", "ask_user", "defer"]
    resolution: Literal["answered", "conditional"] = "answered"
    objection_response: str = ""
    clarification: ClarificationRequest | None = None

    @model_validator(mode="after")
    def action_contract(self) -> "InquiryReflection":
        if self.trajectory not in {"continue", "revise", "escalate"}:
            raise ValueError("Invalid reflection trajectory")
        if self.next_action == "ask_user" and self.clarification is None:
            raise ValueError("ask_user requires a specific clarification")
        if self.next_action != "ask_user" and self.clarification is not None:
            raise ValueError("clarification is only used by ask_user")
        return self


class SynthesisOutput(DeliberateOutput):
    conclusion: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1, max_length=24)
    limitations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class Passage(Record):
    id: str
    text: str


class SourceRecord(Record):
    id: str
    url: str
    title: str = ""
    snippet: str = ""
    aliases: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    status: Literal["discovered", "read", "failed"] = "discovered"
    retrieved_at: str | None = None
    content_type: str = ""
    truncated: bool = False
    passages: list[Passage] = Field(default_factory=list)
    error: str | None = None


class SearchRecord(Record):
    query: str
    num: int
    question_id: str
    purpose: str
    source_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    timestamp: str = Field(default_factory=utc_now)


class QuestionRecord(Record):
    specification: FrameQuestion
    frame_version: int
    status: Literal["pending", "active", "answered", "conditional", "unresolved"] = "pending"
    phase: Literal["plan", "retrieve", "investigate", "challenge", "reflect", "done"] = "plan"
    rounds: int = 0
    research_plans: int = 0
    stalled_research_plans: int = 0
    research_feedback: str = ""
    instructions: str = ""
    findings: list[FindingOutput] = Field(default_factory=list)
    challenges: list[ChallengeOutput] = Field(default_factory=list)
    reflections: list[InquiryReflection] = Field(default_factory=list)
    pending_calls: list[RetrievalCall] = Field(default_factory=list)
    unresolved_reason: str = ""


class UserClarification(Record):
    id: str
    question: str
    reason: str
    question_id: str | None = None
    blocking: bool = True
    answer: str | None = None
    active: bool = True


class InquiryEvent(Record):
    sequence: int
    timestamp: str = Field(default_factory=utc_now)
    step: int
    kind: str
    question_id: str | None = None
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class InquiryState(Record):
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    original_question: str = Field(min_length=1, max_length=20000)
    purpose: str = ""
    user_context: str = Field(default="", max_length=200000)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: Literal[
        "ready", "running", "awaiting_user", "budget_exhausted", "completed", "failed"
    ] = "ready"
    phase: Literal[
        "frame", "work", "synthesize", "review_synthesis", "reflect_synthesis", "generate", "done"
    ] = "frame"
    limits: InquiryLimits = Field(default_factory=InquiryLimits)
    usage: Usage = Field(default_factory=Usage)
    frames: list[FrameOutput] = Field(default_factory=list)
    questions: dict[str, QuestionRecord] = Field(default_factory=dict)
    archived_questions: list[QuestionRecord] = Field(default_factory=list)
    sources: dict[str, SourceRecord] = Field(default_factory=dict)
    searches: list[SearchRecord] = Field(default_factory=list)
    clarifications: list[UserClarification] = Field(default_factory=list)
    events: list[InquiryEvent] = Field(default_factory=list)
    synthesis: SynthesisOutput | None = None
    synthesis_challenge: ChallengeOutput | None = None
    synthesis_reflection: InquiryReflection | None = None
    synthesis_rounds: int = 0
    answer: str | None = None
    instructions: str = ""
    stop_reason: str = ""
    error: str | None = None
    in_flight: dict[str, str] | None = None

    @property
    def frame(self) -> FrameOutput | None:
        return self.frames[-1] if self.frames else None

    @property
    def pending_clarifications(self) -> list[UserClarification]:
        return [c for c in self.clarifications if c.active and c.blocking and c.answer is None]
