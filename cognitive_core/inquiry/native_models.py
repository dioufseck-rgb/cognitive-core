"""Domain contracts for an inquiry executed by native Cognitive Core nodes.

These types describe data only.  They do not schedule work, call a model, or
own persistence.  The native Coordinator owns the serialized Study.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cognitive_core.inquiry.models import Claim, FindingOutput, SynthesisOutput
from cognitive_core.primitives.schemas import (
    ChallengeOutput,
    FrameClarification,
    FrameOutput,
    GenerateOutput,
    ReflectOutput,
    VerifyOutput,
)


class StudyValidationError(ValueError):
    """A candidate cannot be applied to the accepted native study."""


class StudyTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["study", "question", "report"]
    question_id: str | None = None
    expected_study_revision: int | None = Field(default=None, ge=0)
    frame_version: int | None = Field(default=None, ge=0)
    report_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def question_identity(self):
        if self.kind == "question" and not self.question_id:
            raise ValueError("A question target requires question_id")
        if self.kind != "question" and self.question_id is not None:
            raise ValueError("Only a question target may contain question_id")
        return self


class NativeFrameOutput(FrameOutput):
    alternatives: list[str] = Field(
        default_factory=list,
        description="Plausible competing answers; these are hypotheses, not findings.",
    )


class NativeClaim(Claim):
    """A cited claim also states the model's assessment of the evidence link."""

    support_assessment: str = Field(
        default="",
        description=(
            "Explain what the cited passages establish for this specific claim, "
            "including relevant task, comparator, outcome, and transfer limits. "
            "This is a model assessment, not a deterministic entailment guarantee."
        ),
    )
    premises: list[str] = Field(
        default_factory=list,
        description="Explicit premises for an inference; do not invent experiments.",
    )

    @model_validator(mode="after")
    def source_assessment(self):
        if self.basis == "source" and not self.support_assessment.strip():
            raise ValueError("A source claim requires an explicit support_assessment")
        return self


class NativeFindingOutput(FindingOutput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    finding: str = Field(min_length=1)
    claims: list[NativeClaim] = Field(min_length=1, max_length=24)
    question_disposition: Literal["active", "answered", "conditional", "unresolved"] = "active"
    material_change: str = Field(
        default="",
        description="The claim, evidence, premise, or scope that actually changed.",
    )


class SupportAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str = Field(min_length=1)
    passage_ids: list[str] = Field(default_factory=list)
    assessment: Literal["supports", "contradicts", "context_only", "insufficient"]
    reasoning: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class NativeSupportReviewOutput(VerifyOutput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    support_assessments: list[SupportAssessment] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_consistency(self):
        if self.conforms and self.violations:
            raise ValueError("A conforming support review cannot contain violations")
        return self


class NativeChallengeOutput(ChallengeOutput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    overall_assessment: str = Field(min_length=1)


class NativeReflectionOutput(ReflectOutput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    trajectory: Literal["continue", "revise", "escalate"]
    what_changed: str = Field(min_length=1)
    clarification: FrameClarification | None = None


class NativeSynthesisOutput(SynthesisOutput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claims: list[NativeClaim] = Field(min_length=1, max_length=24)
    research_outcome: Literal[
        "answered", "conditional", "evidence_limited", "budget_exhausted"
    ] = "conditional"
    criteria_coverage: dict[str, str] = Field(
        default_factory=dict,
        description="Each answer criterion and its supported, conditional, or unresolved disposition.",
    )


class NativeReportOutput(GenerateOutput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact: str = Field(min_length=1)
    format: Literal["markdown"] = "markdown"


class NativeStudy(BaseModel):
    """Serializable working artifact; the native runtime owns its lifecycle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    revision: int = 0
    original_question: str = Field(min_length=1)
    purpose: str = ""
    context: str = ""
    frame: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = Field(default_factory=list)
    questions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    archived_questions: list[dict[str, Any]] = Field(default_factory=list)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_cache: dict[str, Any] = Field(default_factory=dict)
    searches: list[dict[str, Any]] = Field(default_factory=list)
    retrievals: list[dict[str, Any]] = Field(default_factory=list)
    claims: dict[str, dict[str, Any]] = Field(default_factory=dict)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    reflections: list[dict[str, Any]] = Field(default_factory=list)
    support_reviews: list[dict[str, Any]] = Field(default_factory=list)
    clarifications: dict[str, dict[str, Any]] = Field(default_factory=dict)
    user_amendments: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    reports: list[dict[str, Any]] = Field(default_factory=list)
    final_review: dict[str, Any] | None = None
    governance: dict[str, Any] | None = None
    outcome: str = "open"
    outcome_reason: str = ""
    limits: dict[str, int] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    applied_invocations: dict[str, dict[str, Any]] = Field(default_factory=dict)
