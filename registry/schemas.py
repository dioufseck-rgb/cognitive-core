"""
Cognitive Core - Output Schemas

Structured output contracts for each cognitive primitive.
These define the interface between primitives in a composition chain.
All primitives share a common base; each adds primitive-specific fields.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Base output — shared by all primitives
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A piece of evidence used or needed by a primitive."""
    source: str = Field(description="Where this evidence came from or would come from")
    description: str = Field(description="What this evidence shows or would show")


class BaseOutput(BaseModel):
    """Common fields shared by all cognitive primitive outputs."""
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the output, 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning that led to this output"
    )
    evidence_used: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence that was available and used"
    )
    evidence_missing: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence that would improve confidence if available"
    )


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

class AlternativeCategory(BaseModel):
    """A category that was considered but not selected."""
    category: str = Field(description="The alternative category name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence for this alternative")
    reasoning: str = Field(description="Why this alternative was considered")


class ClassifyOutput(BaseOutput):
    """Output from a Classify primitive."""
    category: str = Field(description="The assigned category")
    alternative_categories: list[AlternativeCategory] = Field(
        default_factory=list,
        description="Other categories considered with their confidence scores"
    )


# ---------------------------------------------------------------------------
# Investigate
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class HypothesisTested(BaseModel):
    """A hypothesis that was formed and tested during investigation."""
    hypothesis: str = Field(description="The hypothesis statement")
    status: HypothesisStatus = Field(description="Whether the hypothesis was supported, rejected, or inconclusive")
    evidence_for: list[str] = Field(default_factory=list, description="Evidence supporting this hypothesis")
    evidence_against: list[str] = Field(default_factory=list, description="Evidence contradicting this hypothesis")
    reasoning: str = Field(description="Reasoning about this hypothesis")


class InvestigateOutput(BaseOutput):
    """Output from an Investigate primitive."""
    finding: str = Field(description="The primary finding or conclusion")
    hypotheses_tested: list[HypothesisTested] = Field(
        default_factory=list,
        description="Hypotheses that were formed and evaluated"
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Suggested next steps based on findings"
    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class Violation(BaseModel):
    """A specific conformance violation found during verification."""
    rule: str = Field(description="The rule or criterion that was violated")
    description: str = Field(description="What the violation is")
    severity: str = Field(description="How severe: critical, major, minor")
    location: str = Field(default="", description="Where in the input the violation occurs")


class VerifyOutput(BaseOutput):
    """Output from a Verify primitive."""
    conforms: bool = Field(description="Whether the input conforms to all criteria")
    violations: list[Violation] = Field(
        default_factory=list,
        description="List of violations found"
    )
    rules_checked: list[str] = Field(
        default_factory=list,
        description="All rules/criteria that were checked"
    )


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class ConstraintResult(BaseModel):
    """Result of checking a constraint on the generated artifact."""
    constraint: str = Field(description="The constraint that was checked")
    satisfied: bool = Field(description="Whether the constraint was satisfied")
    notes: str = Field(default="", description="Any notes about how the constraint was addressed")


class GenerateOutput(BaseOutput):
    """Output from a Generate primitive."""
    artifact: str = Field(description="The generated artifact (text, document, etc.)")
    format: str = Field(default="text", description="Format of the artifact")
    constraints_checked: list[ConstraintResult] = Field(
        default_factory=list,
        description="Constraints that were checked during generation"
    )


# ---------------------------------------------------------------------------
# Challenge
# ---------------------------------------------------------------------------

class Vulnerability(BaseModel):
    """A vulnerability or weakness found during adversarial challenge."""
    description: str = Field(description="What the vulnerability is")
    severity: str = Field(description="How severe: critical, high, medium, low")
    attack_vector: str = Field(description="How this vulnerability could be exploited")
    recommendation: str = Field(description="How to address this vulnerability")


class ChallengeOutput(BaseOutput):
    """Output from a Challenge primitive."""
    survives: bool = Field(description="Whether the input survives the challenge overall")
    vulnerabilities: list[Vulnerability] = Field(
        default_factory=list,
        description="Vulnerabilities found during adversarial analysis"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Aspects that are robust against challenge"
    )
    overall_assessment: str = Field(
        description="Summary assessment of resilience"
    )


# ---------------------------------------------------------------------------
# Think
# ---------------------------------------------------------------------------

class ThinkOutput(BaseOutput):
    """Output from a Think primitive. Freeform reasoning with structured takeaways."""
    thought: str = Field(description="Freeform reasoning — the full chain of thought")
    conclusions: list[str] = Field(
        default_factory=list,
        description="Key takeaways distilled from the reasoning"
    )
    decision: Optional[str] = Field(
        default=None,
        description="Recommended course of action, if one emerges from the reasoning"
    )


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

class SourceResult(BaseModel):
    """Result from a single data source retrieval."""
    source: str = Field(description="Name of the data source")
    status: str = Field(description="success | failed | skipped")
    data: Optional[dict] = Field(default=None, description="Retrieved data if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    latency_ms: Optional[float] = Field(default=None, description="Retrieval latency in milliseconds")
    freshness: Optional[str] = Field(default=None, description="Timestamp or staleness indicator")


class RetrieveOutput(BaseOutput):
    """Output from a Retrieve primitive."""
    data: dict = Field(
        default_factory=dict,
        description="Assembled data keyed by source name"
    )
    sources_queried: list[SourceResult] = Field(
        default_factory=list,
        description="Status of each source that was queried"
    )
    sources_skipped: list[str] = Field(
        default_factory=list,
        description="Sources that were available but not needed"
    )
    retrieval_plan: str = Field(
        default="",
        description="The plan the agent used to decide what to retrieve (agentic mode only)"
    )


# ---------------------------------------------------------------------------
# Schema registry — lookup by primitive name
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, type[BaseOutput]] = {
    "classify": ClassifyOutput,
    "investigate": InvestigateOutput,
    "verify": VerifyOutput,
    "generate": GenerateOutput,
    "challenge": ChallengeOutput,
    "retrieve": RetrieveOutput,
    "think": ThinkOutput,
}


def get_schema(primitive_name: str) -> type[BaseOutput]:
    """Get the output schema class for a primitive."""
    name = primitive_name.lower()
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown primitive: {name}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[name]


def schema_to_json_spec(primitive_name: str) -> str:
    """Get a JSON schema description string for use in prompts."""
    schema_cls = get_schema(primitive_name)
    schema = schema_cls.model_json_schema()
    return _schema_to_readable(schema, schema.get("$defs", {}))


def _schema_to_readable(schema: dict, defs: dict, indent: int = 0) -> str:
    """Convert JSON schema to a human-readable spec for prompts."""
    lines = []
    prefix = "  " * indent

    if schema.get("type") == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for name, prop in props.items():
            req = "(required)" if name in required else "(optional)"
            desc = prop.get("description", "")

            # Handle $ref
            if "$ref" in prop:
                ref_name = prop["$ref"].split("/")[-1]
                ref_schema = defs.get(ref_name, {})
                lines.append(f"{prefix}{name} {req}: {desc}")
                lines.append(_schema_to_readable(ref_schema, defs, indent + 1))
            elif prop.get("type") == "array":
                items = prop.get("items", {})
                lines.append(f"{prefix}{name} {req}: array - {desc}")
                if "$ref" in items:
                    ref_name = items["$ref"].split("/")[-1]
                    ref_schema = defs.get(ref_name, {})
                    lines.append(f"{prefix}  Each item:")
                    lines.append(_schema_to_readable(ref_schema, defs, indent + 2))
            else:
                type_str = prop.get("type", "any")
                lines.append(f"{prefix}{name} ({type_str}) {req}: {desc}")

    return "\n".join(lines)
