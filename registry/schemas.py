"""
Cognitive Core - Output Schemas

Structured output contracts for each cognitive primitive.
These define the interface between primitives in a composition chain.
All primitives share a common base; each adds primitive-specific fields.
"""

try:
    from pydantic import BaseModel, Field, model_validator
except ImportError:
    # Minimal shim for pydantic-free environments.
    # Schemas won't validate, but structural code (imports, primitives registry)
    # will work for simulated execution.
    class Field:  # type: ignore[no-redef]
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return self

    def model_validator(**kwargs):  # type: ignore[no-redef]
        def decorator(fn): return fn
        return decorator

    class _BaseModelMeta(type):
        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)
            return cls

    class BaseModel(metaclass=_BaseModelMeta):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        def dict(self):
            return self.model_dump()
from typing import Any, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Base output — shared by all primitives
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A piece of evidence used or needed by a primitive."""
    source: str = Field(description="Where this evidence came from or would come from")
    description: str = Field(description="What this evidence shows or would show")


class ResourceRequest(BaseModel):
    """
    A request from an agent to the coordinator for a resource
    it needs to proceed.

    This is the general mechanism for demand-driven delegation.
    An agent that can't do its job without some upstream decision,
    analysis, data, authorization, or specialist input expresses
    that need as a ResourceRequest. The coordinator matches the
    need to a capability provider and dispatches it.

    The agent doesn't know WHO will fulfill it — only WHAT it needs,
    described via a contract the coordinator can match.
    """
    need: str = Field(
        description=(
            "What is needed, expressed as a capability name. "
            "Examples: 'eligibility_constraints', 'intake_packet', "
            "'fraud_clearance', 'path_recommendation', 'credit_review'"
        )
    )
    contract: str = Field(
        default="",
        description=(
            "Contract name for the expected response format. "
            "If empty, coordinator infers from capability registry."
        )
    )
    reason: str = Field(
        description="Why this resource is needed to proceed"
    )
    blocking: bool = Field(
        default=True,
        description=(
            "If true, agent cannot proceed without this resource. "
            "Coordinator will suspend the agent until fulfilled. "
            "If false, dispatched as fire-and-forget."
        )
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Relevant state to pass to whoever fulfills this request. "
            "Should contain everything the provider needs to do its work."
        )
    )
    urgency: str = Field(
        default="routine",
        description="Priority: 'routine', 'elevated', or 'critical'"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "List of other need names that must be fulfilled before "
            "this one can be dispatched. Empty means no dependencies — "
            "can run in parallel with other requests. Example: if "
            "'reserve_impact' depends on 'settlement_calculation', "
            "set depends_on=['settlement_calculation']."
        )
    )


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
    resource_requests: list[ResourceRequest] = Field(
        default_factory=list,
        description=(
            "Resources this agent needs from the coordinator to proceed. "
            "Each request describes WHAT is needed (not WHO provides it). "
            "Blocking requests will suspend this workflow until fulfilled. "
            "Only produce these when the required resource is genuinely "
            "absent from the available input and context."
        )
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
    artifact: Any = Field(description="The generated artifact (JSON object, text, etc.)")
    format: str = Field(default="text", description="Format of the artifact")
    constraints_checked: list[ConstraintResult] = Field(
        default_factory=list,
        description="Constraints that were checked during generation"
    )


# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------

class ActionExecution(BaseModel):
    """Record of a single action executed or simulated."""
    action: str = Field(description="The action that was attempted")
    target_system: str = Field(description="The system or service the action targets")
    status: str = Field(description="executed | simulated | blocked | failed")
    confirmation_id: Optional[str] = Field(default=None, description="Confirmation or transaction ID from the target system")
    response_data: Optional[dict] = Field(default=None, description="Response payload from the target system")
    error: Optional[str] = Field(default=None, description="Error message if the action failed")
    latency_ms: Optional[float] = Field(default=None, description="Execution latency in milliseconds")
    reversible: bool = Field(default=True, description="Whether this action can be rolled back")
    rollback_handle: Optional[str] = Field(default=None, description="Identifier to reverse this action if needed")


class AuthorizationCheck(BaseModel):
    """Record of an authorization check performed before action execution."""
    check: str = Field(description="What was checked (e.g., 'supervisor_approval', 'amount_threshold')")
    result: str = Field(description="passed | failed | waived | not_applicable")
    reason: str = Field(default="", description="Why this check passed or failed")


class ActOutput(BaseOutput):
    """Output from an Act primitive. Records what was done (or what would be done)."""
    mode: str = Field(description="execution | dry_run | approval_required")
    actions_taken: list[ActionExecution] = Field(
        default_factory=list,
        description="Actions that were executed, simulated, or blocked"
    )
    authorization_checks: list[AuthorizationCheck] = Field(
        default_factory=list,
        description="Authorization checks performed before execution"
    )
    side_effects: list[str] = Field(
        default_factory=list,
        description="Known side effects of the actions taken"
    )
    requires_human_approval: bool = Field(
        default=False,
        description="Whether any action requires human approval before proceeding"
    )
    approval_brief: Optional[str] = Field(
        default=None,
        description="Summary for the human approver if approval is required"
    )




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
    # Domain-specific fields that prompts may request
    risk_score: Optional[int] = Field(
        default=None,
        description="Numeric risk score if the instruction asks for one"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Named recommendation if the instruction asks for one"
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
    "act": ActOutput,
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
