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
    # ── Layer 2 judgment fields ──
    reasoning_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How sound is the classification reasoning: 0.0 = circular or unsupported, 1.0 = tightly evidence-grounded. Report honestly — used by governance."
    )
    outcome_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How clearly the evidence supports this category over alternatives: 0.0 = genuinely ambiguous, 1.0 = unambiguous."
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
    evidence_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Short labels for notable evidence patterns found during investigation. "
            "Examples: foreign_ip, unknown_device, velocity_spike, habitat_overlap. "
            "Used by delegation policies to route to specialist handlers."
        )
    )
    # ── Layer 2 judgment fields ──
    reasoning_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How sound is the investigative reasoning: 0.0 = speculative or circular, 1.0 = rigorous hypothesis testing grounded in evidence."
    )
    outcome_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How clearly the evidence supports the finding: 0.0 = evidence is ambiguous or incomplete, 1.0 = finding is well-supported."
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
    # ── Layer 2 judgment fields ──
    reasoning_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How sound is the verification reasoning: 0.0 = rules not properly applied, 1.0 = each rule independently and rigorously checked."
    )
    outcome_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How certain is the conformance determination: 0.0 = rule applicability genuinely ambiguous, 1.0 = clear conformance or clear violation."
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
    # ── Layer 2 judgment fields ──
    reasoning_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How well the artifact is grounded in the accumulated evidence and prior step outputs: 0.0 = generic or ungrounded, 1.0 = tightly connected to specific findings."
    )
    outcome_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How confident you are the artifact meets all requirements and constraints: 0.0 = significant gaps or uncertainties, 1.0 = all requirements demonstrably met."
    )


# ---------------------------------------------------------------------------
# Act — removed from primitive registry (see registry/artifacts.py)
#
# ActOutput, ActionExecution, AuthorizationCheck are downstream artifact
# types, not primitive outputs. Import them from cognitive_core.primitives.artifacts.
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
    # ── Layer 2 judgment fields ──
    reasoning_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How rigorous is the adversarial analysis: 0.0 = superficial or biased toward passing, 1.0 = thorough and genuinely adversarial."
    )
    outcome_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How certain is the survives determination: 0.0 = genuinely borderline, 1.0 = clearly passes or clearly fails."
    )


# ---------------------------------------------------------------------------
# Deliberate  (previously: Think)
# ---------------------------------------------------------------------------

class EvaluatedOption(BaseModel):
    """An option considered during deliberation."""
    action: str = Field(description="The action or option considered")
    viable: bool = Field(description="Whether this option is viable given the evidence")
    rationale: str = Field(default="", description="Why this option is or is not viable")


class DeliberateOutput(BaseOutput):
    """Output from a Deliberate primitive.

    Meta-cognitive synthesis over accumulated evidence, producing a warranted
    determination (ActionRecommendation) or formally suspending with a typed
    evidence request. Epistemic function: Dewey's warranted assertion, Simon's
    design phase, Stanovich & West's Type 2 reasoning.
    """
    situation_summary: str = Field(
        description="Synthesis of upstream step findings into factual state-of-the-world"
    )
    evaluation_criteria: str = Field(
        default="",
        description="Criteria extracted from domain framework used to evaluate options"
    )
    options_considered: list[EvaluatedOption] = Field(
        default_factory=list,
        description="Options enumerated and evaluated against criteria"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="The warranted action — what should happen next"
    )
    warrant: Optional[str] = Field(
        default=None,
        description="The logical connection between situation and recommended action (Toulmin warrant)"
    )
    confidence_basis: str = Field(
        default="",
        description="Which upstream step outputs drove the confidence score"
    )

    # ── Layer 2 judgment fields (LLM-reported epistemic self-assessment) ──
    # These are reported by the LLM and used alongside framework-computed
    # mechanical metrics in the WorkflowEpistemicRecord. Optional so existing
    # outputs that predate these fields continue to parse without error.
    reasoning_quality: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description=(
            "Self-assessed quality of the reasoning chain: 0.0 = circular or unsupported, "
            "1.0 = rigorous step-by-step logic tightly connected to evidence. "
            "Report honestly — this is used by the governance layer to decide whether "
            "human review is required. Do not inflate."
        )
    )
    outcome_certainty: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description=(
            "How clearly the available evidence supports the recommended_action: "
            "0.0 = evidence is ambiguous or contradictory, 1.0 = evidence unambiguously "
            "supports one outcome. Distinct from reasoning_quality — you can reason well "
            "about a genuinely ambiguous situation (low certainty, high quality)."
        )
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
# Govern  (previously: Act — replaced entirely)
# ---------------------------------------------------------------------------

class GovernanceTier(str, Enum):
    """Governance tier applied to a decision instance."""
    AUTO = "auto"
    SPOT_CHECK = "spot_check"
    GATE = "gate"
    HOLD = "hold"


class WorkOrder(BaseModel):
    """
    A typed work order issued by the Govern primitive.

    Describes what must happen next — human review, sub-workflow dispatch,
    or external system action — with full governance context attached.
    The coordinator routes the work order to its target without modifying it.
    """
    work_type: str = Field(
        description="'human_review' | 'sub_workflow' | 'external'"
    )
    target: str = Field(
        description="Queue name, workflow ID, or system endpoint to receive this order"
    )
    instructions: str = Field(
        description="What the receiver must do — specific, not generic"
    )
    inputs: dict = Field(
        default_factory=dict,
        description="Typed inputs required by the receiver to do their work"
    )
    governance_tier: GovernanceTier = Field(
        description="Governance tier that applies to this work order"
    )
    sla: Optional[str] = Field(
        default=None,
        description="Time constraint (e.g. '4h', '2 business days') if applicable"
    )
    versioned_contract: str = Field(
        default="",
        description="Schema version of the expected return payload"
    )


class GovernanceDecision(BaseModel):
    """A single governance decision recorded in the accountability chain."""
    step: str = Field(description="Step name at which this decision was made")
    tier: GovernanceTier = Field(description="Tier determined at this step")
    rationale: str = Field(description="Why this tier was determined")
    timestamp: str = Field(default="", description="ISO timestamp of the decision")


class AccountabilityChain(BaseModel):
    """
    Full record of governance decisions made during this decision instance.

    Append-only. Each Govern invocation adds an entry. Never modified after creation.
    """
    decision_instance_id: str = Field(
        description="Unique ID of the decision instance this chain belongs to"
    )
    governance_decisions: list[GovernanceDecision] = Field(
        default_factory=list,
        description="Ordered list of governance decisions made during execution"
    )
    final_tier: GovernanceTier = Field(
        description="The highest tier reached — governs final disposition"
    )
    audit_timestamp: str = Field(
        default="",
        description="ISO timestamp when this chain was finalized"
    )


class GovernOutput(BaseOutput):
    """
    Output from a Govern primitive.

    Meta-cognition about the decision process: delegation, escalation,
    accountability, and institutional embedding. Govern reasons about
    what the accumulated decision state requires next — it does not execute.

    Theoretical anchor: Flavell's metacognition; Simon's authority structures;
    Ostrom's institutional design; Ross & Jensen-Meckling's principal-agent theory.
    """
    tier_applied: GovernanceTier = Field(
        description="Governance tier determined for this decision context"
    )
    tier_rationale: str = Field(
        description="Why this tier is appropriate given the accumulated evidence and context"
    )
    disposition: str = Field(
        description="'proceed' | 'escalate' | 'suspend' | 'hold'"
    )
    work_order: Optional[WorkOrder] = Field(
        default=None,
        description="Typed work order if delegation or human review is required"
    )
    escalation_target: Optional[str] = Field(
        default=None,
        description="Human queue or sub-workflow ID if escalation is required"
    )
    resumption_condition: Optional[str] = Field(
        default=None,
        description="What must happen before this workflow can resume"
    )
    accountability_chain: Optional[AccountabilityChain] = Field(
        default=None,
        description="Full record of governance decisions made during this instance"
    )



# ---------------------------------------------------------------------------
# Reflect  — metacognitive primitive
#
# Reasons about the reasoning, not the case.
# Produces dynamic specs for downstream primitives.
# Second-order: sits between steps, not in the sequence.
# ---------------------------------------------------------------------------

class SensitivityAnalysis(BaseModel):
    """The single fact whose change would most affect the current trajectory."""
    load_bearing_fact: str = Field(description="The fact that matters most")
    current_value: str = Field(description="What it currently says")
    alternative_value: str = Field(description="What it would need to say to change things")
    what_would_change: str = Field(description="What conclusion or path would change")


class ReflectOutput(BaseOutput):
    """Output from a Reflect primitive.

    Metacognitive synthesis over accumulated reasoning state.
    Does not reason about the case — reasons about the reasoning about the case.
    Produces dynamic specs that shape what the next primitive is asked to do.

    Theoretical anchor: Flavell metacognition; Schon reflection-in-action;
    Klein recognition-primed decision model; Simon bounded rationality.
    """
    what_was_established: list[str] = Field(
        default_factory=list,
        description="Claims with genuine evidential support, traceable to specific steps"
    )
    what_was_assumed: list[str] = Field(
        default_factory=list,
        description="Implicit premises the reasoning took for granted without verifying"
    )
    what_changed: str = Field(
        description="How the most recent step updated prior beliefs"
    )
    sensitivity: Optional[SensitivityAnalysis] = Field(
        default=None,
        description="The single fact whose change would most affect the current trajectory"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions the reasoning has not asked but should have"
    )
    trajectory: str = Field(
        description="continue | revise | escalate"
    )
    revision_target: Optional[str] = Field(
        default=None,
        description="Which step to revise — null if continue or escalate"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Why the reasoning has reached an impasse — null if continue or revise"
    )
    # ── Dynamic spec fields — shape the next primitive ──
    next_question: Optional[str] = Field(
        default=None,
        description="The specific question the next step should answer"
    )
    domain_keys_relevant: list[str] = Field(
        default_factory=list,
        description="Keys from the domain index relevant to the next question"
    )
    established_facts_to_skip: list[str] = Field(
        default_factory=list,
        description="What the next step should explicitly not re-examine"
    )
    hypothesis: Optional[str] = Field(
        default=None,
        description="Testable proposition for investigate steps"
    )
    template_guidance: Optional[str] = Field(
        default=None,
        description="How the next step should structure its reasoning"
    )
    # ── Layer 2 judgment fields ──
    reasoning_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How rigorously the reasoning was examined vs just summarized"
    )
    outcome_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How confident in the trajectory assessment"
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
    "deliberate": DeliberateOutput,
    "govern": GovernOutput,
    "reflect": ReflectOutput,
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