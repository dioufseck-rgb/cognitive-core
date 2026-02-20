"""
Cognitive Core — Structured Exception Hierarchy (H-021)

Typed errors so the coordinator can distinguish between:
- Logic failures → escalate to HITL
- Network failures → retry
- Data failures → terminate with audit
- Governance failures → route appropriately

Each error carries: severity, retryable flag, escalation_required flag.
"""

from __future__ import annotations
from enum import Enum


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════
# Base
# ═══════════════════════════════════════════════════════════════

class CognitiveCoreError(Exception):
    """Base exception for all Cognitive Core errors."""
    severity: Severity = Severity.MEDIUM
    retryable: bool = False
    escalation_required: bool = False

    def __init__(self, message: str = "", **kwargs):
        self.detail = kwargs
        super().__init__(message)


# ═══════════════════════════════════════════════════════════════
# Governance Errors — route to HITL or compliance
# ═══════════════════════════════════════════════════════════════

class GovernanceError(CognitiveCoreError):
    """Governance-related failures."""
    severity = Severity.HIGH
    escalation_required = True


class EscalationRequired(GovernanceError):
    """Case requires human escalation based on policy."""
    pass


class TierInvariantViolation(GovernanceError):
    """Attempted to downgrade a governance tier."""
    severity = Severity.CRITICAL


class ReviewTimeout(GovernanceError):
    """HITL review exceeded SLA deadline."""
    retryable = True  # Can be reassigned


class IllegalStateTransition(GovernanceError):
    """Invalid HITL state transition attempted."""
    pass


# ═══════════════════════════════════════════════════════════════
# Execution Errors — step-level failures
# ═══════════════════════════════════════════════════════════════

class ExecutionError(CognitiveCoreError):
    """Step execution failures."""
    severity = Severity.MEDIUM


class StepTimeout(ExecutionError):
    """Step exceeded its execution deadline."""
    retryable = True

    def __init__(self, step_name: str, timeout_seconds: float, elapsed: float):
        self.step_name = step_name
        self.timeout_seconds = timeout_seconds
        self.elapsed = elapsed
        super().__init__(
            f"Step {step_name!r} timed out: {elapsed:.1f}s > {timeout_seconds}s",
            step_name=step_name,
        )


class SchemaValidationFailure(ExecutionError):
    """LLM output failed schema validation."""
    retryable = True  # May succeed on retry with better prompt

    def __init__(self, step_name: str, expected_schema: str = "", actual: str = ""):
        super().__init__(
            f"Schema validation failed at step {step_name!r}",
            step_name=step_name,
            expected_schema=expected_schema,
            actual=actual,
        )


class WriteBoundaryViolation(ExecutionError):
    """Non-Act primitive attempted to call a write tool."""
    severity = Severity.CRITICAL
    escalation_required = True

    def __init__(self, tool_name: str, primitive: str):
        super().__init__(
            f"Write tool {tool_name!r} called from {primitive!r} (not Act)",
            tool_name=tool_name,
            primitive=primitive,
        )


class CompensationFailure(ExecutionError):
    """Act compensation could not be completed."""
    severity = Severity.HIGH
    escalation_required = True


# ═══════════════════════════════════════════════════════════════
# Provider Errors — LLM provider failures
# ═══════════════════════════════════════════════════════════════

class ProviderError(CognitiveCoreError):
    """LLM provider failures."""
    retryable = True


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""
    severity = Severity.LOW

    def __init__(self, provider: str, retry_after: float = 0):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Rate limited by {provider} (retry after {retry_after}s)",
            provider=provider,
        )


class ProviderAuthFailure(ProviderError):
    """Provider authentication failed."""
    severity = Severity.HIGH
    retryable = False  # Auth failures won't fix on retry

    def __init__(self, provider: str):
        super().__init__(f"Authentication failed for {provider}", provider=provider)


class ProviderUnavailable(ProviderError):
    """Provider returned 5xx or is unreachable."""
    severity = Severity.MEDIUM

    def __init__(self, provider: str, status_code: int = 0):
        super().__init__(
            f"Provider {provider} unavailable (HTTP {status_code})",
            provider=provider, status_code=status_code,
        )


class AllProvidersFailed(ProviderError):
    """All configured providers failed after retry/fallback."""
    severity = Severity.HIGH
    retryable = False
    escalation_required = True


# ═══════════════════════════════════════════════════════════════
# Data Errors — input/output integrity failures
# ═══════════════════════════════════════════════════════════════

class DataError(CognitiveCoreError):
    """Data integrity failures."""
    severity = Severity.HIGH


class IntegrityChecksumMismatch(DataError):
    """Document hash doesn't match expected value."""
    escalation_required = True

    def __init__(self, source_name: str, expected: str, actual: str):
        super().__init__(
            f"Integrity mismatch for {source_name!r}: expected {expected[:16]}... got {actual[:16]}...",
            source_name=source_name,
        )


class PiiRedactionFailure(DataError):
    """PII redaction could not be completed safely."""
    severity = Severity.CRITICAL
    escalation_required = True


class ContextOverflowError(DataError):
    """Accumulated context exceeds model token limit."""
    retryable = True  # May succeed with pruning

    def __init__(self, estimated_tokens: int, limit: int):
        super().__init__(
            f"Context overflow: {estimated_tokens} tokens > {limit} limit",
            estimated_tokens=estimated_tokens, limit=limit,
        )


# ═══════════════════════════════════════════════════════════════
# Budget Errors
# ═══════════════════════════════════════════════════════════════

class BudgetError(CognitiveCoreError):
    """Cost budget exceeded."""
    severity = Severity.MEDIUM
    escalation_required = True


class BudgetExceededError(BudgetError):
    """Workflow cost exceeded configured budget cap."""
    def __init__(self, current: float, limit: float, workflow: str = ""):
        super().__init__(
            f"Budget exceeded: ${current:.2f} > ${limit:.2f} for {workflow}",
            current=current, limit=limit, workflow=workflow,
        )
