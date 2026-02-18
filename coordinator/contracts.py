"""
Cognitive Core — Typed Contracts

TypedDict definitions for every dict that crosses a module boundary
in the coordinator. These serve three purposes:

  1. EDITOR: Type checkers (mypy/pyright) catch field mismatches at edit time
  2. RUNTIME: validate() functions catch mismatches at startup or test time
  3. DOCUMENTATION: Single source of truth for what each dict must contain

Usage:
    from coordinator.contracts import PendingApproval, LedgerEntry, validate

    # Type-checked at edit time
    approval: PendingApproval = {...}
    ts = approval["created_at"]   # ✓ editor knows this exists
    ts = approval["suspended_at"] # ✗ editor flags unknown key

    # Validated at runtime (in tests or dev mode)
    validate(approval, PendingApproval, "list_pending_approvals")
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict, Required, get_type_hints


# ═══════════════════════════════════════════════════════════════════
# list_pending_approvals() → cmd_pending()
# ═══════════════════════════════════════════════════════════════════

class PendingApproval(TypedDict):
    """Returned by list_pending_approvals, consumed by cmd_pending."""
    task_id: str
    instance_id: str
    workflow_type: str
    domain: str
    governance_tier: str
    correlation_id: str
    queue: str
    created_at: float
    sla_seconds: int | None
    expires_at: float | None
    priority: int
    callback_method: str
    resume_nonce: str


# ═══════════════════════════════════════════════════════════════════
# store.get_ledger() → cmd_ledger()
# ═══════════════════════════════════════════════════════════════════

class LedgerEntry(TypedDict):
    """Returned by store.get_ledger, consumed by cmd_ledger."""
    id: int
    instance_id: str
    correlation_id: str
    action_type: str
    details: dict[str, Any]
    idempotency_key: str | None
    created_at: float


# ═══════════════════════════════════════════════════════════════════
# delegation inputs resolved by _resolve_inputs()
# ═══════════════════════════════════════════════════════════════════

class DelegationInputs(TypedDict, total=False):
    """
    Resolved delegation inputs passed to handler workflow as case_input.

    CRITICAL: if the handler workflow has a retrieve step, the inputs
    MUST include get_* keys with dict values. Without them,
    _build_tool_registry creates an empty registry and the LLM hangs.

    Scalar fields come from the delegation contract.
    Tool fields (get_*) come from ${source.input.get_*} mappings.
    """
    pass  # Dynamic keys — validated by has_tool_data()


def has_tool_data(inputs: dict[str, Any]) -> bool:
    """Check that delegation inputs contain tool data for retrieve steps."""
    return any(
        isinstance(v, (dict, list)) and k.startswith("get_")
        for k, v in inputs.items()
    )


# ═══════════════════════════════════════════════════════════════════
# _extract_result_summary() → instance.result
# ═══════════════════════════════════════════════════════════════════

class ResultSummary(TypedDict, total=False):
    """Stored in instance.result after completion."""
    step_count: int
    steps: list[dict[str, Any]]
    final_output: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# stats() return value
# ═══════════════════════════════════════════════════════════════════

class CoordinatorStats(TypedDict, total=False):
    """Returned by coord.stats(), consumed by cmd_stats (json.dumps)."""
    total_instances: int
    by_status: dict[str, int]
    total_work_orders: int
    total_ledger_entries: int
    pending_approvals: int


# ═══════════════════════════════════════════════════════════════════
# Runtime validation
# ═══════════════════════════════════════════════════════════════════

def validate(data: dict, contract: type, context: str = "") -> list[str]:
    """
    Validate a dict against a TypedDict contract at runtime.

    Returns list of missing required keys. Empty list = valid.

    Usage in dev/test:
        errors = validate(approval, PendingApproval, "list_pending_approvals")
        assert not errors, f"Contract violation: {errors}"
    """
    hints = get_type_hints(contract)
    required = getattr(contract, "__required_keys__", set(hints.keys()))

    missing = []
    for key in required:
        if key not in data:
            missing.append(f"{context}: missing required key '{key}'")
    return missing


def validate_all(items: list[dict], contract: type, context: str = "") -> list[str]:
    """Validate a list of dicts against a TypedDict."""
    errors = []
    for i, item in enumerate(items):
        errors.extend(validate(item, contract, f"{context}[{i}]"))
    return errors


# ═══════════════════════════════════════════════════════════════════
# Dev-mode boundary assertion (opt-in via COGNITIVE_CORE_STRICT=1)
# ═══════════════════════════════════════════════════════════════════

_STRICT = "COGNITIVE_CORE_STRICT" in  __import__("os").environ


def assert_contract(data: dict, contract: type, context: str = ""):
    """
    In strict mode, raise immediately on contract violation.
    In normal mode, no-op (zero overhead in prod).

    Enable: export COGNITIVE_CORE_STRICT=1
    """
    if not _STRICT:
        return
    errors = validate(data, contract, context)
    if errors:
        raise ContractViolation(
            f"Contract violation at {context}:\n  " + "\n  ".join(errors)
        )


class ContractViolation(Exception):
    """Raised in strict mode when a dict doesn't match its TypedDict contract."""
    pass
