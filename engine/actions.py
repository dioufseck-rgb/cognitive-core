"""
Cognitive Core - Action Registry

Write-side counterpart to ToolRegistry (which handles reads).

An "action" is a callable that takes parameters and performs a side effect:
issuing a credit, sending a letter, updating a status, filing a report.
Actions are registered by name and organized by domain.

The Act primitive's node resolves which actions to take (via LLM or
deterministic mapping), checks authorization, then executes or simulates
through this registry.

Key differences from ToolRegistry:
- Actions have side effects (tools are read-only)
- Actions require authorization checks
- Actions support dry_run (simulation) mode
- Actions track reversibility and rollback handles

Usage:
    registry = ActionRegistry()
    registry.register(
        name="issue_provisional_credit",
        fn=my_credit_fn,
        description="Issue provisional credit to member account",
        authorization_level="agent",
        reversible=True,
        amount_threshold=500.00,
    )

    # At workflow build time, pass to create_act_node
    node = create_act_node(..., action_registry=registry)
"""

import time
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Authorization levels — ordered from least to most privileged
# ---------------------------------------------------------------------------

class AuthLevel(str, Enum):
    """Authorization levels for action execution."""
    SYSTEM = "system"          # Automated, no human in the loop
    AGENT = "agent"            # Frontline agent or automated agent
    SUPERVISOR = "supervisor"  # Requires supervisor approval
    MANAGER = "manager"        # Requires management approval
    COMPLIANCE = "compliance"  # Requires compliance officer sign-off


# ---------------------------------------------------------------------------
# Action result
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Result of executing or simulating a single action."""
    action: str
    target_system: str
    status: str  # executed | simulated | blocked | failed
    confirmation_id: Optional[str] = None
    response_data: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    reversible: bool = True
    rollback_handle: Optional[str] = None


# ---------------------------------------------------------------------------
# Action specification
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    """Registration entry for an action."""
    name: str
    fn: Callable[..., dict[str, Any]]
    description: str = ""
    target_system: str = ""
    authorization_level: str = "supervisor"  # default to supervisor approval
    reversible: bool = True
    amount_threshold: Optional[float] = None  # above this, escalate authorization
    requires_confirmation: bool = False       # require explicit confirmation before execution
    cooldown_seconds: Optional[float] = None  # minimum time between executions
    side_effects: list[str] = field(default_factory=list)
    rollback_fn: Optional[Callable[..., dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Action Registry
# ---------------------------------------------------------------------------

class ActionRegistry:
    """
    Central registry of executable actions.

    Supports:
    1. Registration with authorization and reversibility metadata
    2. Dry-run (simulation) mode — returns what would happen
    3. Execution mode — calls the action function
    4. Authorization checking against the current context
    5. Rollback via stored rollback handles
    """

    def __init__(self):
        self._actions: dict[str, ActionSpec] = {}
        self._execution_log: list[ActionResult] = []

    def register(
        self,
        name: str,
        fn: Callable[..., dict[str, Any]],
        description: str = "",
        target_system: str = "",
        authorization_level: str = "supervisor",
        reversible: bool = True,
        amount_threshold: Optional[float] = None,
        requires_confirmation: bool = False,
        cooldown_seconds: Optional[float] = None,
        side_effects: Optional[list[str]] = None,
        rollback_fn: Optional[Callable[..., dict[str, Any]]] = None,
    ):
        """Register an executable action."""
        self._actions[name] = ActionSpec(
            name=name,
            fn=fn,
            description=description,
            target_system=target_system,
            authorization_level=authorization_level,
            reversible=reversible,
            amount_threshold=amount_threshold,
            requires_confirmation=requires_confirmation,
            cooldown_seconds=cooldown_seconds,
            side_effects=side_effects or [],
            rollback_fn=rollback_fn,
        )

    def get(self, name: str) -> Optional[ActionSpec]:
        return self._actions.get(name)

    def list_actions(self) -> list[str]:
        return list(self._actions.keys())

    def describe(self) -> str:
        """Human-readable description for LLM prompts."""
        if not self._actions:
            return "No actions registered."
        lines = []
        for spec in self._actions.values():
            auth = f" [requires: {spec.authorization_level}]"
            rev = " (reversible)" if spec.reversible else " (IRREVERSIBLE)"
            threshold = f" [amount threshold: ${spec.amount_threshold:.2f}]" if spec.amount_threshold else ""
            lines.append(f"  - {spec.name}{auth}{rev}{threshold}: {spec.description}")
            if spec.side_effects:
                for se in spec.side_effects:
                    lines.append(f"    Side effect: {se}")
        return "\n".join(lines)

    def check_authorization(
        self,
        action_name: str,
        current_level: str = "agent",
        amount: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Check if the current authorization level permits this action.

        Returns:
            (authorized: bool, reason: str)
        """
        spec = self._actions.get(action_name)
        if spec is None:
            return False, f"Action '{action_name}' not registered"

        levels = list(AuthLevel)
        try:
            required_idx = levels.index(AuthLevel(spec.authorization_level))
            current_idx = levels.index(AuthLevel(current_level))
        except ValueError:
            return False, f"Unknown authorization level"

        if current_idx < required_idx:
            return False, (
                f"Action '{action_name}' requires {spec.authorization_level} "
                f"authorization, current level is {current_level}"
            )

        # Check amount threshold
        if spec.amount_threshold and amount and amount > spec.amount_threshold:
            # Escalate by one level
            if current_idx < required_idx + 1 and required_idx + 1 < len(levels):
                escalated = levels[required_idx + 1].value
                return False, (
                    f"Amount ${amount:.2f} exceeds threshold "
                    f"${spec.amount_threshold:.2f} for '{action_name}'. "
                    f"Requires {escalated} authorization."
                )

        return True, "authorized"

    def execute(
        self,
        action_name: str,
        params: dict[str, Any],
        dry_run: bool = True,
    ) -> ActionResult:
        """
        Execute or simulate an action.

        Args:
            action_name: Registered action name
            params: Parameters to pass to the action function
            dry_run: If True, simulate without executing

        Returns:
            ActionResult with execution details
        """
        spec = self._actions.get(action_name)
        if spec is None:
            return ActionResult(
                action=action_name,
                target_system="unknown",
                status="failed",
                error=f"Action '{action_name}' not registered",
            )

        if dry_run:
            result = ActionResult(
                action=action_name,
                target_system=spec.target_system,
                status="simulated",
                reversible=spec.reversible,
                response_data={"dry_run": True, "would_execute": params},
            )
            self._execution_log.append(result)
            return result

        # Live execution
        t0 = time.time()
        try:
            response = spec.fn(params)
            elapsed_ms = (time.time() - t0) * 1000
            result = ActionResult(
                action=action_name,
                target_system=spec.target_system,
                status="executed",
                confirmation_id=response.get("confirmation_id"),
                response_data=response,
                latency_ms=elapsed_ms,
                reversible=spec.reversible,
                rollback_handle=response.get("rollback_handle"),
            )
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            result = ActionResult(
                action=action_name,
                target_system=spec.target_system,
                status="failed",
                error=str(e),
                latency_ms=elapsed_ms,
                reversible=spec.reversible,
            )

        self._execution_log.append(result)
        return result

    def rollback(self, action_name: str, rollback_handle: str) -> ActionResult:
        """Attempt to reverse a previously executed action."""
        spec = self._actions.get(action_name)
        if spec is None:
            return ActionResult(
                action=f"rollback:{action_name}",
                target_system="unknown",
                status="failed",
                error=f"Action '{action_name}' not registered",
            )

        if not spec.reversible:
            return ActionResult(
                action=f"rollback:{action_name}",
                target_system=spec.target_system,
                status="failed",
                error=f"Action '{action_name}' is not reversible",
            )

        if spec.rollback_fn is None:
            return ActionResult(
                action=f"rollback:{action_name}",
                target_system=spec.target_system,
                status="failed",
                error=f"No rollback function registered for '{action_name}'",
            )

        t0 = time.time()
        try:
            response = spec.rollback_fn({"rollback_handle": rollback_handle})
            elapsed_ms = (time.time() - t0) * 1000
            return ActionResult(
                action=f"rollback:{action_name}",
                target_system=spec.target_system,
                status="executed",
                confirmation_id=response.get("confirmation_id"),
                response_data=response,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            return ActionResult(
                action=f"rollback:{action_name}",
                target_system=spec.target_system,
                status="failed",
                error=str(e),
                latency_ms=elapsed_ms,
            )

    def get_execution_log(self) -> list[ActionResult]:
        """Return all actions executed or simulated in this session."""
        return list(self._execution_log)

    def clear_log(self):
        """Reset the execution log."""
        self._execution_log.clear()


# ---------------------------------------------------------------------------
# Case-simulation registry — for dev/test
# ---------------------------------------------------------------------------

def create_simulation_registry() -> ActionRegistry:
    """
    Create an ActionRegistry that simulates common financial actions.

    In dev/test, actions don't touch real systems. They return
    plausible confirmation IDs and response data for downstream
    steps to consume.

    In production, replace these with real API integrations.
    """
    import uuid

    registry = ActionRegistry()

    def _sim_provisional_credit(params: dict) -> dict:
        return {
            "confirmation_id": f"PC-{uuid.uuid4().hex[:8].upper()}",
            "rollback_handle": f"RB-{uuid.uuid4().hex[:8].upper()}",
            "amount": params.get("amount", 0),
            "account": params.get("account_id", "unknown"),
            "status": "posted",
        }

    def _sim_send_letter(params: dict) -> dict:
        return {
            "confirmation_id": f"LTR-{uuid.uuid4().hex[:8].upper()}",
            "delivery_method": params.get("delivery_method", "mail"),
            "recipient": params.get("recipient", "unknown"),
            "status": "queued",
        }

    def _sim_update_case_status(params: dict) -> dict:
        return {
            "confirmation_id": f"CS-{uuid.uuid4().hex[:8].upper()}",
            "rollback_handle": f"RB-{uuid.uuid4().hex[:8].upper()}",
            "case_id": params.get("case_id", "unknown"),
            "new_status": params.get("status", "updated"),
        }

    def _sim_reverse_credit(params: dict) -> dict:
        return {
            "confirmation_id": f"REV-{uuid.uuid4().hex[:8].upper()}",
            "reversed_handle": params.get("rollback_handle", "unknown"),
            "status": "reversed",
        }

    registry.register(
        name="issue_provisional_credit",
        fn=_sim_provisional_credit,
        description="Issue a provisional credit to the member's account",
        target_system="core_banking",
        authorization_level="agent",
        reversible=True,
        amount_threshold=500.00,
        side_effects=[
            "Account balance updated immediately",
            "Transaction posted to statement",
            "Member notification sent (email + push)",
        ],
        rollback_fn=_sim_reverse_credit,
    )

    registry.register(
        name="send_member_letter",
        fn=_sim_send_letter,
        description="Send a letter or notification to the member",
        target_system="correspondence",
        authorization_level="system",
        reversible=False,
        side_effects=[
            "Letter queued for delivery",
            "Copy stored in member correspondence history",
        ],
    )

    registry.register(
        name="update_case_status",
        fn=_sim_update_case_status,
        description="Update the dispute or case status in the case management system",
        target_system="case_management",
        authorization_level="agent",
        reversible=True,
        side_effects=[
            "Case status updated",
            "Audit trail entry created",
            "SLA timer may be reset or stopped",
        ],
        rollback_fn=lambda p: {"confirmation_id": f"REV-{__import__('uuid').uuid4().hex[:8].upper()}", "status": "reverted"},
    )

    registry.register(
        name="file_regulatory_report",
        fn=lambda p: {"confirmation_id": f"REG-{__import__('uuid').uuid4().hex[:8].upper()}", "filing_type": p.get("report_type", "unknown"), "status": "filed"},
        description="File a regulatory report (SAR, CTR, etc.)",
        target_system="regulatory_filing",
        authorization_level="compliance",
        reversible=False,
        requires_confirmation=True,
        side_effects=[
            "Report filed with FinCEN",
            "Regulatory clock started",
            "Internal compliance log updated",
        ],
    )

    registry.register(
        name="schedule_callback",
        fn=lambda p: {"confirmation_id": f"CB-{__import__('uuid').uuid4().hex[:8].upper()}", "scheduled_for": p.get("datetime", "unknown"), "status": "scheduled"},
        description="Schedule a callback or follow-up with the member",
        target_system="scheduling",
        authorization_level="agent",
        reversible=True,
        side_effects=[
            "Callback added to agent queue",
            "Member notified of scheduled callback",
        ],
    )

    return registry
