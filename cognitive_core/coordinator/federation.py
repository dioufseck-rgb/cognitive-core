"""
Cognitive Core — Coordinator Federation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hierarchical federation: departments own coordinators with their own
workflows, domains, resources, and policies. When a local coordinator
cannot fulfill a need, it escalates to a parent coordinator that has
visibility across all children and can route work to whichever child
can serve it.

Architecture:
  ┌─────────────────────────────────┐
  │  Parent Coordinator (Enterprise) │
  │  ─ sees all child capabilities   │
  │  ─ routes cross-department work  │
  │  ─ enforces global policies      │
  └───────┬─────────────┬───────────┘
          │             │
  ┌───────▼──────┐ ┌───▼────────────┐
  │ Child: Claims │ │ Child: BSA/AML  │
  │ ─ adjudication│ │ ─ SAR screening │
  │ ─ fraud review│ │ ─ investigation │
  │ ─ subrogation │ │ ─ CTR filing    │
  └──────────────┘ └────────────────┘

Flow:
  1. BSA coordinator gets a need for "fraud_review" (Claims capability)
  2. BSA doesn't have that capability locally
  3. BSA escalates to Parent: "I need fraud_review for work order X"
  4. Parent sees Claims child has fraud_review capability
  5. Parent dispatches to Claims child
  6. Claims child runs fraud_review workflow, completes
  7. Parent routes result back to BSA coordinator
  8. BSA's work order completes, workflow resumes

Key design:
  - Children register with parent via FederationRegistration
  - Parent maintains a FederatedCapabilityIndex (child_id → capabilities)
  - Escalation is async: child suspends WO, parent dispatches, result returns
  - Lineage tracks cross-coordinator hops for audit
  - Each coordinator keeps its own store — no shared state
  - Parent cannot see child's internal workflow state (encapsulation)
  - Parent can only dispatch to capabilities the child advertised
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ═══════════════════════════════════════════════════════════════════
# FEDERATION REGISTRATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FederatedCapability:
    """A capability advertised by a child coordinator to its parent."""
    need_type: str
    workflow_type: str
    domain: str
    # Capacity hint (parent uses for routing decisions)
    capacity_model: str = "slot"   # slot | volume | batch
    current_utilization: float = 0.0
    max_capacity: int = 10
    # Quality signal
    completed_count: int = 0
    avg_latency_seconds: float = 0.0
    error_rate: float = 0.0


@dataclass
class FederationRegistration:
    """
    A child coordinator's registration with its parent.
    
    Contains the child's identity, advertised capabilities,
    and a dispatch callback the parent uses to send work.
    """
    child_id: str
    child_name: str  # human-readable: "Claims Processing", "BSA/AML"
    capabilities: list[FederatedCapability] = field(default_factory=list)

    # How the parent sends work to this child
    # Signature: dispatch_fn(need_type, work_order_id, context, correlation_id) → receipt_id
    dispatch_fn: Callable[..., str] | None = None

    # How the parent checks work status
    # Signature: poll_fn(receipt_id) → {"status": ..., "result": ...} | None
    poll_fn: Callable[..., dict | None] | None = None

    # Health
    registered_at: float = 0.0
    last_heartbeat: float = 0.0
    status: str = "active"  # active | degraded | offline

    @staticmethod
    def create(
        child_id: str,
        child_name: str,
        capabilities: list[FederatedCapability],
        dispatch_fn: Callable | None = None,
        poll_fn: Callable | None = None,
    ) -> FederationRegistration:
        now = time.time()
        return FederationRegistration(
            child_id=child_id,
            child_name=child_name,
            capabilities=capabilities,
            dispatch_fn=dispatch_fn,
            poll_fn=poll_fn,
            registered_at=now,
            last_heartbeat=now,
        )


# ═══════════════════════════════════════════════════════════════════
# FEDERATED DISPATCH RECEIPT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FederationReceipt:
    """
    Tracks a cross-coordinator dispatch from parent's perspective.
    
    Created when parent routes work to a child. Used to poll for
    completion and route results back to the originating coordinator.
    """
    receipt_id: str
    source_child_id: str       # who asked for the work
    source_work_order_id: str  # WO in source child's store
    target_child_id: str       # who is doing the work
    target_receipt_id: str     # receipt from target's dispatch_fn
    need_type: str
    correlation_id: str
    dispatched_at: float = 0.0
    completed_at: float | None = None
    status: str = "dispatched"  # dispatched | completed | failed | timeout
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @staticmethod
    def create(
        source_child_id: str,
        source_work_order_id: str,
        target_child_id: str,
        target_receipt_id: str,
        need_type: str,
        correlation_id: str,
    ) -> FederationReceipt:
        return FederationReceipt(
            receipt_id=f"fed_{uuid.uuid4().hex[:12]}",
            source_child_id=source_child_id,
            source_work_order_id=source_work_order_id,
            target_child_id=target_child_id,
            target_receipt_id=target_receipt_id,
            need_type=need_type,
            correlation_id=correlation_id,
            dispatched_at=time.time(),
        )


# ═══════════════════════════════════════════════════════════════════
# FEDERATION HUB (PARENT COORDINATOR MIXIN)
# ═══════════════════════════════════════════════════════════════════

class FederationHub:
    """
    Parent-side federation manager.
    
    Maintains the index of child coordinators and their capabilities.
    Routes cross-department work requests and tracks completion.
    
    Usage:
        hub = FederationHub("enterprise")
        hub.register_child(claims_registration)
        hub.register_child(bsa_registration)
        
        # When a child can't fulfill a need locally:
        receipt = hub.federated_dispatch(
            source_child_id="bsa_aml",
            source_work_order_id="wo_123",
            need_type="fraud_review",
            context={...},
            correlation_id="cor_456",
        )
        
        # Poll for completion:
        result = hub.check_completion(receipt.receipt_id)
    """

    def __init__(self, hub_id: str = "enterprise"):
        self.hub_id = hub_id
        self._children: dict[str, FederationRegistration] = {}
        self._capability_index: dict[str, list[str]] = {}  # need_type → [child_ids]
        self._receipts: dict[str, FederationReceipt] = {}
        self._dispatch_log: list[dict[str, Any]] = []

    # ── Child Registration ────────────────────────────────────────

    def register_child(self, registration: FederationRegistration) -> None:
        """Register a child coordinator with the hub."""
        self._children[registration.child_id] = registration
        # Build capability index
        for cap in registration.capabilities:
            self._capability_index.setdefault(cap.need_type, [])
            if registration.child_id not in self._capability_index[cap.need_type]:
                self._capability_index[cap.need_type].append(registration.child_id)

    def unregister_child(self, child_id: str) -> None:
        """Remove a child coordinator."""
        reg = self._children.pop(child_id, None)
        if reg:
            # Rebuild index
            self._capability_index = {}
            for cid, child in self._children.items():
                for cap in child.capabilities:
                    self._capability_index.setdefault(cap.need_type, [])
                    if cid not in self._capability_index[cap.need_type]:
                        self._capability_index[cap.need_type].append(cid)

    def heartbeat(self, child_id: str) -> bool:
        """Update child heartbeat. Returns False if child not registered."""
        child = self._children.get(child_id)
        if not child:
            return False
        child.last_heartbeat = time.time()
        return True

    # ── Capability Discovery ──────────────────────────────────────

    def can_fulfill(self, need_type: str, exclude_child: str = "") -> bool:
        """Check if any child (other than excluded) can fulfill a need."""
        candidates = self._capability_index.get(need_type, [])
        return any(
            cid != exclude_child and self._children[cid].status == "active"
            for cid in candidates
        )

    def find_providers(self, need_type: str, exclude_child: str = "") -> list[FederationRegistration]:
        """Find all active children that can serve a need type."""
        candidates = self._capability_index.get(need_type, [])
        return [
            self._children[cid]
            for cid in candidates
            if cid != exclude_child
            and cid in self._children
            and self._children[cid].status == "active"
        ]

    def list_capabilities(self) -> dict[str, list[str]]:
        """Return full capability index: need_type → [child_ids]."""
        return dict(self._capability_index)

    # ── Federated Dispatch ────────────────────────────────────────

    def federated_dispatch(
        self,
        source_child_id: str,
        source_work_order_id: str,
        need_type: str,
        context: dict[str, Any],
        correlation_id: str,
    ) -> FederationReceipt | None:
        """
        Route a work request from one child to another.
        
        Called when a child coordinator cannot fulfill a need locally
        and escalates to the parent.
        
        Returns a FederationReceipt tracking the cross-coordinator dispatch,
        or None if no child can fulfill the need.
        """
        providers = self.find_providers(need_type, exclude_child=source_child_id)
        if not providers:
            self._dispatch_log.append({
                "event": "no_provider",
                "source": source_child_id,
                "need_type": need_type,
                "timestamp": time.time(),
            })
            return None

        # Select best provider (simple: lowest utilization among active)
        best = min(providers, key=lambda p: self._child_utilization(p, need_type))

        if not best.dispatch_fn:
            self._dispatch_log.append({
                "event": "no_dispatch_fn",
                "source": source_child_id,
                "target": best.child_id,
                "need_type": need_type,
                "timestamp": time.time(),
            })
            return None

        # Dispatch to target child
        try:
            target_receipt_id = best.dispatch_fn(
                need_type=need_type,
                work_order_id=source_work_order_id,
                context=context,
                correlation_id=correlation_id,
            )
        except Exception as e:
            self._dispatch_log.append({
                "event": "dispatch_error",
                "source": source_child_id,
                "target": best.child_id,
                "need_type": need_type,
                "error": str(e),
                "timestamp": time.time(),
            })
            return None

        receipt = FederationReceipt.create(
            source_child_id=source_child_id,
            source_work_order_id=source_work_order_id,
            target_child_id=best.child_id,
            target_receipt_id=target_receipt_id,
            need_type=need_type,
            correlation_id=correlation_id,
        )
        self._receipts[receipt.receipt_id] = receipt

        self._dispatch_log.append({
            "event": "dispatched",
            "receipt_id": receipt.receipt_id,
            "source": source_child_id,
            "target": best.child_id,
            "need_type": need_type,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        })

        return receipt

    # ── Completion Tracking ───────────────────────────────────────

    def check_completion(self, receipt_id: str) -> FederationReceipt | None:
        """
        Poll a federated dispatch for completion.
        
        Calls the target child's poll_fn to check status.
        If complete, updates the receipt with result data.
        """
        receipt = self._receipts.get(receipt_id)
        if not receipt or receipt.status in ("completed", "failed"):
            return receipt

        target = self._children.get(receipt.target_child_id)
        if not target or not target.poll_fn:
            return receipt

        try:
            result = target.poll_fn(receipt.target_receipt_id)
        except Exception as e:
            receipt.status = "failed"
            receipt.error = f"Poll error: {e}"
            return receipt

        if result is None:
            return receipt  # still in progress

        status = result.get("status", "unknown")
        if status == "completed":
            receipt.status = "completed"
            receipt.completed_at = time.time()
            receipt.result = result.get("result", {})
            self._dispatch_log.append({
                "event": "completed",
                "receipt_id": receipt.receipt_id,
                "source": receipt.source_child_id,
                "target": receipt.target_child_id,
                "elapsed": receipt.completed_at - receipt.dispatched_at,
                "timestamp": time.time(),
            })
        elif status == "failed":
            receipt.status = "failed"
            receipt.error = result.get("error", "unknown")
            self._dispatch_log.append({
                "event": "failed",
                "receipt_id": receipt.receipt_id,
                "target": receipt.target_child_id,
                "error": receipt.error,
                "timestamp": time.time(),
            })

        return receipt

    def sweep_receipts(self, timeout_seconds: float = 3600.0) -> list[FederationReceipt]:
        """
        Check all pending receipts for completion or timeout.
        Returns list of newly completed/failed/timed-out receipts.
        """
        changed = []
        now = time.time()
        for receipt in self._receipts.values():
            if receipt.status not in ("dispatched",):
                continue
            # Check timeout
            if now - receipt.dispatched_at > timeout_seconds:
                receipt.status = "timeout"
                receipt.error = f"Exceeded {timeout_seconds}s timeout"
                changed.append(receipt)
                continue
            # Poll
            updated = self.check_completion(receipt.receipt_id)
            if updated and updated.status != "dispatched":
                changed.append(updated)
        return changed

    # ── Internal Helpers ──────────────────────────────────────────

    def _child_utilization(self, child: FederationRegistration, need_type: str) -> float:
        """Estimate utilization for routing. Lower = preferred."""
        for cap in child.capabilities:
            if cap.need_type == need_type:
                return cap.current_utilization
        return 1.0  # unknown = full

    @property
    def pending_count(self) -> int:
        return sum(1 for r in self._receipts.values() if r.status == "dispatched")

    @property
    def dispatch_log(self) -> list[dict[str, Any]]:
        return list(self._dispatch_log)


# ═══════════════════════════════════════════════════════════════════
# FEDERATION SPOKE (CHILD COORDINATOR MIXIN)
# ═══════════════════════════════════════════════════════════════════

class FederationSpoke:
    """
    Child-side federation manager.
    
    Handles escalation of unresolved needs to the parent hub,
    and receiving inbound work from the parent.
    
    Usage:
        spoke = FederationSpoke(
            child_id="bsa_aml",
            child_name="BSA/AML Division",
            parent_hub=hub,
        )
        spoke.advertise_capabilities([
            FederatedCapability(need_type="sar_screen", ...),
        ])
        
        # When local capability not found:
        receipt = spoke.escalate("fraud_review", wo_id, context, cor_id)
        
        # Later, check if parent got it done:
        result = spoke.check_escalation(receipt.receipt_id)
    """

    def __init__(
        self,
        child_id: str,
        child_name: str,
        parent_hub: FederationHub | None = None,
    ):
        self.child_id = child_id
        self.child_name = child_name
        self._parent_hub = parent_hub
        self._inbound_work: dict[str, dict[str, Any]] = {}  # receipt_id → work
        self._escalation_receipts: dict[str, FederationReceipt] = {}

    def connect_to_parent(self, hub: FederationHub) -> None:
        """Connect to a parent hub (can be done after construction)."""
        self._parent_hub = hub

    def advertise_capabilities(self, capabilities: list[FederatedCapability]) -> None:
        """Register this child's capabilities with the parent hub."""
        if not self._parent_hub:
            raise RuntimeError("No parent hub connected")

        registration = FederationRegistration.create(
            child_id=self.child_id,
            child_name=self.child_name,
            capabilities=capabilities,
            dispatch_fn=self._receive_work,
            poll_fn=self._poll_work,
        )
        self._parent_hub.register_child(registration)

    # ── Escalation (outbound) ─────────────────────────────────────

    @property
    def has_parent(self) -> bool:
        return self._parent_hub is not None

    def escalate(
        self,
        need_type: str,
        work_order_id: str,
        context: dict[str, Any],
        correlation_id: str,
    ) -> FederationReceipt | None:
        """
        Escalate an unresolved need to the parent hub.
        
        Returns a FederationReceipt if the parent accepted the dispatch,
        or None if the parent can't fulfill it either.
        """
        if not self._parent_hub:
            return None

        receipt = self._parent_hub.federated_dispatch(
            source_child_id=self.child_id,
            source_work_order_id=work_order_id,
            need_type=need_type,
            context=context,
            correlation_id=correlation_id,
        )
        if receipt:
            self._escalation_receipts[receipt.receipt_id] = receipt
        return receipt

    def check_escalation(self, receipt_id: str) -> FederationReceipt | None:
        """Check if an escalated work request has been completed."""
        if not self._parent_hub:
            return None
        return self._parent_hub.check_completion(receipt_id)

    def check_all_escalations(self) -> list[FederationReceipt]:
        """Check all pending escalations. Returns newly completed ones."""
        completed = []
        for receipt_id, receipt in list(self._escalation_receipts.items()):
            if receipt.status != "dispatched":
                continue
            updated = self.check_escalation(receipt_id)
            if updated and updated.status != "dispatched":
                completed.append(updated)
        return completed

    # ── Inbound Work (from parent) ────────────────────────────────

    def _receive_work(
        self,
        need_type: str,
        work_order_id: str,
        context: dict[str, Any],
        correlation_id: str,
    ) -> str:
        """
        Called by the parent hub to dispatch work to this child.
        Returns a receipt_id that the parent can use to poll.
        
        The child coordinator should process this work using its
        local workflows and update the inbound work entry on completion.
        """
        inbound_id = f"inb_{uuid.uuid4().hex[:12]}"
        self._inbound_work[inbound_id] = {
            "inbound_id": inbound_id,
            "need_type": need_type,
            "source_work_order_id": work_order_id,
            "context": context,
            "correlation_id": correlation_id,
            "received_at": time.time(),
            "status": "received",
            "result": None,
        }
        return inbound_id

    def _poll_work(self, inbound_id: str) -> dict | None:
        """
        Called by the parent hub to check inbound work status.
        Returns {"status": ..., "result": ...} when done, None if pending.
        """
        work = self._inbound_work.get(inbound_id)
        if not work:
            return {"status": "failed", "error": "Unknown inbound_id"}
        if work["status"] in ("completed", "failed"):
            return {
                "status": work["status"],
                "result": work.get("result", {}),
                "error": work.get("error"),
            }
        return None  # still in progress

    def complete_inbound(self, inbound_id: str, result: dict[str, Any]) -> None:
        """Mark inbound work as completed with result."""
        work = self._inbound_work.get(inbound_id)
        if work:
            work["status"] = "completed"
            work["result"] = result
            work["completed_at"] = time.time()

    def fail_inbound(self, inbound_id: str, error: str) -> None:
        """Mark inbound work as failed."""
        work = self._inbound_work.get(inbound_id)
        if work:
            work["status"] = "failed"
            work["error"] = error

    @property
    def pending_inbound_count(self) -> int:
        return sum(1 for w in self._inbound_work.values() if w["status"] == "received")

    def get_pending_inbound(self) -> list[dict[str, Any]]:
        """Get all pending inbound work items for processing."""
        return [w for w in self._inbound_work.values() if w["status"] == "received"]
