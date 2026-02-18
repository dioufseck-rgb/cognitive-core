"""
Cognitive Core — HITL Capability-Based Routing (S-020)

Routes governance approval tasks to qualified reviewers based on
required capabilities. Each domain/tier combination declares what
capability is needed, and reviewers are assigned capabilities.

Architecture:
  - RoutingConfig: maps (domain, tier) → required_capability
  - Reviewer: has a set of capabilities
  - RoutingManager: maintains queues and routes tasks

Usage:
    from engine.hitl_routing import RoutingManager, Reviewer, CapabilityRoute

    mgr = RoutingManager()
    mgr.add_route(CapabilityRoute(domain="card_dispute", tier="gate", capability="fraud_review_l2"))
    mgr.add_route(CapabilityRoute(domain="*", tier="hold", capability="compliance_review"))

    mgr.register_reviewer(Reviewer(id="alice", capabilities={"fraud_review_l2", "compliance_review"}))
    mgr.register_reviewer(Reviewer(id="bob", capabilities={"fraud_review_l2"}))

    # Route a task
    routed = mgr.route_task(instance_id="wf_1", domain="card_dispute", tier="gate")
    # Returns list of qualified reviewer IDs: ["alice", "bob"]
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.hitl_routing")


@dataclass
class CapabilityRoute:
    """Maps a (domain, tier) combination to a required capability."""
    domain: str      # "*" matches all domains
    tier: str        # "*" matches all tiers
    capability: str  # Required capability name


@dataclass
class Reviewer:
    """A human reviewer with assigned capabilities."""
    id: str
    name: str = ""
    capabilities: set[str] = field(default_factory=set)
    active: bool = True
    max_concurrent: int = 10  # Max simultaneous tasks
    current_load: int = 0

    @property
    def available(self) -> bool:
        return self.active and self.current_load < self.max_concurrent


@dataclass
class RoutedTask:
    """A task that has been routed to qualified reviewers."""
    task_id: str
    instance_id: str
    domain: str
    tier: str
    required_capability: str
    qualified_reviewers: list[str]
    assigned_to: str = ""      # Reviewer who claimed it
    status: str = "pending"    # pending, assigned, completed, expired
    created_at: float = 0.0
    assigned_at: float = 0.0
    completed_at: float = 0.0


class RoutingManager:
    """
    Routes governance approval tasks to qualified reviewers.

    Maintains capability routes, reviewer registry, and task queues.
    """

    def __init__(self):
        self._routes: list[CapabilityRoute] = []
        self._reviewers: dict[str, Reviewer] = {}
        self._tasks: dict[str, RoutedTask] = {}
        self._lock = threading.Lock()

    # ── Route Configuration ──────────────────────────────────

    def add_route(self, route: CapabilityRoute):
        """Add a routing rule."""
        with self._lock:
            self._routes.append(route)
            logger.info(
                "Route added: (%s, %s) → %s",
                route.domain, route.tier, route.capability,
            )

    def remove_route(self, domain: str, tier: str) -> bool:
        """Remove a routing rule."""
        with self._lock:
            before = len(self._routes)
            self._routes = [
                r for r in self._routes
                if not (r.domain == domain and r.tier == tier)
            ]
            return len(self._routes) < before

    def get_required_capability(self, domain: str, tier: str) -> str | None:
        """
        Find the required capability for a (domain, tier) pair.
        Checks exact match first, then wildcard matches.
        """
        with self._lock:
            # Exact match
            for route in self._routes:
                if route.domain == domain and route.tier == tier:
                    return route.capability

            # Domain wildcard
            for route in self._routes:
                if route.domain == "*" and route.tier == tier:
                    return route.capability

            # Tier wildcard
            for route in self._routes:
                if route.domain == domain and route.tier == "*":
                    return route.capability

            # Full wildcard
            for route in self._routes:
                if route.domain == "*" and route.tier == "*":
                    return route.capability

            return None

    # ── Reviewer Management ──────────────────────────────────

    def register_reviewer(self, reviewer: Reviewer):
        """Register or update a reviewer."""
        with self._lock:
            self._reviewers[reviewer.id] = reviewer
            logger.info("Reviewer registered: %s (%s)", reviewer.id, reviewer.capabilities)

    def deactivate_reviewer(self, reviewer_id: str) -> bool:
        with self._lock:
            if reviewer_id in self._reviewers:
                self._reviewers[reviewer_id].active = False
                return True
            return False

    def activate_reviewer(self, reviewer_id: str) -> bool:
        with self._lock:
            if reviewer_id in self._reviewers:
                self._reviewers[reviewer_id].active = True
                return True
            return False

    def get_qualified_reviewers(self, capability: str) -> list[Reviewer]:
        """Find all active, available reviewers with a given capability."""
        with self._lock:
            return [
                r for r in self._reviewers.values()
                if capability in r.capabilities and r.available
            ]

    # ── Task Routing ─────────────────────────────────────────

    def route_task(
        self,
        instance_id: str,
        domain: str,
        tier: str,
        context: dict[str, Any] | None = None,
    ) -> RoutedTask | None:
        """
        Route an approval task to qualified reviewers.

        Returns RoutedTask with qualified reviewer list, or None if
        no capability route is configured.
        """
        capability = self.get_required_capability(domain, tier)
        if capability is None:
            logger.warning(
                "No capability route for (%s, %s) — task will go to generic queue",
                domain, tier,
            )
            return None

        qualified = self.get_qualified_reviewers(capability)
        qualified_ids = [r.id for r in qualified]

        task = RoutedTask(
            task_id=f"rt_{uuid.uuid4().hex[:12]}",
            instance_id=instance_id,
            domain=domain,
            tier=tier,
            required_capability=capability,
            qualified_reviewers=qualified_ids,
            created_at=time.time(),
        )

        with self._lock:
            self._tasks[task.task_id] = task

        logger.info(
            "Task %s routed: (%s, %s) needs %s → %d qualified reviewers",
            task.task_id, domain, tier, capability, len(qualified_ids),
        )
        return task

    def assign_task(self, task_id: str, reviewer_id: str) -> bool:
        """Assign (claim) a task to a specific reviewer."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != "pending":
                return False
            if reviewer_id not in task.qualified_reviewers:
                logger.warning(
                    "Reviewer %s not qualified for task %s (needs %s)",
                    reviewer_id, task_id, task.required_capability,
                )
                return False

            reviewer = self._reviewers.get(reviewer_id)
            if reviewer and not reviewer.available:
                logger.warning("Reviewer %s at capacity", reviewer_id)
                return False

            task.assigned_to = reviewer_id
            task.status = "assigned"
            task.assigned_at = time.time()
            if reviewer:
                reviewer.current_load += 1

            logger.info("Task %s assigned to %s", task_id, reviewer_id)
            return True

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.status = "completed"
            task.completed_at = time.time()

            # Release reviewer capacity
            if task.assigned_to:
                reviewer = self._reviewers.get(task.assigned_to)
                if reviewer and reviewer.current_load > 0:
                    reviewer.current_load -= 1

            return True

    def get_pending_for_reviewer(self, reviewer_id: str) -> list[RoutedTask]:
        """Get all pending tasks a reviewer is qualified for."""
        with self._lock:
            reviewer = self._reviewers.get(reviewer_id)
            if reviewer is None:
                return []
            return [
                t for t in self._tasks.values()
                if t.status == "pending"
                and reviewer_id in t.qualified_reviewers
            ]

    def get_assigned_for_reviewer(self, reviewer_id: str) -> list[RoutedTask]:
        """Get tasks assigned to a specific reviewer."""
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.assigned_to == reviewer_id
                and t.status == "assigned"
            ]

    # ── Stats ────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            status_counts = defaultdict(int)
            for t in self._tasks.values():
                status_counts[t.status] += 1

            return {
                "total_routes": len(self._routes),
                "total_reviewers": len(self._reviewers),
                "active_reviewers": sum(1 for r in self._reviewers.values() if r.active),
                "total_tasks": len(self._tasks),
                "tasks_by_status": dict(status_counts),
            }
