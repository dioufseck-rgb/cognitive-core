"""
Cognitive Core — Dispatch Optimizer

Implements Spec v1.1 Sections 9 (Dispatch Optimizer), 14 (Solver Architecture),
17.2 (Greedy Fallback), and 17.4 (Exploration Policy).

The optimizer is the "brain" between demand (pending work orders) and
supply (registered resources). It wires together:

  1. Eligibility filtering (from ddd.py — hard boolean predicates)
  2. Physics binding (from physics.py — domain reality → cost matrix)
  3. Archetype solving (from archetypes.py — mathematical optimization)
  4. Greedy fallback (bounded latency guarantee)
  5. Exploration policy (new resource bootstrapping)

And produces DispatchDecision objects with full audit trails.

This module is pure logic — no I/O, no persistence, no network.
The coordinator runtime (runtime.py) wires it to the real world.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from coordinator.archetypes import (
    ARCHETYPE_REGISTRY,
    AssignmentArchetype,
    AssignmentParams,
    ArchetypeSolution,
)
from coordinator.ddd import (
    DDDWorkOrder,
    ResourceRegistration,
    ResourceRegistry,
    CapacityReservation,
    DispatchDecision,
    EligibilityResult,
    RankingScore,
)
from coordinator.physics import (
    PHYSICS_REGISTRY,
    OptimizationConfig,
    Assignment,
    DefaultAssignmentPhysics,
)

log = logging.getLogger("cognitive_core.optimizer")


# ═══════════════════════════════════════════════════════════════════
# DISPATCH OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

class DispatchOptimizer:
    """
    Three-stage dispatch pipeline (Spec Sections 9, 14, 17.2, 17.4).

    Stage 1: Physics binding — load physics class, extract cost parameters
    Stage 2: Solve — try archetype solver within time budget; greedy fallback
    Stage 3: Exploration overlay — epsilon-greedy with safety gates

    All decisions carry full audit trails for Invariant 8 (Audit Completeness).
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        archetype_registry: dict[str, Any] | None = None,
        physics_registry: dict[str, type] | None = None,
        # Injectable production hooks (wired by Coordinator)
        ddr_callback: Any | None = None,         # fn(DispatchDecision, eligible, excluded, config, solution) → None
        reservation_log: Any | None = None,       # ReservationEventLog instance
        learning_enforcer: Any | None = None,     # LearningScopeEnforcer instance
    ):
        self._registry = registry
        self._archetypes = archetype_registry or ARCHETYPE_REGISTRY
        self._physics = physics_registry or PHYSICS_REGISTRY
        # Production hooks
        self._ddr_callback = ddr_callback
        self._reservation_log = reservation_log
        self._learning_enforcer = learning_enforcer

    def dispatch(
        self,
        work_orders: list[DDDWorkOrder],
        workflow: str,
        domain: str,
        config: OptimizationConfig | None = None,
    ) -> list[DispatchDecision]:
        """
        Full dispatch pipeline. Returns one DispatchDecision per work order.

        Steps:
        1. Eligibility filter → eligible resources + audit trail
        2. Physics binding → cost matrix
        3. Solve (optimal or greedy fallback)
        4. Exploration overlay
        5. Commit winning reservations, release losers
        6. Return decisions with full audit trail
        """
        config = config or OptimizationConfig()
        now = time.time()

        if not work_orders:
            return []

        # ── Stage 0: Eligibility Filter ──────────────────────────
        eligible, eligibility_audit = self._registry.filter_eligible(
            workflow, domain, work_orders[0] if work_orders else None,
        )

        if not eligible:
            log.warning(
                "No eligible resources for %s/%s (%d excluded)",
                workflow, domain, len(eligibility_audit),
            )
            return [
                DispatchDecision(
                    work_order_id=wo.work_order_id,
                    selected_resource_id=None,
                    tier="no_eligible_resources",
                    eligibility_results=eligibility_audit,
                    timestamp=now,
                )
                for wo in work_orders
            ]

        # ── Stage 1: Physics Binding ─────────────────────────────
        physics_cls = self._physics.get(config.physics, DefaultAssignmentPhysics)
        physics = physics_cls()
        params = physics.extract_parameters(work_orders, eligible, config)

        if not params.cost_matrix:
            return [
                DispatchDecision(
                    work_order_id=wo.work_order_id,
                    selected_resource_id=None,
                    tier="empty_params",
                    eligibility_results=eligibility_audit,
                    timestamp=now,
                )
                for wo in work_orders
            ]

        # ── Stage 2: Solve ───────────────────────────────────────
        # Learning guardrail: if adaptive adjustments proposed, enforce bounds
        if self._learning_enforcer and hasattr(config, 'objectives'):
            safe_weights = self._learning_enforcer.enforce(
                config.objectives,  # base weights from YAML
                config.objectives,  # proposed (same until ADP layer exists)
            )
            # If enforcer rejected, it returned base weights (no-op here)
            # When ADP layer is live, proposed_weights will differ

        solver_seed = self._compute_seed(work_orders, eligible)
        archetype = self._archetypes.get(config.archetype)

        solution: ArchetypeSolution | None = None
        tier = "optimal"

        if archetype:
            try:
                solution = archetype.solve(
                    params,
                    time_budget_seconds=config.solver_time_budget_seconds,
                    seed=solver_seed,
                )
                if solution.solver_status not in ("optimal", "feasible"):
                    solution = None
                    tier = "fallback"
            except Exception as e:
                log.warning("Solver error: %s, falling back to greedy", e)
                solution = None
                tier = "fallback"

        if solution is None and config.greedy_fallback:
            # Greedy fallback uses the SAME cost matrix (Invariant 12)
            solution = self._greedy_solve(params, work_orders, solver_seed)
            tier = "fallback"

        if solution is None:
            return [
                DispatchDecision(
                    work_order_id=wo.work_order_id,
                    selected_resource_id=None,
                    tier="no_solution",
                    eligibility_results=eligibility_audit,
                    timestamp=now,
                )
                for wo in work_orders
            ]

        # ── Stage 3: Interpret + Build Decisions ─────────────────
        assignments = physics.interpret_solution(
            solution, work_orders, eligible, params.cost_matrix,
        )
        assignment_map = {a.work_order_id: a for a in assignments}

        # ── Stage 4: Exploration Overlay ─────────────────────────
        if config.exploration_enabled:
            assignment_map = self._apply_exploration(
                assignment_map, work_orders, eligible, config,
                params.cost_matrix, physics, now,
            )

        # ── Stage 5: Reserve Capacity + Build Final Decisions ────
        decisions: list[DispatchDecision] = []
        for wo in work_orders:
            assignment = assignment_map.get(wo.work_order_id)
            if assignment:
                # Attempt reservation
                reservation = self._registry.reserve(
                    assignment.resource_id,
                    wo.work_order_id,
                    amount=1.0,
                )
                if reservation:
                    # Log reservation events
                    if self._reservation_log:
                        self._reservation_log.record(
                            reservation.reservation_id,
                            assignment.resource_id,
                            wo.work_order_id,
                            "acquire", 1.0,
                        )
                    self._registry.commit_reservation(reservation.reservation_id)
                    if self._reservation_log:
                        self._reservation_log.record(
                            reservation.reservation_id,
                            assignment.resource_id,
                            wo.work_order_id,
                            "commit",
                        )
                    decision = DispatchDecision(
                        work_order_id=wo.work_order_id,
                        selected_resource_id=assignment.resource_id,
                        reservation_id=reservation.reservation_id,
                        tier=tier,
                        eligibility_results=eligibility_audit,
                        ranking_scores=[RankingScore(
                            resource_id=assignment.resource_id,
                            total_score=assignment.score,
                            feature_scores=assignment.feature_scores,
                        )],
                        timestamp=now,
                    )
                    decisions.append(decision)
                    # DDR callback — persist audit artifact
                    if self._ddr_callback:
                        self._ddr_callback(
                            decision, eligible, eligibility_audit,
                            config, solution,
                        )
                else:
                    # Reservation failed — resource hit capacity between
                    # eligibility check and reservation. Defer.
                    decision = DispatchDecision(
                        work_order_id=wo.work_order_id,
                        selected_resource_id=None,
                        tier="reservation_denied",
                        eligibility_results=eligibility_audit,
                        timestamp=now,
                    )
                    decisions.append(decision)
                    if self._ddr_callback:
                        self._ddr_callback(
                            decision, eligible, eligibility_audit,
                            config, solution,
                        )
            else:
                # Unassigned by solver
                decision = DispatchDecision(
                    work_order_id=wo.work_order_id,
                    selected_resource_id=None,
                    tier="unassigned",
                    eligibility_results=eligibility_audit,
                    timestamp=now,
                )
                decisions.append(decision)
                if self._ddr_callback:
                    self._ddr_callback(
                        decision, eligible, eligibility_audit,
                        config, solution,
                    )

        # Log solve metadata
        log.info(
            "Dispatch: %d work orders, %d eligible resources, "
            "tier=%s, solver=%s, %dms, %d assigned, %d unassigned",
            len(work_orders), len(eligible), tier,
            solution.solver_name if solution else "none",
            solution.solve_time_ms if solution else 0,
            sum(1 for d in decisions if d.selected_resource_id),
            sum(1 for d in decisions if not d.selected_resource_id),
        )

        return decisions

    def dispatch_single(
        self,
        work_order: DDDWorkOrder,
        workflow: str,
        domain: str,
        config: OptimizationConfig | None = None,
    ) -> DispatchDecision:
        """
        Single work order dispatch — simplified path.

        Per the spec (Section 14, Integration Point note): for simple
        cases (one work order, one or few available resources), the
        Coordinator assigns directly using eligibility + ranking
        without invoking the archetype solver.
        """
        results = self.dispatch([work_order], workflow, domain, config)
        return results[0] if results else DispatchDecision(
            work_order_id=work_order.work_order_id,
            selected_resource_id=None,
            tier="error",
            timestamp=time.time(),
        )

    # ─── Greedy Fallback (Section 17.2) ──────────────────────────

    def _greedy_solve(
        self,
        params: AssignmentParams,
        work_orders: list[DDDWorkOrder],
        seed: int,
    ) -> ArchetypeSolution:
        """
        Greedy fallback using the SAME cost matrix as the optimal solver.
        Ensures Fallback Equivalence (Invariant 12).

        1. Sort work orders by urgency (SLA time remaining)
        2. For each, assign to lowest-cost resource with capacity
        3. O(W log W + W × R)
        """
        w = params.num_work_orders
        r = params.num_resources
        cost = params.cost_matrix
        remaining_cap = list(params.capacities)

        # Sort by urgency: most urgent first
        def urgency_key(idx: int) -> float:
            wo = work_orders[idx] if idx < len(work_orders) else None
            if wo and wo.sla_seconds and wo.created_at:
                remaining = wo.sla_seconds - (time.time() - wo.created_at)
                return remaining  # lower = more urgent
            if wo and wo.priority == "critical":
                return -1e6
            if wo and wo.priority == "high":
                return 0
            return 1e6

        order = sorted(range(w), key=urgency_key)

        assignments: list[tuple[int, int]] = []
        unassigned: list[int] = []
        total_cost = 0.0

        for i in order:
            best_j = -1
            best_c = float("inf")
            for j in range(r):
                if remaining_cap[j] > 0 and cost[i][j] < best_c:
                    best_c = cost[i][j]
                    best_j = j
            if best_j >= 0:
                assignments.append((i, best_j))
                remaining_cap[best_j] -= 1
                total_cost += best_c
            else:
                unassigned.append(i)

        return ArchetypeSolution(
            assignments=assignments,
            unassigned=unassigned,
            objective_value=total_cost,
            solver_status="feasible" if assignments else "infeasible",
            solver_name="greedy_fallback",
            solver_seed=seed,
        )

    # ─── Exploration Policy (Section 17.4) ───────────────────────

    def _apply_exploration(
        self,
        assignment_map: dict[str, Assignment],
        work_orders: list[DDDWorkOrder],
        eligible: list[ResourceRegistration],
        config: OptimizationConfig,
        cost_matrix: list[list[float]],
        physics: Any,
        now: float,
    ) -> dict[str, Assignment]:
        """
        Epsilon-greedy exploration overlay (Spec Section 17.4).

        For each work order eligible for exploration:
        - With probability epsilon: reassign to least-explored unproven resource
        - Epsilon decays as unproven resources accumulate data

        Safety gates (non-negotiable):
        - SLA gate: critical-priority work orders excluded
        - Circuit breaker gate: open breakers excluded
        - Max exploration ceiling: hard cap per cycle
        - Capacity gate: still must pass capacity check
        """
        proven, unproven = self._registry.partition_by_maturity(
            eligible, config.maturity_threshold,
        )

        if not unproven:
            return assignment_map  # all resources proven, nothing to explore

        # Compute decaying epsilon
        min_completions = min(r.completed_work_orders for r in unproven)
        epsilon = config.base_epsilon / (1.0 + min_completions)

        # Track exploration count for ceiling enforcement
        max_explore = max(1, int(len(work_orders) * config.max_exploration_pct))
        explore_count = 0

        # Build resource index for cost lookup
        res_idx = {r.resource_id: i for i, r in enumerate(eligible)}

        for wo in work_orders:
            # SLA gate: critical work orders never explored
            if wo.priority not in config.exploration_sla_gate:
                continue

            # Ceiling gate
            if explore_count >= max_explore:
                break

            # Epsilon check: deterministic from work order ID hash
            wo_hash = int(hashlib.sha256(
                wo.work_order_id.encode()
            ).hexdigest()[:8], 16)
            if (wo_hash % 10000) / 10000.0 >= epsilon:
                continue  # exploit (keep current assignment)

            # Find least-explored unproven resource with capacity
            best = None
            best_completions = float("inf")
            for r in unproven:
                if r.circuit_breaker.status == "open":
                    continue  # circuit breaker gate
                if not r.capacity.can_accept():
                    continue  # capacity gate
                if r.completed_work_orders < best_completions:
                    best = r
                    best_completions = r.completed_work_orders

            if best and best.resource_id in res_idx:
                j = res_idx[best.resource_id]
                wo_idx = next(
                    (i for i, w in enumerate(work_orders) if w.work_order_id == wo.work_order_id),
                    None,
                )
                if wo_idx is not None and wo_idx < len(cost_matrix) and j < len(cost_matrix[wo_idx]):
                    score = cost_matrix[wo_idx][j]
                    assignment_map[wo.work_order_id] = Assignment(
                        work_order_id=wo.work_order_id,
                        resource_id=best.resource_id,
                        score=score,
                        feature_scores={
                            "exploration": 1.0,
                            "epsilon": epsilon,
                            "resource_completions": float(best.completed_work_orders),
                        },
                    )
                    explore_count += 1
                    log.info(
                        "Exploration: %s → %s (completions=%d, epsilon=%.4f)",
                        wo.work_order_id, best.resource_id,
                        best.completed_work_orders, epsilon,
                    )

        return assignment_map

    # ─── Utilities ───────────────────────────────────────────────

    @staticmethod
    def _compute_seed(
        work_orders: list[DDDWorkOrder],
        resources: list[ResourceRegistration],
    ) -> int:
        """
        Deterministic seed from input hash (Invariant 9: Solver Determinism).
        """
        h = hashlib.sha256()
        for wo in work_orders:
            h.update(wo.work_order_id.encode())
        for res in resources:
            h.update(res.resource_id.encode())
        return int(h.hexdigest()[:8], 16)
