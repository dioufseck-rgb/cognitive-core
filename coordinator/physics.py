"""
Cognitive Core — Domain Physics

Implements Spec v1.1 Section 13 (Domain Physics) and Section 11 (Boundary Rule).

Domain physics classes are the bridge between real-world domain concepts
and mathematical optimization. Each physics class:

1. extract_parameters(): Transforms work orders + resources + config into
   archetype-specific parameter objects (cost matrices, constraint vectors).

2. interpret_solution(): Transforms archetype solutions back into domain
   assignments (work_order_id → resource_id with scores).

The physics class is the ONLY place where domain knowledge enters the
optimization pipeline. The archetype solver is domain-agnostic.

Per the Boundary Rule (Section 11):
  - Physics classes live in Python code, not YAML
  - YAML carries objective weights and governance thresholds only
  - The physics class reads YAML weights but contains the cost functions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from coordinator.archetypes import AssignmentParams, ArchetypeSolution, INF_COST
from coordinator.ddd import DDDWorkOrder, ResourceRegistration


# ═══════════════════════════════════════════════════════════════════
# DOMAIN TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OptimizationConfig:
    """
    Parsed from domain YAML `optimization:` section.
    These are tunable weights and thresholds — NOT code.
    """
    archetype: str = "assignment"
    physics: str = "default"
    objectives: dict[str, float] = field(default_factory=lambda: {
        "minimize_cost": 0.3,
        "minimize_wait_time": 0.5,
        "minimize_sla_risk": 0.2,
    })
    constraints: dict[str, Any] = field(default_factory=dict)
    solver_time_budget_seconds: float = 5.0
    greedy_fallback: bool = True
    # Exploration
    exploration_enabled: bool = True
    maturity_threshold: int = 10
    base_epsilon: float = 0.10
    novelty_bonus: float = 0.05
    max_exploration_pct: float = 0.15
    exploration_sla_gate: list[str] = field(
        default_factory=lambda: ["routine", "high"]
    )
    # Adaptive bounds (Section 15)
    cfa_bounds: dict[str, float] = field(default_factory=lambda: {
        "max_cost_adjustment": 0.30,
        "max_wait_adjustment": 0.25,
        "max_sla_adjustment": 0.20,
    })


@dataclass
class Assignment:
    """A resolved assignment: work order → resource with scoring detail."""
    work_order_id: str
    resource_id: str
    score: float
    feature_scores: dict[str, float] = field(default_factory=dict)


def parse_optimization_config(yaml_section: dict[str, Any] | None) -> OptimizationConfig:
    """Parse the `optimization:` section from domain YAML."""
    if not yaml_section:
        return OptimizationConfig()

    exploration = yaml_section.get("exploration", {})
    cfa = yaml_section.get("cfa_bounds", {})

    # Parse solver_time_budget with optional unit suffix
    budget_raw = yaml_section.get("solver_time_budget", "5s")
    if isinstance(budget_raw, str):
        budget_raw = budget_raw.rstrip("s")
    budget = float(budget_raw)

    return OptimizationConfig(
        archetype=yaml_section.get("archetype", "assignment"),
        physics=yaml_section.get("physics", "default"),
        objectives=yaml_section.get("objectives", {
            "minimize_cost": 0.3,
            "minimize_wait_time": 0.5,
            "minimize_sla_risk": 0.2,
        }),
        constraints=yaml_section.get("constraints", {}),
        solver_time_budget_seconds=budget,
        greedy_fallback=yaml_section.get("greedy_fallback", True),
        exploration_enabled=exploration.get("enabled", True),
        maturity_threshold=exploration.get("maturity_threshold", 10),
        base_epsilon=exploration.get("base_epsilon", 0.10),
        novelty_bonus=exploration.get("novelty_bonus", 0.05),
        max_exploration_pct=exploration.get("max_exploration_pct", 0.15),
        exploration_sla_gate=exploration.get(
            "sla_gate", ["routine", "high"]
        ),
        cfa_bounds=cfa or {
            "max_cost_adjustment": 0.30,
            "max_wait_adjustment": 0.25,
            "max_sla_adjustment": 0.20,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# PHYSICS INTERFACE
# ═══════════════════════════════════════════════════════════════════

class DefaultAssignmentPhysics:
    """
    Generic assignment physics (Spec Section 13).

    Computes cost matrix from resource attributes weighted by
    domain YAML objective weights. Works for any domain that uses
    the assignment archetype. Domain-specific physics override this
    when they need custom cost functions (e.g., haversine for geography).

    Cost formula per (work_order, resource) pair:
      cost = w_cost * normalized_cost_rate
           + w_wait * normalized_load_pct
           + w_sla  * sla_risk_score
           - w_quality * normalized_quality_score
           - novelty_bonus (if resource is unproven)
    """
    archetype = "assignment"

    def extract_parameters(
        self,
        work_orders: list[DDDWorkOrder],
        resources: list[ResourceRegistration],
        config: OptimizationConfig,
    ) -> AssignmentParams:
        """
        Transform domain reality into a cost matrix.

        Returns AssignmentParams ready for the Assignment archetype solver.
        """
        w = len(work_orders)
        r = len(resources)

        if w == 0 or r == 0:
            return AssignmentParams(
                cost_matrix=[],
                capacities=[],
            )

        # Objective weights (normalized to sum to 1)
        obj = config.objectives
        total_w = sum(obj.values()) or 1.0
        w_cost = obj.get("minimize_cost", 0.3) / total_w
        w_wait = obj.get("minimize_wait_time", 0.5) / total_w
        w_sla = obj.get("minimize_sla_risk", 0.2) / total_w
        w_quality = obj.get("maximize_quality", 0.0) / total_w

        # Normalization ranges (avoid division by zero)
        cost_rates = [res.attributes.get("cost_rate", 50.0) for res in resources]
        max_cost = max(cost_rates) or 1.0

        quality_scores = [res.attributes.get("quality_score", 0.5) for res in resources]
        max_quality = max(quality_scores) or 1.0

        # Build cost matrix
        cost_matrix: list[list[float]] = []
        for i, wo in enumerate(work_orders):
            row: list[float] = []
            for j, res in enumerate(resources):
                cost = self._compute_cell_cost(
                    wo, res, config,
                    w_cost, w_wait, w_sla, w_quality,
                    max_cost, max_quality,
                )
                row.append(cost)
            cost_matrix.append(row)

        # Capacities: remaining capacity per resource
        capacities = []
        for res in resources:
            cap = res.capacity
            if cap.model.value == "slot":
                remaining = max(0, cap.max_concurrent - cap.current_load)
            elif cap.model.value == "volume":
                remaining = max(0, int(cap.max_volume - cap.current_volume))
            elif cap.model.value == "batch":
                remaining = max(0, cap.batch_threshold - cap.batch_items) if cap.batch_status.value == "collecting" else 0
            else:
                remaining = 1
            capacities.append(remaining)

        return AssignmentParams(
            cost_matrix=cost_matrix,
            capacities=capacities,
        )

    def _compute_cell_cost(
        self,
        wo: DDDWorkOrder,
        res: ResourceRegistration,
        config: OptimizationConfig,
        w_cost: float,
        w_wait: float,
        w_sla: float,
        w_quality: float,
        max_cost: float,
        max_quality: float,
    ) -> float:
        """Compute the cost of assigning work_order to resource."""
        # Cost rate (normalized 0–1, lower is better → direct)
        cost_rate = res.attributes.get("cost_rate", 50.0) / max_cost

        # Load percentage (higher load → higher cost)
        load_pct = res.capacity.utilization_pct

        # SLA risk: how close is the work order to breaching SLA?
        sla_risk = self._compute_sla_risk(wo, res)

        # Quality score (inverted: higher quality → lower cost)
        quality = res.attributes.get("quality_score", 0.5) / max_quality

        # Novelty bonus for unproven resources (Section 17.4)
        novelty = 0.0
        if res.completed_work_orders < config.maturity_threshold:
            novelty = config.novelty_bonus

        total = (
            w_cost * cost_rate
            + w_wait * load_pct
            + w_sla * sla_risk
            - w_quality * quality
            - novelty
        )

        # Clamp to positive (costs can't be negative in assignment)
        return max(0.001, total)

    def _compute_sla_risk(
        self,
        wo: DDDWorkOrder,
        res: ResourceRegistration,
    ) -> float:
        """
        SLA risk score (0 = safe, 1 = critical).

        Based on the ratio of resource's average turnaround to the
        work order's remaining SLA time.
        """
        if not wo.sla_seconds:
            return 0.0

        avg_turnaround = res.attributes.get("avg_turnaround_seconds", 3600.0)
        elapsed = 0.0
        if wo.created_at:
            import time as _time
            elapsed = _time.time() - wo.created_at

        remaining = wo.sla_seconds - elapsed
        if remaining <= 0:
            return 1.0  # already breached

        ratio = avg_turnaround / remaining
        return min(1.0, max(0.0, ratio))

    def interpret_solution(
        self,
        solution: ArchetypeSolution,
        work_orders: list[DDDWorkOrder],
        resources: list[ResourceRegistration],
        cost_matrix: list[list[float]] | None = None,
    ) -> list[Assignment]:
        """
        Map archetype solution back to domain assignments.
        """
        assignments: list[Assignment] = []

        for wo_idx, res_idx in solution.assignments:
            wo = work_orders[wo_idx]
            res = resources[res_idx]
            cost = 0.0
            if cost_matrix and wo_idx < len(cost_matrix) and res_idx < len(cost_matrix[wo_idx]):
                cost = cost_matrix[wo_idx][res_idx]

            assignments.append(Assignment(
                work_order_id=wo.work_order_id,
                resource_id=res.resource_id,
                score=cost,
                feature_scores={
                    "cost_rate": res.attributes.get("cost_rate", 0.0),
                    "load_pct": res.capacity.utilization_pct,
                    "quality_score": res.attributes.get("quality_score", 0.5),
                    "completed_work_orders": float(res.completed_work_orders),
                },
            ))

        return assignments


class ClaimsAssignmentPhysics(DefaultAssignmentPhysics):
    """
    Insurance claims physics — extends default with geographic distance.

    Adds haversine distance between claim location and adjuster location
    as a cost factor weighted by the `minimize_travel` objective.
    """

    def _compute_cell_cost(
        self,
        wo: DDDWorkOrder,
        res: ResourceRegistration,
        config: OptimizationConfig,
        w_cost: float,
        w_wait: float,
        w_sla: float,
        w_quality: float,
        max_cost: float,
        max_quality: float,
    ) -> float:
        base_cost = super()._compute_cell_cost(
            wo, res, config, w_cost, w_wait, w_sla, w_quality,
            max_cost, max_quality,
        )

        # Add geographic distance if available
        obj = config.objectives
        total_w = sum(obj.values()) or 1.0
        w_travel = obj.get("minimize_travel", 0.0) / total_w

        if w_travel > 0:
            wo_lat = wo.inputs.get("latitude")
            wo_lon = wo.inputs.get("longitude")
            res_lat = res.attributes.get("latitude")
            res_lon = res.attributes.get("longitude")

            if all(v is not None for v in [wo_lat, wo_lon, res_lat, res_lon]):
                dist = _haversine(wo_lat, wo_lon, res_lat, res_lon)
                # Normalize: 100 miles = 1.0 cost unit
                base_cost += w_travel * (dist / 100.0)

        return max(0.001, base_cost)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in miles between two lat/lon points."""
    R = 3959.0  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ═══════════════════════════════════════════════════════════════════
# PHYSICS REGISTRY
# ═══════════════════════════════════════════════════════════════════

PHYSICS_REGISTRY: dict[str, type] = {
    "default": DefaultAssignmentPhysics,
    "claims_assignment": ClaimsAssignmentPhysics,
}
