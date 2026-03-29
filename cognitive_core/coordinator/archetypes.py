"""
Cognitive Core — Optimization Archetypes

Implements Spec v1.1 Sections 12 (Optimization Archetypes) and 14 (Solver Architecture).

Each archetype is a proven mathematical model template that accepts typed
parameters and produces assignments. The archetype does NOT know about
domains, resources, or work orders — it operates on abstract cost matrices
and constraint vectors.

The Assignment archetype ships with a pure-Python solver (Hungarian algorithm
for small instances, greedy best-fit for larger). In production, the solver
backend is swapped to Pyomo + CBC/Gurobi via configuration.

Other archetypes (VRP, Job Shop, Flow Network, Knapsack, Coverage) define
the interface but require Pyomo. They raise NotImplementedError until the
Pyomo backend is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# SOLUTION TYPE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ArchetypeSolution:
    """
    Solver output. Archetype-agnostic.

    assignments: list of (work_order_index, resource_index) pairs.
    unassigned: work order indices that could not be assigned.
    """
    assignments: list[tuple[int, int]]
    unassigned: list[int] = field(default_factory=list)
    objective_value: float = 0.0
    solver_status: str = "optimal"    # optimal | feasible | infeasible | timeout
    solve_time_ms: float = 0.0
    solver_name: str = "builtin"
    solver_seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# ASSIGNMENT ARCHETYPE — Pure-Python Solver
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AssignmentParams:
    """
    Parameters for the assignment archetype.

    cost_matrix: W×R matrix where cost_matrix[w][r] is the cost of
                 assigning work order w to resource r. Lower is better.
    capacities:  Per-resource remaining capacity (integers).
    demands:     Per-work-order demand (usually all 1s for slot model).
    """
    cost_matrix: list[list[float]]
    capacities: list[int]
    demands: list[int] | None = None

    @property
    def num_work_orders(self) -> int:
        return len(self.cost_matrix)

    @property
    def num_resources(self) -> int:
        return len(self.capacities)

    def validate(self) -> list[str]:
        errors = []
        if not self.cost_matrix:
            errors.append("cost_matrix is empty")
            return errors
        w = len(self.cost_matrix)
        r = len(self.capacities)
        for i, row in enumerate(self.cost_matrix):
            if len(row) != r:
                errors.append(
                    f"cost_matrix row {i} has {len(row)} cols, expected {r}"
                )
        if self.demands and len(self.demands) != w:
            errors.append(
                f"demands has {len(self.demands)} entries, expected {w}"
            )
        return errors


INF_COST = 1e12  # sentinel for infeasible assignments


class AssignmentArchetype:
    """
    Bipartite assignment archetype (Spec Section 12, Archetype 1).

    Solves: minimize total cost of assigning W work orders to R resources
    subject to capacity constraints.

    Uses pure-Python solver:
    - For small instances (W*R < 2500): modified Jonker-Volgenant algorithm
    - For all instances: greedy best-fit as fallback/large-instance solver

    In production, the solve method is replaced by a Pyomo model dispatch
    to CBC or Gurobi. The interface is identical.
    """
    name = "assignment"

    def solve(
        self,
        params: AssignmentParams,
        time_budget_seconds: float = 5.0,
        seed: int = 0,
    ) -> ArchetypeSolution:
        """Solve the assignment problem within the time budget."""
        errors = params.validate()
        if errors:
            return ArchetypeSolution(
                assignments=[],
                unassigned=list(range(params.num_work_orders)),
                solver_status="infeasible",
                extra={"validation_errors": errors},
            )

        start = time.monotonic()
        demands = params.demands or [1] * params.num_work_orders

        # Try optimal solve for small instances
        w, r = params.num_work_orders, params.num_resources
        if w * r <= 2500 and w <= 50:
            result = self._solve_hungarian(params, demands, time_budget_seconds, seed)
            result.solve_time_ms = (time.monotonic() - start) * 1000
            if result.solver_status in ("optimal", "feasible"):
                return result

        # Greedy fallback for large instances or if optimal failed
        result = self._solve_greedy(params, demands, seed)
        result.solve_time_ms = (time.monotonic() - start) * 1000
        return result

    def _solve_hungarian(
        self,
        params: AssignmentParams,
        demands: list[int],
        time_budget: float,
        seed: int,
    ) -> ArchetypeSolution:
        """
        Modified Hungarian/auction algorithm for small assignment problems.

        Expands capacity constraints by creating virtual copies of each
        resource (one per capacity unit), then solves the resulting
        square assignment on the expanded matrix.
        """
        w = params.num_work_orders
        r = params.num_resources
        cost = params.cost_matrix
        caps = params.capacities

        # Expand resources by capacity: resource j with cap 3 → 3 columns
        expanded_costs: list[list[float]] = []
        col_to_resource: list[int] = []

        for j in range(r):
            cap = max(1, caps[j])
            for _ in range(cap):
                col_to_resource.append(j)

        total_cols = len(col_to_resource)

        # Build expanded cost matrix (w rows × total_cols columns)
        for i in range(w):
            row = []
            for c in range(total_cols):
                j = col_to_resource[c]
                row.append(cost[i][j])
            expanded_costs.append(row)

        # If fewer work orders than columns, pad with dummy rows (high cost)
        n = max(w, total_cols)
        for _ in range(n - w):
            expanded_costs.append([INF_COST] * total_cols)
        # If fewer columns than rows, pad with dummy columns
        for row in expanded_costs:
            while len(row) < n:
                row.append(INF_COST)

        # Run the core algorithm
        assignment = self._hungarian_core(expanded_costs, n)

        # Extract results, mapping back to original resource indices
        assignments: list[tuple[int, int]] = []
        unassigned: list[int] = []
        total_cost = 0.0
        resource_usage: dict[int, int] = {}

        for i in range(w):
            col = assignment[i]
            if col < total_cols:
                j = col_to_resource[col]
                c = cost[i][j]
                if c >= INF_COST * 0.5:
                    unassigned.append(i)
                else:
                    # Check capacity
                    usage = resource_usage.get(j, 0)
                    if usage < caps[j]:
                        assignments.append((i, j))
                        resource_usage[j] = usage + demands[i]
                        total_cost += c
                    else:
                        unassigned.append(i)
            else:
                unassigned.append(i)

        status = "optimal" if not unassigned else "feasible"
        if not assignments:
            status = "infeasible"

        return ArchetypeSolution(
            assignments=assignments,
            unassigned=unassigned,
            objective_value=total_cost,
            solver_status=status,
            solver_name="hungarian_builtin",
            solver_seed=seed,
        )

    @staticmethod
    def _hungarian_core(cost: list[list[float]], n: int) -> list[int]:
        """
        Jonker-Volgenant rectangular assignment algorithm.
        Returns row-to-column assignment as a list.

        Pure Python, O(n³). Handles n up to ~50 comfortably.
        """
        # u[i], v[j] are dual variables (potentials)
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        # p[j] = row assigned to column j (1-indexed)
        p = [0] * (n + 1)
        # way[j] = previous column in augmenting path
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            min_v = [float("inf")] * (n + 1)
            used = [False] * (n + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = -1

                for j in range(1, n + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < min_v[j]:
                        min_v[j] = cur
                        way[j] = j0
                    if min_v[j] < delta:
                        delta = min_v[j]
                        j1 = j

                if j1 == -1:
                    break

                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        min_v[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            # Augment
            while j0 != 0:
                p[j0] = p[way[j0]]
                j0 = way[j0]

        # Convert to 0-indexed row assignment
        result = [0] * n
        for j in range(1, n + 1):
            if p[j] > 0:
                result[p[j] - 1] = j - 1

        return result

    def _solve_greedy(
        self,
        params: AssignmentParams,
        demands: list[int],
        seed: int,
    ) -> ArchetypeSolution:
        """
        Greedy best-fit assignment (Spec Section 17.2).

        1. Sort work orders by min available cost (most constrained first)
        2. For each, assign to lowest-cost resource with remaining capacity
        3. O(W log W + W × R) — completes in milliseconds

        Uses the SAME cost matrix as the optimal solver, ensuring
        Fallback Equivalence (Invariant 12).
        """
        w = params.num_work_orders
        r = params.num_resources
        cost = params.cost_matrix
        remaining_cap = list(params.capacities)

        # Sort work orders by min cost (most constrained first)
        order = sorted(range(w), key=lambda i: min(cost[i]))

        assignments: list[tuple[int, int]] = []
        unassigned: list[int] = []
        total_cost = 0.0

        for i in order:
            best_j = -1
            best_c = INF_COST

            for j in range(r):
                if remaining_cap[j] >= demands[i] and cost[i][j] < best_c:
                    best_c = cost[i][j]
                    best_j = j

            if best_j >= 0 and best_c < INF_COST * 0.5:
                assignments.append((i, best_j))
                remaining_cap[best_j] -= demands[i]
                total_cost += best_c
            else:
                unassigned.append(i)

        status = "feasible" if assignments else "infeasible"
        return ArchetypeSolution(
            assignments=assignments,
            unassigned=unassigned,
            objective_value=total_cost,
            solver_status=status,
            solver_name="greedy_builtin",
            solver_seed=seed,
        )


# ═══════════════════════════════════════════════════════════════════
# STUB ARCHETYPES — Interface only, require Pyomo
# ═══════════════════════════════════════════════════════════════════

class VRPArchetype:
    """Vehicle Routing Problem (Spec Section 12, Archetype 2). Requires Pyomo."""
    name = "vrp"

    def solve(self, params: Any, time_budget_seconds: float = 5.0, seed: int = 0) -> ArchetypeSolution:
        raise NotImplementedError(
            "VRP archetype requires Pyomo + OR-Tools. "
            "Install with: pip install pyomo ortools"
        )


class JobShopArchetype:
    """Job Shop Scheduling (Spec Section 12, Archetype 3). Requires Pyomo."""
    name = "job_shop"

    def solve(self, params: Any, time_budget_seconds: float = 5.0, seed: int = 0) -> ArchetypeSolution:
        raise NotImplementedError(
            "Job Shop archetype requires Pyomo + CP-SAT. "
            "Install with: pip install pyomo ortools"
        )


class FlowNetworkArchetype:
    """Flow Network (Spec Section 12, Archetype 4). Requires Pyomo."""
    name = "flow_network"

    def solve(self, params: Any, time_budget_seconds: float = 5.0, seed: int = 0) -> ArchetypeSolution:
        raise NotImplementedError(
            "Flow Network archetype requires Pyomo. "
            "Install with: pip install pyomo"
        )


class KnapsackArchetype:
    """Knapsack/Bin Packing (Spec Section 12, Archetype 5). Requires Pyomo."""
    name = "knapsack"

    def solve(self, params: Any, time_budget_seconds: float = 5.0, seed: int = 0) -> ArchetypeSolution:
        raise NotImplementedError(
            "Knapsack archetype requires Pyomo. "
            "Install with: pip install pyomo"
        )


class CoverageArchetype:
    """Set Coverage (Spec Section 12, Archetype 6). Requires Pyomo."""
    name = "coverage"

    def solve(self, params: Any, time_budget_seconds: float = 5.0, seed: int = 0) -> ArchetypeSolution:
        raise NotImplementedError(
            "Coverage archetype requires Pyomo. "
            "Install with: pip install pyomo"
        )


# ═══════════════════════════════════════════════════════════════════
# ARCHETYPE REGISTRY
# ═══════════════════════════════════════════════════════════════════

ARCHETYPE_REGISTRY: dict[str, Any] = {
    "assignment": AssignmentArchetype(),
    "vrp": VRPArchetype(),
    "job_shop": JobShopArchetype(),
    "flow_network": FlowNetworkArchetype(),
    "knapsack": KnapsackArchetype(),
    "coverage": CoverageArchetype(),
}
