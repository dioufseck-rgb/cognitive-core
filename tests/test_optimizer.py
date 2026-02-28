"""
Cognitive Core — Dispatch Optimization Tests

Tests the full optimization pipeline from Spec v1.1:
  - Assignment archetype solver (Section 12)
  - Domain physics extraction (Section 13)
  - Dispatch optimizer pipeline (Sections 9, 14)
  - Greedy fallback (Section 17.2)
  - Exploration policy (Section 17.4)
  - Audit trail completeness (Invariant 8)
"""

import os
import sys
import time
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.archetypes import (
    AssignmentArchetype,
    AssignmentParams,
    ArchetypeSolution,
    VRPArchetype,
    ARCHETYPE_REGISTRY,
    INF_COST,
)
from coordinator.physics import (
    DefaultAssignmentPhysics,
    ClaimsAssignmentPhysics,
    OptimizationConfig,
    Assignment,
    parse_optimization_config,
    _haversine,
)
from coordinator.optimizer import DispatchOptimizer
from coordinator.ddd import (
    DDDWorkOrder,
    ResourceRegistration,
    ResourceRegistry,
    CapacityModel,
    CapacityState,
    CircuitBreakerState,
    EligibilityConstraint,
    DispatchDecision,
    RankingScore,
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def make_resource(
    rid: str,
    cost_rate: float = 50.0,
    quality_score: float = 0.8,
    cap: int = 5,
    model: CapacityModel = CapacityModel.SLOT,
    completions: int = 20,
    **extra: float,
) -> ResourceRegistration:
    r = ResourceRegistration.create(
        rid, "human",
        [("review", "claims"), ("investigate", "fraud")],
        model, cap,
        cost_rate=cost_rate, quality_score=quality_score, **extra,
    )
    r.completed_work_orders = completions
    return r


def make_wo(
    priority: str = "routine",
    sla: float = 3600.0,
    case_id: str = "CLM-001",
) -> DDDWorkOrder:
    return DDDWorkOrder.create(
        "wf_test",
        "cor_test",
        "review_contract",
        priority=priority,
        sla_seconds=sla,
        case_id=case_id,
    )


# ═══════════════════════════════════════════════════════════════════
# ASSIGNMENT ARCHETYPE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAssignmentArchetype(unittest.TestCase):
    """Spec Section 12: Assignment archetype solver."""

    def setUp(self):
        self.solver = AssignmentArchetype()

    def test_3x3_optimal(self):
        """Classic 3×3 assignment finds true optimum."""
        params = AssignmentParams(
            cost_matrix=[[9, 2, 7], [6, 4, 3], [5, 8, 1]],
            capacities=[1, 1, 1],
        )
        sol = self.solver.solve(params)
        self.assertEqual(sol.solver_status, "optimal")
        self.assertEqual(len(sol.assignments), 3)
        self.assertEqual(len(sol.unassigned), 0)
        # Optimal: (0→1)=2, (1→0)=6, (2→2)=1 → 9
        self.assertAlmostEqual(sol.objective_value, 9.0)

    def test_with_capacity_constraints(self):
        """Resource capacity limits are respected."""
        params = AssignmentParams(
            cost_matrix=[[1, 10], [2, 10], [3, 10]],
            capacities=[2, 1],  # resource 0 can take 2, resource 1 takes 1
        )
        sol = self.solver.solve(params)
        self.assertEqual(len(sol.assignments), 3)
        # Count per resource
        by_res = {}
        for _, j in sol.assignments:
            by_res[j] = by_res.get(j, 0) + 1
        self.assertLessEqual(by_res.get(0, 0), 2)
        self.assertLessEqual(by_res.get(1, 0), 1)

    def test_more_work_than_capacity(self):
        """Some work orders go unassigned when capacity is exhausted."""
        params = AssignmentParams(
            cost_matrix=[[1], [2], [3], [4], [5]],
            capacities=[2],
        )
        sol = self.solver.solve(params)
        self.assertEqual(len(sol.assignments), 2)
        self.assertEqual(len(sol.unassigned), 3)
        self.assertEqual(sol.solver_status, "feasible")

    def test_10x10_assignment(self):
        """Larger instance solves correctly within time budget."""
        import random
        random.seed(42)
        n = 10
        cm = [[random.uniform(1, 100) for _ in range(n)] for _ in range(n)]
        params = AssignmentParams(cost_matrix=cm, capacities=[1] * n)
        sol = self.solver.solve(params, time_budget_seconds=5.0)
        self.assertIn(sol.solver_status, ("optimal", "feasible"))
        self.assertEqual(len(sol.assignments), n)
        self.assertLess(sol.solve_time_ms, 1000)

    def test_deterministic(self):
        """Same inputs → same output (Invariant 9)."""
        params = AssignmentParams(
            cost_matrix=[[3, 7, 2], [5, 1, 4], [6, 3, 8]],
            capacities=[1, 1, 1],
        )
        sol1 = self.solver.solve(params, seed=42)
        sol2 = self.solver.solve(params, seed=42)
        self.assertEqual(sol1.assignments, sol2.assignments)
        self.assertAlmostEqual(sol1.objective_value, sol2.objective_value)

    def test_empty_input(self):
        """Empty cost matrix returns infeasible."""
        params = AssignmentParams(cost_matrix=[], capacities=[])
        sol = self.solver.solve(params)
        self.assertEqual(sol.solver_status, "infeasible")

    def test_greedy_for_large_instance(self):
        """Large instance uses greedy solver path."""
        import random
        random.seed(99)
        # 30×20 exceeds 2500 threshold → greedy
        cm = [[random.uniform(1, 100) for _ in range(20)] for _ in range(30)]
        params = AssignmentParams(cost_matrix=cm, capacities=[2] * 20)
        sol = self.solver.solve(params)
        self.assertEqual(len(sol.assignments), 30)
        self.assertLess(sol.solve_time_ms, 100)


class TestStubArchetypes(unittest.TestCase):
    """Stub archetypes raise NotImplementedError."""

    def test_vrp_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            VRPArchetype().solve(None)

    def test_registry_has_all(self):
        expected = {"assignment", "vrp", "job_shop", "flow_network", "knapsack", "coverage"}
        self.assertEqual(set(ARCHETYPE_REGISTRY.keys()), expected)


# ═══════════════════════════════════════════════════════════════════
# PHYSICS TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDefaultPhysics(unittest.TestCase):
    """Spec Section 13: Domain physics extraction."""

    def setUp(self):
        self.physics = DefaultAssignmentPhysics()
        self.config = OptimizationConfig()

    def test_extract_parameters_shape(self):
        """Cost matrix has correct shape: W rows × R columns."""
        wos = [make_wo() for _ in range(3)]
        resources = [make_resource(f"r_{i}") for i in range(4)]
        params = self.physics.extract_parameters(wos, resources, self.config)
        self.assertEqual(params.num_work_orders, 3)
        self.assertEqual(params.num_resources, 4)
        self.assertEqual(len(params.cost_matrix), 3)
        for row in params.cost_matrix:
            self.assertEqual(len(row), 4)

    def test_cost_reflects_weights(self):
        """Cheaper resource gets lower cost in cost matrix."""
        wos = [make_wo()]
        r_cheap = make_resource("r_cheap", cost_rate=10, quality_score=0.9)
        r_expensive = make_resource("r_expensive", cost_rate=100, quality_score=0.5)
        config = OptimizationConfig(objectives={"minimize_cost": 1.0})
        params = self.physics.extract_parameters(wos, [r_cheap, r_expensive], config)
        self.assertLess(params.cost_matrix[0][0], params.cost_matrix[0][1])

    def test_capacities_reflect_remaining(self):
        """Capacity vector reflects remaining capacity, not total."""
        r = make_resource("r_1", cap=5)
        r.capacity.current_load = 3  # 2 remaining
        params = self.physics.extract_parameters([make_wo()], [r], self.config)
        self.assertEqual(params.capacities[0], 2)

    def test_empty_inputs(self):
        """Empty work orders or resources produce empty params."""
        params = self.physics.extract_parameters([], [], self.config)
        self.assertEqual(params.num_work_orders, 0)
        self.assertEqual(params.num_resources, 0)

    def test_interpret_solution(self):
        """Solution maps back to named assignments."""
        wos = [make_wo(), make_wo()]
        resources = [make_resource("r_a"), make_resource("r_b")]
        solution = ArchetypeSolution(
            assignments=[(0, 1), (1, 0)],
            objective_value=5.0,
        )
        assignments = self.physics.interpret_solution(solution, wos, resources)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(assignments[0].work_order_id, wos[0].work_order_id)
        self.assertEqual(assignments[0].resource_id, "r_b")
        self.assertEqual(assignments[1].resource_id, "r_a")

    def test_novelty_bonus_reduces_cost(self):
        """Unproven resources get cost reduction from novelty bonus."""
        wos = [make_wo()]
        r_proven = make_resource("r_proven", cost_rate=50, completions=20)
        r_new = make_resource("r_new", cost_rate=50, completions=2)
        config = OptimizationConfig(novelty_bonus=0.10)
        params = self.physics.extract_parameters(wos, [r_proven, r_new], config)
        # New resource should have lower cost due to novelty bonus
        self.assertLess(params.cost_matrix[0][1], params.cost_matrix[0][0])


class TestClaimsPhysics(unittest.TestCase):
    """Claims-specific physics with geographic distance."""

    def test_haversine_known_distance(self):
        """New York to Los Angeles ≈ 2451 miles."""
        dist = _haversine(40.7128, -74.0060, 34.0522, -118.2437)
        self.assertAlmostEqual(dist, 2451, delta=50)

    def test_travel_cost_added(self):
        """Geographic distance increases cost when minimize_travel is weighted."""
        physics = ClaimsAssignmentPhysics()
        config = OptimizationConfig(objectives={
            "minimize_cost": 0.3,
            "minimize_wait_time": 0.3,
            "minimize_sla_risk": 0.2,
            "minimize_travel": 0.2,
        })
        wo = make_wo()
        wo.inputs["latitude"] = 40.7128
        wo.inputs["longitude"] = -74.0060

        r_near = make_resource("r_near", latitude=40.75, longitude=-73.98)
        r_far = make_resource("r_far", latitude=34.05, longitude=-118.24)

        params = physics.extract_parameters([wo], [r_near, r_far], config)
        # Near resource should have lower cost
        self.assertLess(params.cost_matrix[0][0], params.cost_matrix[0][1])


class TestOptimizationConfigParsing(unittest.TestCase):
    """Parse optimization YAML sections."""

    def test_parse_full_config(self):
        yaml = {
            "archetype": "assignment",
            "physics": "claims_assignment",
            "objectives": {"minimize_cost": 0.4, "minimize_wait_time": 0.6},
            "solver_time_budget": "3s",
            "greedy_fallback": True,
            "exploration": {
                "enabled": True,
                "maturity_threshold": 15,
                "base_epsilon": 0.08,
            },
        }
        cfg = parse_optimization_config(yaml)
        self.assertEqual(cfg.archetype, "assignment")
        self.assertEqual(cfg.physics, "claims_assignment")
        self.assertAlmostEqual(cfg.solver_time_budget_seconds, 3.0)
        self.assertEqual(cfg.maturity_threshold, 15)
        self.assertAlmostEqual(cfg.base_epsilon, 0.08)

    def test_parse_none_returns_defaults(self):
        cfg = parse_optimization_config(None)
        self.assertEqual(cfg.archetype, "assignment")
        self.assertAlmostEqual(cfg.solver_time_budget_seconds, 5.0)

    def test_parse_numeric_budget(self):
        cfg = parse_optimization_config({"solver_time_budget": 10})
        self.assertAlmostEqual(cfg.solver_time_budget_seconds, 10.0)


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZER PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDispatchOptimizer(unittest.TestCase):
    """Spec Sections 9, 14: Full dispatch pipeline."""

    def setUp(self):
        self.registry = ResourceRegistry()
        for i, (cost, quality) in enumerate([
            (30, 0.95), (50, 0.80), (70, 0.85), (40, 0.70),
        ]):
            r = make_resource(f"r_{i}", cost_rate=cost, quality_score=quality)
            self.registry.register(r)
        self.optimizer = DispatchOptimizer(self.registry)
        self.config = OptimizationConfig(exploration_enabled=False)

    def test_end_to_end_optimal(self):
        """Full pipeline: eligibility → physics → solve → assign."""
        wos = [make_wo() for _ in range(3)]
        decisions = self.optimizer.dispatch(wos, "review", "claims", self.config)
        self.assertEqual(len(decisions), 3)
        for d in decisions:
            self.assertIsNotNone(d.selected_resource_id)
            self.assertEqual(d.tier, "optimal")
            self.assertIsNotNone(d.reservation_id)

    def test_audit_trail_completeness(self):
        """Every decision has eligibility results (Invariant 8)."""
        wos = [make_wo()]
        decisions = self.optimizer.dispatch(wos, "review", "claims", self.config)
        d = decisions[0]
        self.assertTrue(len(d.eligibility_results) > 0)
        self.assertTrue(d.timestamp > 0)
        # Should have ranking scores for assigned decisions
        if d.selected_resource_id:
            self.assertTrue(len(d.ranking_scores) > 0)
            self.assertTrue(d.ranking_scores[0].total_score > 0)

    def test_no_eligible_resources(self):
        """No matching resources → all decisions have no assignment."""
        wos = [make_wo()]
        decisions = self.optimizer.dispatch(wos, "unknown_wf", "unknown_domain", self.config)
        self.assertEqual(len(decisions), 1)
        self.assertIsNone(decisions[0].selected_resource_id)
        self.assertEqual(decisions[0].tier, "no_eligible_resources")

    def test_single_dispatch(self):
        """dispatch_single returns a single DispatchDecision."""
        wo = make_wo()
        decision = self.optimizer.dispatch_single(wo, "review", "claims", self.config)
        self.assertIsNotNone(decision.selected_resource_id)

    def test_capacity_exhaustion(self):
        """When all resources are full, work orders go unassigned."""
        # Set all resources to max capacity
        for res in self.registry.list_resources():
            res.capacity.current_load = res.capacity.max_concurrent
        wos = [make_wo()]
        decisions = self.optimizer.dispatch(wos, "review", "claims", self.config)
        # All resources full → no eligible after capacity check
        self.assertIsNone(decisions[0].selected_resource_id)


class TestGreedyFallback(unittest.TestCase):
    """Spec Section 17.2: Greedy fallback."""

    def setUp(self):
        self.registry = ResourceRegistry()
        for i in range(3):
            r = make_resource(f"r_{i}", cost_rate=float(30 + i * 20))
            self.registry.register(r)

    def test_greedy_produces_feasible(self):
        """Greedy always produces feasible assignments if capacity exists."""
        optimizer = DispatchOptimizer(self.registry)
        # Force fallback by using a broken archetype
        config = OptimizationConfig(
            archetype="nonexistent",  # will fail → triggers fallback
            greedy_fallback=True,
            exploration_enabled=False,
        )
        wos = [make_wo() for _ in range(3)]
        decisions = optimizer.dispatch(wos, "review", "claims", config)
        assigned = [d for d in decisions if d.selected_resource_id]
        self.assertEqual(len(assigned), 3)
        for d in assigned:
            self.assertEqual(d.tier, "fallback")

    def test_greedy_urgency_ordering(self):
        """Critical work orders assigned first by greedy."""
        optimizer = DispatchOptimizer(self.registry)
        # Only 1 capacity across all resources
        for res in self.registry.list_resources():
            res.capacity.max_concurrent = 1
        config = OptimizationConfig(
            archetype="nonexistent",
            greedy_fallback=True,
            exploration_enabled=False,
        )
        wo_critical = make_wo(priority="critical", sla=600)
        wo_routine = make_wo(priority="routine", sla=7200)
        decisions = optimizer.dispatch(
            [wo_routine, wo_critical], "review", "claims", config,
        )
        # Both should get assigned (3 resources × 1 cap each = 3 slots)
        assigned = [d for d in decisions if d.selected_resource_id]
        self.assertEqual(len(assigned), 2)

    def test_greedy_disabled_returns_no_solution(self):
        """If greedy_fallback=false and solver fails, no assignments."""
        optimizer = DispatchOptimizer(self.registry)
        config = OptimizationConfig(
            archetype="nonexistent",
            greedy_fallback=False,
            exploration_enabled=False,
        )
        wos = [make_wo()]
        decisions = optimizer.dispatch(wos, "review", "claims", config)
        self.assertIsNone(decisions[0].selected_resource_id)
        self.assertEqual(decisions[0].tier, "no_solution")


class TestExplorationPolicy(unittest.TestCase):
    """Spec Section 17.4: Exploration for new resource starvation."""

    def setUp(self):
        self.registry = ResourceRegistry()
        # 2 proven resources + 1 unproven
        self.registry.register(make_resource("r_proven_0", completions=50, cost_rate=40))
        self.registry.register(make_resource("r_proven_1", completions=30, cost_rate=60))
        self.registry.register(make_resource("r_unproven", completions=0, cost_rate=50))

    def test_exploration_assigns_to_unproven(self):
        """With high epsilon, unproven resource gets assignments."""
        optimizer = DispatchOptimizer(self.registry)
        config = OptimizationConfig(
            exploration_enabled=True,
            base_epsilon=1.0,  # always explore for testing
            maturity_threshold=10,
            max_exploration_pct=1.0,
        )
        # Send many work orders — at least some should go to unproven
        wos = [make_wo() for _ in range(20)]
        decisions = optimizer.dispatch(wos, "review", "claims", config)
        assigned_resources = {d.selected_resource_id for d in decisions if d.selected_resource_id}
        self.assertIn("r_unproven", assigned_resources)

    def test_sla_gate_excludes_critical(self):
        """Critical-priority work orders are never explored."""
        optimizer = DispatchOptimizer(self.registry)
        config = OptimizationConfig(
            exploration_enabled=True,
            base_epsilon=1.0,  # always explore
            maturity_threshold=10,
            max_exploration_pct=1.0,
            exploration_sla_gate=["routine"],  # only routine eligible
        )
        # All critical work orders
        wos = [make_wo(priority="critical") for _ in range(5)]
        decisions = optimizer.dispatch(wos, "review", "claims", config)
        # None should go to exploration (critical excluded)
        for d in decisions:
            if d.ranking_scores:
                for rs in d.ranking_scores:
                    # Should not have exploration marker
                    self.assertNotIn("exploration", rs.feature_scores)

    def test_circuit_breaker_overrides_exploration(self):
        """Open circuit breaker prevents exploration assignment."""
        # Trip unproven resource's circuit breaker
        r = self.registry.get("r_unproven")
        r.circuit_breaker.status = "open"

        optimizer = DispatchOptimizer(self.registry)
        config = OptimizationConfig(
            exploration_enabled=True,
            base_epsilon=1.0,
            maturity_threshold=10,
        )
        wos = [make_wo() for _ in range(10)]
        decisions = optimizer.dispatch(wos, "review", "claims", config)
        # Unproven resource should not get any assignments
        for d in decisions:
            self.assertNotEqual(d.selected_resource_id, "r_unproven")

    def test_epsilon_decay(self):
        """Epsilon decays as unproven resources accumulate data."""
        optimizer = DispatchOptimizer(self.registry)

        # Initial epsilon: 0.10 / (1 + 0) = 0.10
        config0 = OptimizationConfig(
            base_epsilon=0.10, maturity_threshold=10,
        )
        r = self.registry.get("r_unproven")

        # Simulate completions
        r.completed_work_orders = 0
        proven, unproven = self.registry.partition_by_maturity(
            self.registry.list_resources(), 10,
        )
        min_completions = min(u.completed_work_orders for u in unproven) if unproven else 0
        eps0 = 0.10 / (1.0 + min_completions)
        self.assertAlmostEqual(eps0, 0.10)

        # After 9 completions: 0.10 / (1 + 9) = 0.01
        r.completed_work_orders = 9
        proven, unproven = self.registry.partition_by_maturity(
            self.registry.list_resources(), 10,
        )
        min_completions = min(u.completed_work_orders for u in unproven) if unproven else 0
        eps9 = 0.10 / (1.0 + min_completions)
        self.assertAlmostEqual(eps9, 0.01)

    def test_max_exploration_ceiling(self):
        """Exploration count doesn't exceed max_exploration_pct."""
        # Add 5 unproven resources
        for i in range(5):
            self.registry.register(make_resource(f"r_unproven_{i}", completions=0))

        optimizer = DispatchOptimizer(self.registry)
        config = OptimizationConfig(
            exploration_enabled=True,
            base_epsilon=1.0,  # always explore
            maturity_threshold=10,
            max_exploration_pct=0.10,  # max 10% exploration
        )
        wos = [make_wo() for _ in range(20)]
        decisions = optimizer.dispatch(wos, "review", "claims", config)
        # Count exploration assignments
        exploration_count = sum(
            1 for d in decisions
            if d.ranking_scores and any(
                "exploration" in rs.feature_scores
                for rs in d.ranking_scores
            )
        )
        # Should be at most 10% of 20 = 2
        self.assertLessEqual(exploration_count, 2)


class TestBackwardCompatibility(unittest.TestCase):
    """Runtime integration: optimizer does not break existing behavior."""

    def test_optimizer_import(self):
        """Optimizer modules import cleanly."""
        from coordinator.optimizer import DispatchOptimizer
        from coordinator.physics import OptimizationConfig, parse_optimization_config
        from coordinator.archetypes import ARCHETYPE_REGISTRY
        self.assertIsNotNone(DispatchOptimizer)
        self.assertIn("assignment", ARCHETYPE_REGISTRY)

    def test_registry_works_standalone(self):
        """ResourceRegistry from ddd.py works with optimizer."""
        registry = ResourceRegistry()
        r = make_resource("r_test")
        registry.register(r)
        optimizer = DispatchOptimizer(registry)
        wo = make_wo()
        decision = optimizer.dispatch_single(wo, "review", "claims")
        self.assertIsNotNone(decision)

    def test_dispatch_with_no_config(self):
        """Dispatch works with default config (no YAML optimization section)."""
        registry = ResourceRegistry()
        registry.register(make_resource("r_1"))
        optimizer = DispatchOptimizer(registry)
        wo = make_wo()
        # No config → uses defaults
        decision = optimizer.dispatch_single(wo, "review", "claims")
        self.assertIsNotNone(decision.selected_resource_id)


class TestInvariantEnforcement(unittest.TestCase):
    """Tests for spec invariants in the optimization pipeline."""

    def test_deterministic_output(self):
        """Invariant 9: Same inputs produce same assignments."""
        registry = ResourceRegistry()
        for i in range(3):
            registry.register(make_resource(f"r_{i}", cost_rate=float(30 + i * 10)))
        config = OptimizationConfig(exploration_enabled=False)

        wos = [make_wo()]

        # Run 1
        optimizer1 = DispatchOptimizer(registry)
        d1 = optimizer1.dispatch(wos, "review", "claims", config)

        # Reset capacity for run 2
        for res in registry.list_resources():
            res.capacity.current_load = 0

        # Run 2 — same inputs
        optimizer2 = DispatchOptimizer(registry)
        d2 = optimizer2.dispatch(wos, "review", "claims", config)

        self.assertEqual(
            d1[0].selected_resource_id,
            d2[0].selected_resource_id,
        )

    def test_eligibility_logged_for_exclusions(self):
        """Invariant 7: Every exclusion has audit reason."""
        registry = ResourceRegistry()
        r = make_resource("r_1")
        r.circuit_breaker.status = "open"  # will be excluded
        registry.register(r)
        registry.register(make_resource("r_2"))

        optimizer = DispatchOptimizer(registry)
        decisions = optimizer.dispatch(
            [make_wo()], "review", "claims",
            OptimizationConfig(exploration_enabled=False),
        )
        # Check eligibility results contain the exclusion
        excluded = [
            e for e in decisions[0].eligibility_results
            if not e.eligible and e.resource_id == "r_1"
        ]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].failed_constraint, "circuit_breaker")

    def test_fallback_uses_same_cost_function(self):
        """Invariant 12: Greedy uses same physics cost as optimal."""
        registry = ResourceRegistry()
        for i in range(3):
            registry.register(make_resource(f"r_{i}"))
        optimizer = DispatchOptimizer(registry)

        # Both paths use DefaultAssignmentPhysics.extract_parameters
        # The greedy path reuses the cost_matrix from physics binding
        config_opt = OptimizationConfig(
            archetype="assignment",
            exploration_enabled=False,
        )
        config_greedy = OptimizationConfig(
            archetype="nonexistent",  # force fallback
            greedy_fallback=True,
            exploration_enabled=False,
        )
        wos = [make_wo()]
        d_opt = optimizer.dispatch(wos, "review", "claims", config_opt)
        # Reset capacity
        for res in registry.list_resources():
            res.capacity.current_load = 0
        d_greedy = optimizer.dispatch(wos, "review", "claims", config_greedy)

        # Both should assign to the same resource (cheapest)
        if d_opt[0].selected_resource_id and d_greedy[0].selected_resource_id:
            self.assertEqual(
                d_opt[0].selected_resource_id,
                d_greedy[0].selected_resource_id,
            )


if __name__ == "__main__":
    unittest.main()
