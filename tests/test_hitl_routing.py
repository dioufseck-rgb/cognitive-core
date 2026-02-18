"""Tests for S-020: HITL Capability-Based Routing."""

import importlib.util
import os
import sys
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_mod_path = os.path.join(_base, "engine", "hitl_routing.py")
_spec = importlib.util.spec_from_file_location("engine.hitl_routing", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.hitl_routing"] = _mod
_spec.loader.exec_module(_mod)

RoutingManager = _mod.RoutingManager
Reviewer = _mod.Reviewer
CapabilityRoute = _mod.CapabilityRoute


class TestCapabilityRouting(unittest.TestCase):
    """Test route lookup."""

    def setUp(self):
        self.mgr = RoutingManager()
        self.mgr.add_route(CapabilityRoute("card_dispute", "gate", "fraud_review_l2"))
        self.mgr.add_route(CapabilityRoute("card_dispute", "hold", "compliance_review"))
        self.mgr.add_route(CapabilityRoute("*", "hold", "general_compliance"))

    def test_exact_match(self):
        cap = self.mgr.get_required_capability("card_dispute", "gate")
        self.assertEqual(cap, "fraud_review_l2")

    def test_exact_match_hold(self):
        cap = self.mgr.get_required_capability("card_dispute", "hold")
        self.assertEqual(cap, "compliance_review")

    def test_wildcard_domain(self):
        cap = self.mgr.get_required_capability("unknown_domain", "hold")
        self.assertEqual(cap, "general_compliance")

    def test_no_route_returns_none(self):
        cap = self.mgr.get_required_capability("unknown", "spot_check")
        self.assertIsNone(cap)

    def test_remove_route(self):
        self.assertTrue(self.mgr.remove_route("card_dispute", "gate"))
        self.assertIsNone(self.mgr.get_required_capability("card_dispute", "gate"))

    def test_remove_nonexistent(self):
        self.assertFalse(self.mgr.remove_route("nonexistent", "gate"))


class TestWildcardRouting(unittest.TestCase):
    """Test wildcard matching priority."""

    def test_tier_wildcard(self):
        mgr = RoutingManager()
        mgr.add_route(CapabilityRoute("card_dispute", "*", "fraud_any_tier"))
        cap = mgr.get_required_capability("card_dispute", "spot_check")
        self.assertEqual(cap, "fraud_any_tier")

    def test_full_wildcard(self):
        mgr = RoutingManager()
        mgr.add_route(CapabilityRoute("*", "*", "fallback_review"))
        cap = mgr.get_required_capability("any_domain", "any_tier")
        self.assertEqual(cap, "fallback_review")

    def test_exact_beats_wildcard(self):
        mgr = RoutingManager()
        mgr.add_route(CapabilityRoute("*", "gate", "generic_gate"))
        mgr.add_route(CapabilityRoute("card_dispute", "gate", "specific_gate"))
        cap = mgr.get_required_capability("card_dispute", "gate")
        self.assertEqual(cap, "specific_gate")


class TestReviewerManagement(unittest.TestCase):
    """Test reviewer registration and lookup."""

    def setUp(self):
        self.mgr = RoutingManager()

    def test_register_reviewer(self):
        self.mgr.register_reviewer(Reviewer(
            id="alice", capabilities={"fraud_review_l2", "compliance_review"},
        ))
        reviewers = self.mgr.get_qualified_reviewers("fraud_review_l2")
        self.assertEqual(len(reviewers), 1)
        self.assertEqual(reviewers[0].id, "alice")

    def test_multiple_reviewers(self):
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"fraud_review_l2"}))
        self.mgr.register_reviewer(Reviewer(id="bob", capabilities={"fraud_review_l2"}))
        self.mgr.register_reviewer(Reviewer(id="carol", capabilities={"compliance_review"}))

        fraud = self.mgr.get_qualified_reviewers("fraud_review_l2")
        self.assertEqual(len(fraud), 2)
        compliance = self.mgr.get_qualified_reviewers("compliance_review")
        self.assertEqual(len(compliance), 1)

    def test_deactivated_reviewer_excluded(self):
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"fraud_review_l2"}))
        self.mgr.deactivate_reviewer("alice")
        reviewers = self.mgr.get_qualified_reviewers("fraud_review_l2")
        self.assertEqual(len(reviewers), 0)

    def test_reactivate_reviewer(self):
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"fraud_review_l2"}))
        self.mgr.deactivate_reviewer("alice")
        self.mgr.activate_reviewer("alice")
        reviewers = self.mgr.get_qualified_reviewers("fraud_review_l2")
        self.assertEqual(len(reviewers), 1)

    def test_no_matching_capability(self):
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"fraud_review_l2"}))
        reviewers = self.mgr.get_qualified_reviewers("legal_review")
        self.assertEqual(len(reviewers), 0)


class TestTaskRouting(unittest.TestCase):
    """Test end-to-end task routing."""

    def setUp(self):
        self.mgr = RoutingManager()
        self.mgr.add_route(CapabilityRoute("card_dispute", "gate", "fraud_review_l2"))
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"fraud_review_l2"}))
        self.mgr.register_reviewer(Reviewer(id="bob", capabilities={"fraud_review_l2"}))

    def test_route_task(self):
        task = self.mgr.route_task("wf_123", "card_dispute", "gate")
        self.assertIsNotNone(task)
        self.assertTrue(task.task_id.startswith("rt_"))
        self.assertEqual(task.required_capability, "fraud_review_l2")
        self.assertEqual(set(task.qualified_reviewers), {"alice", "bob"})
        self.assertEqual(task.status, "pending")

    def test_route_no_config_returns_none(self):
        task = self.mgr.route_task("wf_123", "unknown_domain", "gate")
        self.assertIsNone(task)

    def test_route_no_reviewers(self):
        self.mgr.add_route(CapabilityRoute("special", "gate", "rare_capability"))
        task = self.mgr.route_task("wf_123", "special", "gate")
        self.assertIsNotNone(task)
        self.assertEqual(task.qualified_reviewers, [])


class TestTaskAssignment(unittest.TestCase):
    """Test task claiming and assignment."""

    def setUp(self):
        self.mgr = RoutingManager()
        self.mgr.add_route(CapabilityRoute("d", "gate", "review"))
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"review"}))
        self.mgr.register_reviewer(Reviewer(id="bob", capabilities={"review"}))
        self.task = self.mgr.route_task("wf_1", "d", "gate")

    def test_assign_task(self):
        self.assertTrue(self.mgr.assign_task(self.task.task_id, "alice"))
        self.assertEqual(self.task.status, "assigned")
        self.assertEqual(self.task.assigned_to, "alice")

    def test_assign_unqualified_fails(self):
        self.mgr.register_reviewer(Reviewer(id="carol", capabilities={"other"}))
        self.assertFalse(self.mgr.assign_task(self.task.task_id, "carol"))

    def test_assign_nonexistent_task_fails(self):
        self.assertFalse(self.mgr.assign_task("nonexistent", "alice"))

    def test_double_assign_fails(self):
        self.mgr.assign_task(self.task.task_id, "alice")
        self.assertFalse(self.mgr.assign_task(self.task.task_id, "bob"))

    def test_complete_task(self):
        self.mgr.assign_task(self.task.task_id, "alice")
        self.assertTrue(self.mgr.complete_task(self.task.task_id))
        self.assertEqual(self.task.status, "completed")

    def test_complete_releases_capacity(self):
        self.mgr.assign_task(self.task.task_id, "alice")
        reviewer = self.mgr._reviewers["alice"]
        self.assertEqual(reviewer.current_load, 1)
        self.mgr.complete_task(self.task.task_id)
        self.assertEqual(reviewer.current_load, 0)


class TestCapacityLimits(unittest.TestCase):
    """Test reviewer capacity enforcement."""

    def test_at_capacity_excluded(self):
        mgr = RoutingManager()
        mgr.add_route(CapabilityRoute("d", "gate", "review"))
        mgr.register_reviewer(Reviewer(
            id="alice", capabilities={"review"}, max_concurrent=1,
        ))

        t1 = mgr.route_task("wf_1", "d", "gate")
        mgr.assign_task(t1.task_id, "alice")

        # Alice is at capacity — she shouldn't appear in qualified list
        reviewers = mgr.get_qualified_reviewers("review")
        self.assertEqual(len(reviewers), 0)

    def test_capacity_restored_after_complete(self):
        mgr = RoutingManager()
        mgr.add_route(CapabilityRoute("d", "gate", "review"))
        mgr.register_reviewer(Reviewer(
            id="alice", capabilities={"review"}, max_concurrent=1,
        ))

        t1 = mgr.route_task("wf_1", "d", "gate")
        mgr.assign_task(t1.task_id, "alice")
        mgr.complete_task(t1.task_id)

        # Alice should be available again
        reviewers = mgr.get_qualified_reviewers("review")
        self.assertEqual(len(reviewers), 1)


class TestReviewerQueueView(unittest.TestCase):
    """Test queue visibility for reviewers."""

    def setUp(self):
        self.mgr = RoutingManager()
        self.mgr.add_route(CapabilityRoute("d1", "gate", "cap_a"))
        self.mgr.add_route(CapabilityRoute("d2", "gate", "cap_b"))
        self.mgr.register_reviewer(Reviewer(id="alice", capabilities={"cap_a", "cap_b"}))
        self.mgr.register_reviewer(Reviewer(id="bob", capabilities={"cap_a"}))

    def test_pending_for_reviewer(self):
        self.mgr.route_task("wf_1", "d1", "gate")
        self.mgr.route_task("wf_2", "d2", "gate")

        alice_pending = self.mgr.get_pending_for_reviewer("alice")
        bob_pending = self.mgr.get_pending_for_reviewer("bob")

        # Alice has both capabilities
        self.assertEqual(len(alice_pending), 2)
        # Bob only has cap_a
        self.assertEqual(len(bob_pending), 1)

    def test_assigned_for_reviewer(self):
        t1 = self.mgr.route_task("wf_1", "d1", "gate")
        self.mgr.assign_task(t1.task_id, "alice")

        assigned = self.mgr.get_assigned_for_reviewer("alice")
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0].instance_id, "wf_1")


class TestStats(unittest.TestCase):
    """Test routing statistics."""

    def test_stats(self):
        mgr = RoutingManager()
        mgr.add_route(CapabilityRoute("d", "gate", "cap"))
        mgr.register_reviewer(Reviewer(id="alice", capabilities={"cap"}))
        t1 = mgr.route_task("wf_1", "d", "gate")
        mgr.assign_task(t1.task_id, "alice")
        mgr.route_task("wf_2", "d", "gate")

        s = mgr.stats()
        self.assertEqual(s["total_routes"], 1)
        self.assertEqual(s["total_reviewers"], 1)
        self.assertEqual(s["total_tasks"], 2)
        self.assertEqual(s["tasks_by_status"]["assigned"], 1)
        self.assertEqual(s["tasks_by_status"]["pending"], 1)


if __name__ == "__main__":
    unittest.main()
