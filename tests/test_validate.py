"""
Cognitive Core — Spec Validator Tests

Tests for three-layer YAML validation: syntactic, semantic, pragmatic.
"""

import os
import unittest
import tempfile
import yaml
from pathlib import Path

# Import validate.py directly to avoid engine/__init__.py langgraph dependency
import importlib.util
import sys as _sys
_validate_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engine", "validate.py"
)
_spec = importlib.util.spec_from_file_location("engine_validate", _validate_path)
_mod = importlib.util.module_from_spec(_spec)
_sys.modules["engine_validate"] = _mod
_spec.loader.exec_module(_mod)

validate_workflow_syntax = _mod.validate_workflow_syntax
validate_workflow_semantics = _mod.validate_workflow_semantics
validate_domain_syntax = _mod.validate_domain_syntax
validate_domain_against_workflow = _mod.validate_domain_against_workflow
validate_coordinator_syntax = _mod.validate_coordinator_syntax
validate_delegation_targets = _mod.validate_delegation_targets
validate_domain_governance_consistency = _mod.validate_domain_governance_consistency
validate_workflow_domain_pairs = _mod.validate_workflow_domain_pairs
validate_all = _mod.validate_all
ValidationResult = _mod.ValidationResult

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
# Syntactic tests
# ═══════════════════════════════════════════════════════════════════

class TestWorkflowSyntax(unittest.TestCase):

    def test_minimal_valid_workflow(self):
        wf = {
            "name": "test",
            "steps": [{
                "name": "step1",
                "primitive": "classify",
                "params": {"categories": "a,b", "criteria": "test"},
            }],
        }
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertEqual(len(r.errors), 0, r.summary())

    def test_missing_name(self):
        wf = {"steps": [{"name": "s1", "primitive": "classify",
                         "params": {"categories": "a", "criteria": "b"}}]}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("Missing required field 'name'" in e.message for e in r.errors))

    def test_missing_steps(self):
        wf = {"name": "test"}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("Missing required field 'steps'" in e.message for e in r.errors))

    def test_empty_steps(self):
        wf = {"name": "test", "steps": []}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("empty" in e.message for e in r.errors))

    def test_unknown_primitive(self):
        wf = {"name": "test", "steps": [{"name": "s1", "primitive": "explode"}]}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("Unknown primitive" in e.message for e in r.errors))

    def test_missing_required_params(self):
        wf = {"name": "test", "steps": [{"name": "s1", "primitive": "classify", "params": {}}]}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("categories" in e.message for e in r.errors))
        self.assertTrue(any("criteria" in e.message for e in r.errors))

    def test_duplicate_step_name(self):
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify", "params": {"categories": "a", "criteria": "b"}},
            {"name": "s1", "primitive": "generate", "params": {"requirements": "x", "format": "y", "constraints": "z"}},
        ]}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("Duplicate" in e.message for e in r.errors))

    def test_agentic_workflow_valid(self):
        wf = {"name": "test", "mode": "agentic", "goal": "do things"}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertEqual(len(r.errors), 0, r.summary())

    def test_agentic_without_goal(self):
        wf = {"name": "test", "mode": "agentic"}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("goal" in e.message for e in r.errors))

    def test_invalid_transition_target(self):
        """Caught at semantic level, not syntactic."""
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify",
             "params": {"categories": "a", "criteria": "b"},
             "transitions": [{"goto": "nonexistent"}]},
        ]}
        r = validate_workflow_syntax(wf, "test.yaml")
        # Syntactic should pass — it's a valid structure
        self.assertEqual(len(r.errors), 0)

    def test_agent_decide_missing_options(self):
        wf = {"name": "test", "steps": [{
            "name": "s1", "primitive": "classify",
            "params": {"categories": "a", "criteria": "b"},
            "transitions": [{"agent_decide": {"prompt": "decide"}}],
        }]}
        r = validate_workflow_syntax(wf, "test.yaml")
        self.assertTrue(any("options" in e.message for e in r.errors))


class TestDomainSyntax(unittest.TestCase):

    def test_valid_domain(self):
        dom = {"domain_name": "test", "governance": "auto"}
        r = validate_domain_syntax(dom, "test.yaml")
        self.assertEqual(len(r.errors), 0)

    def test_missing_domain_name(self):
        dom = {"governance": "auto"}
        r = validate_domain_syntax(dom, "test.yaml")
        self.assertTrue(any("domain_name" in e.message for e in r.errors))

    def test_invalid_governance_tier(self):
        dom = {"domain_name": "test", "governance": "yolo"}
        r = validate_domain_syntax(dom, "test.yaml")
        self.assertTrue(any("Invalid governance tier" in e.message for e in r.errors))


class TestCoordinatorSyntax(unittest.TestCase):

    def test_delegation_missing_target(self):
        cfg = {"delegations": [{"name": "test", "conditions": [], "contract": "c"}]}
        r = validate_coordinator_syntax(cfg, "config.yaml")
        self.assertTrue(any("target_workflow" in e.message for e in r.errors))

    def test_delegation_invalid_mode(self):
        cfg = {"delegations": [{
            "name": "test", "conditions": [], "contract": "c",
            "target_workflow": "w", "target_domain": "d",
            "mode": "chaos", "inputs": {},
        }]}
        r = validate_coordinator_syntax(cfg, "config.yaml")
        self.assertTrue(any("Invalid mode" in e.message for e in r.errors))

    def test_unknown_operator(self):
        cfg = {"delegations": [{
            "name": "test", "conditions": [{"operator": "vibes"}],
            "contract": "c", "target_workflow": "w", "target_domain": "d", "inputs": {},
        }]}
        r = validate_coordinator_syntax(cfg, "config.yaml")
        self.assertTrue(any("Unknown operator" in e.message for e in r.errors))


# ═══════════════════════════════════════════════════════════════════
# Semantic tests
# ═══════════════════════════════════════════════════════════════════

class TestWorkflowSemantics(unittest.TestCase):

    def test_broken_transition_target(self):
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify",
             "params": {"categories": "a", "criteria": "b"},
             "transitions": [{"goto": "nonexistent"}]},
        ]}
        r = validate_workflow_semantics(wf, "test.yaml")
        self.assertTrue(any("nonexistent" in e.message for e in r.errors))

    def test_broken_agent_decide_target(self):
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify",
             "params": {"categories": "a", "criteria": "b"},
             "transitions": [{"agent_decide": {
                 "options": [{"step": "ghost", "description": "nope"}]
             }}]},
        ]}
        r = validate_workflow_semantics(wf, "test.yaml")
        self.assertTrue(any("ghost" in e.message for e in r.errors))

    def test_sequential_steps_reachable(self):
        """Steps connected by implicit sequential flow should not be flagged."""
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify", "params": {"categories": "a", "criteria": "b"}},
            {"name": "s2", "primitive": "generate", "params": {"requirements": "r", "format": "f", "constraints": "c"}},
            {"name": "s3", "primitive": "verify", "params": {"rules": "r"}},
        ]}
        r = validate_workflow_semantics(wf, "test.yaml")
        unreachable = [i for i in r.warnings if "unreachable" in i.message.lower()]
        self.assertEqual(len(unreachable), 0,
            f"Sequential steps should be reachable: {[str(u) for u in unreachable]}")

    def test_truly_unreachable_step(self):
        """A step with no path to it: s1 → s2 → end, plus a disconnected 'island'."""
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify",
             "params": {"categories": "a", "criteria": "b"},
             "transitions": [{"default": "s2"}]},
            {"name": "s2", "primitive": "generate",
             "params": {"requirements": "r", "format": "f", "constraints": "c"},
             "transitions": [{"default": "__end__"}]},
            {"name": "island", "primitive": "think", "params": {"instruction": "i"}},
        ]}
        r = validate_workflow_semantics(wf, "test.yaml")
        unreachable_names = [i.location for i in r.warnings if "unreachable" in i.message.lower()]
        self.assertIn("step:island", unreachable_names)

    def test_valid_previous_ref(self):
        """${previous.*} is a valid runtime reference, not a step name."""
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify", "params": {"categories": "a", "criteria": "b"}},
            {"name": "s2", "primitive": "generate",
             "params": {"requirements": "${previous.category}", "format": "f", "constraints": "c"}},
        ]}
        r = validate_workflow_semantics(wf, "test.yaml")
        ref_warnings = [i for i in r.warnings if "previous" in i.message]
        self.assertEqual(len(ref_warnings), 0, "previous.* should be recognized as valid")


class TestDomainWorkflowCrossRef(unittest.TestCase):

    def test_unresolved_domain_ref(self):
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify",
             "params": {"categories": "${domain.classify.categories}", "criteria": "test"}},
        ]}
        dom = {"domain_name": "test", "governance": "auto"}
        # Domain doesn't define classify.categories
        r = validate_domain_against_workflow(dom, "dom.yaml", wf, "wf.yaml")
        self.assertTrue(any("does not resolve" in e.message for e in r.errors))

    def test_resolved_domain_ref(self):
        wf = {"name": "test", "steps": [
            {"name": "s1", "primitive": "classify",
             "params": {"categories": "${domain.classify.categories}", "criteria": "test"}},
        ]}
        dom = {"domain_name": "test", "governance": "auto",
               "classify": {"categories": "a, b, c"}}
        r = validate_domain_against_workflow(dom, "dom.yaml", wf, "wf.yaml")
        self.assertEqual(len(r.errors), 0, r.summary())


# ═══════════════════════════════════════════════════════════════════
# Pragmatic tests
# ═══════════════════════════════════════════════════════════════════

class TestDelegationTargets(unittest.TestCase):

    def test_missing_target_workflow(self):
        cfg = {"delegations": [{
            "name": "test", "conditions": [],
            "target_workflow": "nonexistent", "target_domain": "exists",
            "contract": "c", "inputs": {},
        }]}
        r = validate_delegation_targets(cfg, "config.yaml",
            {"real_wf": {}}, {"exists": {}})
        self.assertTrue(any("nonexistent" in e.message for e in r.errors))

    def test_missing_tool_data_warning(self):
        cfg = {"delegations": [{
            "name": "test", "conditions": [],
            "target_workflow": "wf_with_retrieve", "target_domain": "dom",
            "contract": "c", "inputs": {"some_field": "value"},
        }]}
        wfs = {"wf_with_retrieve": {
            "steps": [{"name": "s1", "primitive": "retrieve"}]
        }}
        r = validate_delegation_targets(cfg, "config.yaml", wfs, {"dom": {}})
        self.assertTrue(any("get_*" in w.message for w in r.warnings))


class TestGovernanceConsistency(unittest.TestCase):

    def test_domain_tier_not_in_coordinator(self):
        domains = {"test_domain": {"governance": "ultra_secure"}}
        coord = {"governance_tiers": {"auto": {}, "gate": {}}}
        r = validate_domain_governance_consistency(domains, coord, "config.yaml")
        self.assertTrue(any("ultra_secure" in e.message for e in r.errors))

    def test_valid_tier(self):
        domains = {"test_domain": {"governance": "auto"}}
        coord = {"governance_tiers": {"auto": {}, "gate": {}}}
        r = validate_domain_governance_consistency(domains, coord, "config.yaml")
        self.assertEqual(len(r.errors), 0)


class TestWorkflowDomainPairs(unittest.TestCase):

    def test_domain_references_missing_workflow(self):
        domains = {"test_domain": {"workflow": "ghost_workflow"}}
        workflows = {"real_workflow": {}}
        r = validate_workflow_domain_pairs(workflows, domains)
        self.assertTrue(any("ghost_workflow" in e.message for e in r.errors))


# ═══════════════════════════════════════════════════════════════════
# Full project validation
# ═══════════════════════════════════════════════════════════════════

class TestFullProjectValidation(unittest.TestCase):
    """Run validation against the actual project specs."""

    def test_project_has_no_errors(self):
        result = validate_all(_project_root)
        if result.errors:
            self.fail(
                f"Project spec validation found {len(result.errors)} errors:\n"
                + "\n".join(str(e) for e in result.errors)
            )

    def test_project_warnings_are_known(self):
        result = validate_all(_project_root)
        # We expect 3 known pragmatic warnings about delegation tool data
        # If new warnings appear, investigate them
        for w in result.warnings:
            self.assertEqual(w.layer, "pragmatic",
                f"Unexpected non-pragmatic warning: {w}")


if __name__ == "__main__":
    unittest.main()
