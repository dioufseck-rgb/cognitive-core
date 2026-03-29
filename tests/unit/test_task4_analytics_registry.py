"""
TASK 4 — Tests: Analytics Artifact Registry

Verifies:
- AnalyticsRegistry loads and validates on startup
- Artifact missing required fields raises InvalidArtifactError at load time
- Eligibility predicates evaluate correctly against case input
- Selection confidence below 0.75 returns abstained, not forced match
- Incompatible artifact pair raises IncompatibleArtifactError at workflow start
- abstained triggers fallback per CC_ANALYTICS_FALLBACK
- Proof ledger records analytics_artifact_loaded at workflow start per artifact
"""

from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_registry(tmpdir: str, content: str) -> str:
    path = os.path.join(tmpdir, "registry.yaml")
    Path(path).write_text(textwrap.dedent(content))
    return path


def _minimal_artifact(**overrides) -> dict:
    base = {
        "artifact_name": "causal.test_v1",
        "artifact_type": "causal_dag",
        "version": "1.0",
        "authored_by": "test-team",
        "eval_gate_passed": "2026-01-01",
        "eligibility_predicates": [{"field": "domain", "operator": "eq", "value": "fraud"}],
    }
    base.update(overrides)
    return base


def _yaml_registry(artifacts: list, incompatibilities: list | None = None) -> str:
    import yaml
    doc: dict = {"artifacts": artifacts}
    if incompatibilities:
        doc["incompatibilities"] = incompatibilities
    return yaml.dump(doc)


# ── Registry Load & Validation ────────────────────────────────────────────────

class TestRegistryLoadAndValidation(unittest.TestCase):

    def test_loads_from_default_registry_yaml(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        reg = AnalyticsRegistry()  # uses config/analytics/registry.yaml
        self.assertGreater(reg.artifact_count, 0, "Registry should load at least one artifact")

    def test_missing_required_field_raises_at_load_time(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry, InvalidArtifactError
        import yaml

        # artifact missing 'eval_gate_passed'
        bad = {
            "artifact_name": "causal.bad_v1",
            "artifact_type": "causal_dag",
            "version": "1.0",
            "authored_by": "team",
            # eval_gate_passed is missing
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({"artifacts": [bad]}))
            with self.assertRaises(InvalidArtifactError) as ctx:
                AnalyticsRegistry(registry_path=path)
        self.assertIn("eval_gate_passed", str(ctx.exception))

    def test_missing_artifact_name_raises(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry, InvalidArtifactError
        import yaml

        bad = {
            "artifact_type": "causal_dag",
            "version": "1.0",
            "authored_by": "team",
            "eval_gate_passed": "2026-01-01",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({"artifacts": [bad]}))
            with self.assertRaises(InvalidArtifactError):
                AnalyticsRegistry(registry_path=path)

    def test_valid_artifact_loads_without_error(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({
                "artifacts": [_minimal_artifact()]
            }))
            reg = AnalyticsRegistry(registry_path=path)
        self.assertEqual(reg.artifact_count, 1)
        self.assertIn("causal.test_v1", reg.artifact_names)

    def test_empty_registry_file_loads_gracefully(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, "artifacts: []\n")
            reg = AnalyticsRegistry(registry_path=path)
        self.assertEqual(reg.artifact_count, 0)

    def test_missing_registry_file_loads_gracefully(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        reg = AnalyticsRegistry(registry_path="/tmp/nonexistent_registry_xyz.yaml")
        self.assertEqual(reg.artifact_count, 0)


# ── Lookup ────────────────────────────────────────────────────────────────────

class TestRegistryLookup(unittest.TestCase):

    def _reg_with_one(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({
                "artifacts": [_minimal_artifact()]
            }))
            return AnalyticsRegistry(registry_path=path)

    def test_lookup_existing_artifact(self):
        reg = self._reg_with_one()
        artifact = reg.lookup("causal.test_v1")
        self.assertEqual(artifact["artifact_name"], "causal.test_v1")

    def test_lookup_unknown_raises(self):
        from cognitive_core.analytics.registry import ArtifactNotFoundError
        reg = self._reg_with_one()
        with self.assertRaises(ArtifactNotFoundError):
            reg.lookup("does.not.exist")


# ── Eligibility Predicates ────────────────────────────────────────────────────

class TestEligibilityPredicates(unittest.TestCase):

    def _reg(self, artifacts):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({"artifacts": artifacts}))
            return AnalyticsRegistry(registry_path=path)

    def test_eq_predicate_matches(self):
        reg = self._reg([_minimal_artifact()])
        eligible = reg.list_eligible({"domain": "fraud"})
        self.assertEqual(len(eligible), 1)

    def test_eq_predicate_no_match(self):
        reg = self._reg([_minimal_artifact()])
        eligible = reg.list_eligible({"domain": "lending"})
        self.assertEqual(len(eligible), 0)

    def test_in_predicate_matches(self):
        artifact = _minimal_artifact(
            eligibility_predicates=[
                {"field": "domain", "operator": "in", "value": ["fraud", "aml", "kyc"]}
            ]
        )
        reg = self._reg([artifact])
        self.assertEqual(len(reg.list_eligible({"domain": "aml"})), 1)
        self.assertEqual(len(reg.list_eligible({"domain": "lending"})), 0)

    def test_ne_predicate(self):
        artifact = _minimal_artifact(
            eligibility_predicates=[
                {"field": "priority", "operator": "ne", "value": "low"}
            ]
        )
        reg = self._reg([artifact])
        self.assertEqual(len(reg.list_eligible({"priority": "high"})), 1)
        self.assertEqual(len(reg.list_eligible({"priority": "low"})), 0)

    def test_multiple_predicates_all_must_match(self):
        artifact = _minimal_artifact(
            eligibility_predicates=[
                {"field": "domain", "operator": "eq", "value": "fraud"},
                {"field": "tier", "operator": "eq", "value": "gate"},
            ]
        )
        reg = self._reg([artifact])
        # Both match
        self.assertEqual(len(reg.list_eligible({"domain": "fraud", "tier": "gate"})), 1)
        # Only one matches
        self.assertEqual(len(reg.list_eligible({"domain": "fraud", "tier": "auto"})), 0)

    def test_filter_by_artifact_type(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        import yaml
        arts = [
            _minimal_artifact(artifact_name="causal.v1", artifact_type="causal_dag"),
            _minimal_artifact(
                artifact_name="sda.v1",
                artifact_type="sequential_decision",
                eligibility_predicates=[{"field": "domain", "operator": "eq", "value": "fraud"}],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({"artifacts": arts}))
            reg = AnalyticsRegistry(registry_path=path)

        ctx = {"domain": "fraud"}
        self.assertEqual(len(reg.list_eligible(ctx, artifact_type="causal_dag")), 1)
        self.assertEqual(len(reg.list_eligible(ctx, artifact_type="sequential_decision")), 1)
        self.assertEqual(len(reg.list_eligible(ctx, artifact_type="constraint_checker")), 0)


# ── Incompatibilities ─────────────────────────────────────────────────────────

class TestIncompatibilities(unittest.TestCase):

    def _reg(self, artifacts, incompatibilities=None):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        import yaml
        doc = {"artifacts": artifacts}
        if incompatibilities:
            doc["incompatibilities"] = incompatibilities
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump(doc))
            return AnalyticsRegistry(registry_path=path)

    def test_incompatible_pair_raises(self):
        from cognitive_core.analytics.registry import IncompatibleArtifactError

        arts = [
            _minimal_artifact(artifact_name="constraints.bsa_v1"),
            _minimal_artifact(artifact_name="constraints.reg_e_v1"),
        ]
        reg = self._reg(arts, incompatibilities=[["constraints.bsa_v1", "constraints.reg_e_v1"]])

        with self.assertRaises(IncompatibleArtifactError) as ctx:
            reg.check_incompatibilities(["constraints.bsa_v1", "constraints.reg_e_v1"])
        self.assertIn("constraints.bsa_v1", str(ctx.exception))

    def test_compatible_pair_does_not_raise(self):
        arts = [
            _minimal_artifact(artifact_name="causal.v1"),
            _minimal_artifact(artifact_name="sda.v1"),
        ]
        reg = self._reg(arts)  # no incompatibilities declared
        reg.check_incompatibilities(["causal.v1", "sda.v1"])  # must not raise

    def test_default_registry_bsa_reg_e_incompatible(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry, IncompatibleArtifactError
        reg = AnalyticsRegistry()  # loads config/analytics/registry.yaml
        with self.assertRaises(IncompatibleArtifactError):
            reg.check_incompatibilities(["constraints.bsa_v1", "constraints.reg_e_v1"])


# ── Selector ──────────────────────────────────────────────────────────────────

class TestArtifactSelector(unittest.TestCase):

    def _reg(self):
        from cognitive_core.analytics.registry import AnalyticsRegistry
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, yaml.dump({
                "artifacts": [_minimal_artifact()]
            }))
            return AnalyticsRegistry(registry_path=path)

    def _llm_with_score(self, artifact_name: str, score: float) -> MagicMock:
        """Return a mock LLM that reports the given confidence score."""
        llm = MagicMock()
        response = MagicMock()
        response.content = json.dumps({"scores": {artifact_name: score}})
        llm.invoke.return_value = response
        return llm

    def test_high_confidence_returns_selected(self):
        from cognitive_core.analytics.selector import ArtifactSelector
        reg = self._reg()
        llm = self._llm_with_score("causal.test_v1", 0.95)
        sel = ArtifactSelector(threshold=0.75)
        result = sel.select(reg, "causal_dag", {"domain": "fraud"}, tier="auto", llm=llm)
        self.assertEqual(result.outcome, "selected")
        self.assertIsNotNone(result.artifact)
        self.assertAlmostEqual(result.confidence, 0.95, places=2)

    def test_low_confidence_returns_abstained(self):
        from cognitive_core.analytics.selector import ArtifactSelector
        reg = self._reg()
        llm = self._llm_with_score("causal.test_v1", 0.50)
        sel = ArtifactSelector(threshold=0.75, fallback="escalate")
        result = sel.select(reg, "causal_dag", {"domain": "fraud"}, tier="auto", llm=llm)
        self.assertEqual(result.outcome, "abstained")
        self.assertEqual(result.fallback_action, "escalate")

    def test_no_eligible_returns_no_eligible_artifact(self):
        from cognitive_core.analytics.selector import ArtifactSelector
        reg = self._reg()
        sel = ArtifactSelector()
        result = sel.select(reg, "causal_dag", {"domain": "lending"}, tier="auto")
        self.assertEqual(result.outcome, "no_eligible_artifact")

    def test_abstained_at_hold_tier_always_escalates(self):
        from cognitive_core.analytics.selector import ArtifactSelector
        reg = self._reg()
        llm = self._llm_with_score("causal.test_v1", 0.30)
        # Even with fallback=skip, hold tier forces escalate
        sel = ArtifactSelector(threshold=0.75, fallback="skip")
        result = sel.select(reg, "causal_dag", {"domain": "fraud"}, tier="hold", llm=llm)
        self.assertEqual(result.outcome, "abstained")
        self.assertEqual(result.fallback_action, "escalate",
                         "hold tier must always escalate regardless of CC_ANALYTICS_FALLBACK")

    def test_fallback_skip_returns_skip_action(self):
        from cognitive_core.analytics.selector import ArtifactSelector
        reg = self._reg()
        llm = self._llm_with_score("causal.test_v1", 0.40)
        sel = ArtifactSelector(threshold=0.75, fallback="skip")
        result = sel.select(reg, "causal_dag", {"domain": "fraud"}, tier="auto", llm=llm)
        self.assertEqual(result.outcome, "abstained")
        self.assertEqual(result.fallback_action, "skip")

    def test_fallback_fail_raises(self):
        from cognitive_core.analytics.selector import ArtifactSelector, ArtifactSelectionFailedError
        reg = self._reg()
        llm = self._llm_with_score("causal.test_v1", 0.20)
        sel = ArtifactSelector(threshold=0.75, fallback="fail")
        with self.assertRaises(ArtifactSelectionFailedError):
            sel.select(reg, "causal_dag", {"domain": "fraud"}, tier="auto", llm=llm)

    def test_no_llm_eligible_artifact_returns_selected(self):
        """Without LLM, eligible artifacts get score 1.0 → selected."""
        from cognitive_core.analytics.selector import ArtifactSelector
        reg = self._reg()
        sel = ArtifactSelector(threshold=0.75)
        result = sel.select(reg, "causal_dag", {"domain": "fraud"}, tier="auto", llm=None)
        self.assertEqual(result.outcome, "selected")


# ── Proof Ledger Event ────────────────────────────────────────────────────────

class TestAnalyticsArtifactLoadedProofEvent(unittest.TestCase):

    def test_analytics_artifact_loaded_recorded_per_artifact(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        # Load the real registry artifacts by name
        gov.load_analytics_artifacts(
            domain="fraud",
            workflow_type="fraud_triage",
            artifact_names=["causal.fraud_dag_v1"],
        )

        loaded_events = [
            e for e in gov._proof_ledger
            if e["event"] == "analytics_artifact_loaded"
        ]
        self.assertGreaterEqual(len(loaded_events), 1,
                                f"analytics_artifact_loaded event missing; "
                                f"ledger: {gov._proof_ledger}")
        self.assertEqual(loaded_events[0]["artifact_name"], "causal.fraud_dag_v1")

    def test_unknown_artifact_records_fallback_event(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov.load_analytics_artifacts(
            domain="fraud",
            workflow_type="fraud_triage",
            artifact_names=["causal.nonexistent_v99"],
        )

        fallback_events = [
            e for e in gov._proof_ledger
            if e["event"] == "analytics_fallback_applied"
        ]
        self.assertGreaterEqual(len(fallback_events), 1)

    def test_multiple_artifacts_each_get_an_event(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov.load_analytics_artifacts(
            domain="fraud",
            workflow_type="test_workflow",
            artifact_names=["causal.fraud_dag_v1", "sda.fraud_policy_v1"],
        )

        loaded_events = [
            e for e in gov._proof_ledger
            if e["event"] == "analytics_artifact_loaded"
        ]
        loaded_names = {e["artifact_name"] for e in loaded_events}
        self.assertIn("causal.fraud_dag_v1", loaded_names)
        self.assertIn("sda.fraud_policy_v1", loaded_names)


if __name__ == "__main__":
    unittest.main()
