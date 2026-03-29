"""
Cognitive Core — Analytics Artifact Registry (TASK 4)

Loads artifact configurations from config/analytics/registry.yaml at startup.
Read-only at runtime — no runtime registration path exists.

Usage:
    from cognitive_core.analytics.registry import AnalyticsRegistry, InvalidArtifactError

    reg = AnalyticsRegistry()          # loads + validates on construction
    artifact = reg.lookup("causal.fraud_dag_v1")
    reg.check_incompatibilities(["causal.fraud_dag_v1", "sda.fraud_policy_v1"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("cognitive_core.analytics.registry")

# Required fields every artifact must declare
_REQUIRED_FIELDS = {"artifact_name", "artifact_type", "version", "authored_by", "eval_gate_passed"}


# ── Exceptions ────────────────────────────────────────────────────────────────

class InvalidArtifactError(ValueError):
    """Raised at load time when an artifact is missing required fields."""

    def __init__(self, artifact_name: str, missing: set[str]):
        self.artifact_name = artifact_name
        self.missing = missing
        super().__init__(
            f"Artifact '{artifact_name}' is invalid — missing required fields: "
            + ", ".join(sorted(missing))
        )


class IncompatibleArtifactError(RuntimeError):
    """Raised at workflow start when an incompatible artifact pair is selected."""

    def __init__(self, a: str, b: str):
        self.artifact_a = a
        self.artifact_b = b
        super().__init__(
            f"Incompatible artifacts selected together: '{a}' and '{b}'"
        )


class ArtifactNotFoundError(KeyError):
    """Raised when lookup() cannot find a named artifact."""

    def __init__(self, name: str):
        self.artifact_name = name
        super().__init__(f"Artifact not found in registry: '{name}'")


# ── Predicate Evaluation ──────────────────────────────────────────────────────

def _eval_predicate(predicate: dict[str, Any], context: dict[str, Any]) -> bool:
    """
    Evaluate a single eligibility predicate against a context dict.

    Supported operators: eq, ne, lt, gt, lte, gte, in, not_in, contains.
    No arbitrary code execution — predicate is a declarative struct only.
    """
    field = predicate.get("field", "")
    operator = predicate.get("operator", "eq")
    expected = predicate.get("value")

    actual = context.get(field)

    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "lt":
        return actual is not None and actual < expected
    if operator == "gt":
        return actual is not None and actual > expected
    if operator == "lte":
        return actual is not None and actual <= expected
    if operator == "gte":
        return actual is not None and actual >= expected
    if operator == "in":
        return actual in (expected or [])
    if operator == "not_in":
        return actual not in (expected or [])
    if operator == "contains":
        return expected in (actual or "")

    logger.warning("Unknown predicate operator '%s' — treating as False", operator)
    return False


def _eval_predicates(predicates: list[dict], context: dict[str, Any]) -> bool:
    """All predicates must match (AND semantics)."""
    return all(_eval_predicate(p, context) for p in (predicates or []))


# ── Registry ──────────────────────────────────────────────────────────────────

class AnalyticsRegistry:
    """
    Artifact registry. Loads and validates config/analytics/registry.yaml
    (or CC_ANALYTICS_REGISTRY_PATH) at construction time.

    Operations:
      lookup(name)                           → artifact dict
      list_eligible(context, artifact_type)  → [artifact dict, ...]
      check_incompatibilities(names)         → raises IncompatibleArtifactError
    """

    def __init__(self, registry_path: str | None = None):
        path = registry_path or os.environ.get(
            "CC_ANALYTICS_REGISTRY_PATH", "config/analytics/registry.yaml"
        )
        self._path = Path(path)
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._incompatibilities: list[list[str]] = []
        self._load()

    # ── Load & Validate ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            logger.debug("Analytics registry not found at %s — no artifacts loaded", self._path)
            return

        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}

        for artifact in raw.get("artifacts", []):
            self._validate(artifact)
            name = artifact["artifact_name"]
            self._artifacts[name] = artifact
            logger.info("Loaded artifact: %s (%s v%s)",
                        name, artifact["artifact_type"], artifact["version"])

        self._incompatibilities = raw.get("incompatibilities", [])
        logger.info("Registry loaded: %d artifacts, %d incompatibility pairs",
                    len(self._artifacts), len(self._incompatibilities))

    @staticmethod
    def _validate(artifact: dict[str, Any]) -> None:
        """Raise InvalidArtifactError at load time if required fields are missing."""
        name = artifact.get("artifact_name", "<unnamed>")
        missing = _REQUIRED_FIELDS - set(artifact.keys())
        if missing:
            raise InvalidArtifactError(name, missing)

    # ── Queries ──────────────────────────────────────────────────────────────

    def lookup(self, name: str) -> dict[str, Any]:
        """Return the artifact config for the given name."""
        if name not in self._artifacts:
            raise ArtifactNotFoundError(name)
        return dict(self._artifacts[name])

    def list_eligible(
        self,
        context: dict[str, Any],
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return all artifacts whose eligibility_predicates match the context.
        Optionally filter by artifact_type.
        """
        results = []
        for artifact in self._artifacts.values():
            if artifact_type and artifact.get("artifact_type") != artifact_type:
                continue
            predicates = artifact.get("eligibility_predicates", [])
            if _eval_predicates(predicates, context):
                results.append(dict(artifact))
        return results

    def check_incompatibilities(self, selected_names: list[str]) -> None:
        """
        Raise IncompatibleArtifactError if any registered incompatibility pair
        is present in the selected_names list.
        Called at workflow start before execution begins.
        """
        name_set = set(selected_names)
        for pair in self._incompatibilities:
            if len(pair) >= 2 and pair[0] in name_set and pair[1] in name_set:
                raise IncompatibleArtifactError(pair[0], pair[1])

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def artifact_names(self) -> list[str]:
        return list(self._artifacts.keys())

    @property
    def artifact_count(self) -> int:
        return len(self._artifacts)
