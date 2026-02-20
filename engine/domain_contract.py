"""
Cognitive Core — Domain Contract Validator

Reads the schema-first sections of a domain YAML and provides:
  1. Tool contracts (required_keys per data source) → I0 validation
  2. Enum definitions → output validation
  3. Artifact contracts (required_keys, redactions, validations) → I3 validation
  4. Cross-step consistency rules → I7 validation
  5. Step confidence floors → I4 validation
  6. Governance triggers → I5 context

Usage:
    from engine.domain_contract import DomainContract

    contract = DomainContract.from_yaml("domains/electronics_return.yaml")

    # I0: validate retrieve completeness
    violations = contract.check_retrieve_completeness(step_output)

    # I3: validate artifact schema
    violations = contract.check_artifact("generate_response", artifact_dict)

    # I7: validate cross-step consistency
    violations = contract.check_consistency(step_outputs)

    # I4: validate confidence floors
    violations = contract.check_confidence("classify_type", 0.42)

    # All at once
    all_violations = contract.validate_run(step_outputs)
"""

from __future__ import annotations

import yaml
import os
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class ToolContract:
    """Declares expected keys for a data source."""
    name: str
    required_keys: list[str]


@dataclass
class EnumDef:
    """A named set of allowed values."""
    name: str
    values: list[str]


@dataclass
class ArtifactContract:
    """Declares the schema for a generate step's artifact."""
    name: str
    required_keys: list[str]
    redactions: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)


@dataclass
class ConsistencyRule:
    """A cross-step consistency rule."""
    rule: str
    severity: str = "critical"  # critical | warning
    description: str = ""


@dataclass
class StepSpec:
    """Per-step metadata from the domain."""
    name: str
    primitive: str
    confidence_floor: float | None = None
    output_schema: dict[str, str] | None = None
    artifact_schema: str | None = None
    coherence_with: str | None = None
    coherence_table: dict[str, list[str]] | None = None


@dataclass
class Violation:
    """A single validation failure."""
    invariant: str          # I0, I3, I4, I7, etc.
    step: str
    message: str
    severity: str = "error"  # error | warning


class DomainContract:
    """
    Parses a schema-first domain YAML and provides validation methods.

    Supports two formats:
      1. Schema-first (v2): has schema_version, domain.name, tools, enums, etc.
      2. Legacy (v1): flat YAML with step_name: {params} structure
    """

    def __init__(
        self,
        domain_name: str = "",
        tool_contracts: dict[str, ToolContract] | None = None,
        enums: dict[str, EnumDef] | None = None,
        artifact_contracts: dict[str, ArtifactContract] | None = None,
        consistency_rules: list[ConsistencyRule] | None = None,
        step_specs: dict[str, StepSpec] | None = None,
        governance: dict[str, Any] | None = None,
        raw: dict[str, Any] | None = None,
    ):
        self.domain_name = domain_name
        self.tool_contracts = tool_contracts or {}
        self.enums = enums or {}
        self.artifact_contracts = artifact_contracts or {}
        self.consistency_rules = consistency_rules or []
        self.step_specs = step_specs or {}
        self.governance = governance or {}
        self.raw = raw or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DomainContract":
        """Load a domain YAML and extract contracts."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Detect format version
        if "schema_version" in raw or "domain" in raw:
            return cls._parse_v2(raw)
        else:
            return cls._parse_legacy(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "DomainContract":
        """Parse from an already-loaded dict."""
        if "schema_version" in raw or "domain" in raw:
            return cls._parse_v2(raw)
        else:
            return cls._parse_legacy(raw)

    @classmethod
    def _parse_v2(cls, raw: dict) -> "DomainContract":
        """Parse schema-first (v2) format."""
        domain_meta = raw.get("domain", {})
        domain_name = domain_meta.get("name", "") if isinstance(domain_meta, dict) else ""

        # Tool contracts
        tool_contracts = {}
        for name, spec in raw.get("tools", {}).items():
            if isinstance(spec, dict):
                tool_contracts[name] = ToolContract(
                    name=name,
                    required_keys=spec.get("required_keys", []),
                )

        # Enums
        enums = {}
        for name, spec in raw.get("enums", {}).items():
            if isinstance(spec, dict):
                enums[name] = EnumDef(name=name, values=list(spec.keys()))
            elif isinstance(spec, list):
                enums[name] = EnumDef(name=name, values=spec)

        # Artifact contracts
        artifact_contracts = {}
        for name, spec in raw.get("artifacts", {}).items():
            if isinstance(spec, dict):
                artifact_contracts[name] = ArtifactContract(
                    name=name,
                    required_keys=spec.get("required_keys", []),
                    redactions=spec.get("redactions", []),
                    validations=spec.get("validations", []),
                )

        # Step specs
        step_specs = {}
        for step in raw.get("steps", []):
            if isinstance(step, dict) and "name" in step:
                ss = StepSpec(
                    name=step["name"],
                    primitive=step.get("primitive", ""),
                    confidence_floor=step.get("confidence_floor"),
                    output_schema=step.get("output_schema"),
                    artifact_schema=step.get("artifact_schema"),
                    coherence_with=step.get("coherence_with"),
                    coherence_table=step.get("coherence_table"),
                )
                step_specs[step["name"]] = ss

        # Consistency rules
        consistency_rules = []
        for rule_spec in raw.get("cross_step_consistency", []):
            if isinstance(rule_spec, dict):
                consistency_rules.append(ConsistencyRule(
                    rule=rule_spec.get("rule", ""),
                    severity=rule_spec.get("severity", "critical"),
                    description=rule_spec.get("description", ""),
                ))

        # Governance
        governance = raw.get("governance", {})

        return cls(
            domain_name=domain_name,
            tool_contracts=tool_contracts,
            enums=enums,
            artifact_contracts=artifact_contracts,
            consistency_rules=consistency_rules,
            step_specs=step_specs,
            governance=governance,
            raw=raw,
        )

    @classmethod
    def _parse_legacy(cls, raw: dict) -> "DomainContract":
        """Parse legacy (v1) format — extract what we can."""
        domain_name = raw.get("domain_name", "")

        # Legacy domains don't have structured tool/artifact/enum sections.
        # Return an empty contract — the eval harness falls back to
        # hardcoded ARTIFACT_SCHEMAS for these domains.
        return cls(domain_name=domain_name, raw=raw)

    # ═══════════════════════════════════════════════════════════════
    # I0: Retrieve Completeness
    # ═══════════════════════════════════════════════════════════════

    def check_retrieve_completeness(self, retrieve_output: dict) -> list[Violation]:
        """Check that all tool contracts are satisfied."""
        violations = []
        data = retrieve_output.get("data", {})

        for tool_name, contract in self.tool_contracts.items():
            source_data = data.get(tool_name, {})
            if not source_data:
                violations.append(Violation(
                    invariant="I0",
                    step="retrieve",
                    message=f"Tool '{tool_name}' returned no data",
                    severity="error",
                ))
                continue

            if isinstance(source_data, dict):
                missing = [k for k in contract.required_keys
                           if k not in source_data]
                if missing:
                    violations.append(Violation(
                        invariant="I0",
                        step="retrieve",
                        message=f"Tool '{tool_name}' missing keys: {missing}",
                        severity="error",
                    ))

                # Check for empty values on required keys
                empty = [k for k in contract.required_keys
                         if k in source_data
                         and source_data[k] in (None, "", [], {})]
                if empty:
                    violations.append(Violation(
                        invariant="I0",
                        step="retrieve",
                        message=f"Tool '{tool_name}' has empty required keys: {empty}",
                        severity="warning",
                    ))

        return violations

    # ═══════════════════════════════════════════════════════════════
    # I3: Artifact Schema Validation
    # ═══════════════════════════════════════════════════════════════

    def check_artifact(self, step_name: str, artifact: Any) -> list[Violation]:
        """Validate a generate step's artifact against its contract."""
        violations = []

        # Find the artifact schema name for this step
        step_spec = self.step_specs.get(step_name)
        schema_name = step_spec.artifact_schema if step_spec else None

        # Try direct lookup by step name if no artifact_schema reference
        contract = None
        if schema_name:
            contract = self.artifact_contracts.get(schema_name)
        if not contract:
            # Fall back: try matching step name patterns
            for ac_name, ac in self.artifact_contracts.items():
                if step_name.replace("generate_", "") in ac_name:
                    contract = ac
                    break

        if not contract:
            return violations  # No contract defined — skip

        # Check artifact exists and is a dict
        if artifact is None:
            violations.append(Violation(
                invariant="I3",
                step=step_name,
                message="Artifact is None",
            ))
            return violations

        if not isinstance(artifact, dict):
            violations.append(Violation(
                invariant="I3",
                step=step_name,
                message=f"Artifact is {type(artifact).__name__}, expected dict",
            ))
            return violations

        # Check required keys
        missing = [k for k in contract.required_keys if k not in artifact]
        if missing:
            violations.append(Violation(
                invariant="I3",
                step=step_name,
                message=f"Missing required keys: {missing}",
            ))

        # Check redactions — these fields must NOT appear
        leaked = [r for r in contract.redactions
                  if _deep_contains(artifact, r)]
        if leaked:
            violations.append(Violation(
                invariant="I3",
                step=step_name,
                message=f"Redacted fields leaked into artifact: {leaked}",
                severity="error",
            ))

        # Check enum validations
        for validation in contract.validations:
            v_result = self._check_validation(validation, artifact)
            if v_result:
                violations.append(Violation(
                    invariant="I3",
                    step=step_name,
                    message=f"Validation failed: {validation} ({v_result})",
                    severity="warning",
                ))

        return violations

    def get_artifact_required_keys(self, step_name: str) -> list[str] | None:
        """Get required keys for a step's artifact. Returns None if no contract."""
        step_spec = self.step_specs.get(step_name)
        schema_name = step_spec.artifact_schema if step_spec else None
        contract = None
        if schema_name:
            contract = self.artifact_contracts.get(schema_name)
        if not contract:
            for ac_name, ac in self.artifact_contracts.items():
                if step_name.replace("generate_", "") in ac_name:
                    contract = ac
                    break
        return contract.required_keys if contract else None

    # ═══════════════════════════════════════════════════════════════
    # I4: Confidence Floor
    # ═══════════════════════════════════════════════════════════════

    def check_confidence(self, step_name: str, confidence: float | None) -> list[Violation]:
        """Check if a step's confidence meets its floor."""
        violations = []
        spec = self.step_specs.get(step_name)
        if not spec or spec.confidence_floor is None or confidence is None:
            return violations

        if confidence < spec.confidence_floor:
            violations.append(Violation(
                invariant="I4",
                step=step_name,
                message=(f"Confidence {confidence:.2f} below floor "
                         f"{spec.confidence_floor:.2f}"),
            ))
        return violations

    # ═══════════════════════════════════════════════════════════════
    # I7: Cross-Step Consistency
    # ═══════════════════════════════════════════════════════════════

    def check_consistency(self, step_outputs: dict[str, dict]) -> list[Violation]:
        """Evaluate all cross-step consistency rules."""
        violations = []

        # Check declared consistency rules
        for rule in self.consistency_rules:
            result = self._evaluate_consistency_rule(rule, step_outputs)
            if result:
                violations.append(Violation(
                    invariant="I7",
                    step="cross_step",
                    message=f"{rule.description or rule.rule}: {result}",
                    severity=rule.severity,
                ))

        # Check coherence tables from step specs
        for step_name, spec in self.step_specs.items():
            if spec.coherence_with and spec.coherence_table:
                ref_step = step_outputs.get(spec.coherence_with, {})
                this_step = step_outputs.get(step_name, {})

                ref_category = ref_step.get("category")
                this_finding = this_step.get("finding")

                if ref_category and this_finding:
                    allowed = spec.coherence_table.get(ref_category, [])
                    if allowed and this_finding not in allowed:
                        violations.append(Violation(
                            invariant="I7",
                            step=step_name,
                            message=(f"{ref_category} incompatible with "
                                     f"{this_finding} (allowed: {allowed})"),
                        ))

        return violations

    def _evaluate_consistency_rule(
        self, rule: ConsistencyRule, step_outputs: dict[str, dict]
    ) -> str | None:
        """
        Evaluate a consistency rule expression.

        Supports patterns like:
          "If step.field == value AND step2.field contains_any ['a','b'] => violation"

        Returns a description of the violation, or None if rule passes.
        """
        rule_text = rule.rule.strip()
        if not rule_text:
            return None

        # Parse: "If <conditions> => violation"
        if "=>" not in rule_text:
            return None

        condition_part = rule_text.split("=>")[0].strip()
        if condition_part.lower().startswith("if "):
            condition_part = condition_part[3:].strip()

        # Split on AND
        clauses = [c.strip() for c in condition_part.split(" AND ")]

        all_match = True
        for clause in clauses:
            if not clause:
                continue
            if not self._evaluate_clause(clause, step_outputs):
                all_match = False
                break

        if all_match:
            return "consistency violation"
        return None

    def _evaluate_clause(self, clause: str, step_outputs: dict[str, dict]) -> bool:
        """Evaluate a single clause like 'step.field == value'."""
        clause = clause.strip()

        # Handle "contains_any"
        if "contains_any" in clause:
            parts = clause.split("contains_any")
            if len(parts) == 2:
                field_path = parts[0].strip()
                values_str = parts[1].strip()
                actual = self._resolve_path(field_path, step_outputs)
                try:
                    target_values = yaml.safe_load(values_str)
                except Exception:
                    target_values = []
                if isinstance(actual, list) and isinstance(target_values, list):
                    return any(v in actual for v in target_values)
            return False

        # Handle "in [...]" / "in ['...', '...']"
        if " in " in clause and "[" in clause:
            parts = clause.split(" in ", 1)
            if len(parts) == 2:
                field_path = parts[0].strip()
                values_str = parts[1].strip()
                actual = self._resolve_path(field_path, step_outputs)
                try:
                    target_values = yaml.safe_load(values_str)
                except Exception:
                    target_values = []
                if isinstance(target_values, list):
                    return actual in target_values
            return False

        # Handle comparisons: ==, !=
        for op in ["!=", "=="]:
            if op in clause:
                parts = clause.split(op, 1)
                if len(parts) == 2:
                    field_path = parts[0].strip()
                    expected = parts[1].strip().strip("'\"")
                    actual = self._resolve_path(field_path, step_outputs)
                    # Type coercion
                    if expected.lower() == "true":
                        expected = True
                    elif expected.lower() == "false":
                        expected = False
                    elif expected.isdigit():
                        expected = int(expected)

                    if op == "==":
                        return actual == expected
                    else:
                        return actual != expected
                break

        return False

    def _resolve_path(self, path: str, step_outputs: dict[str, dict]) -> Any:
        """Resolve 'step_name.field' from step outputs."""
        parts = path.strip().split(".")
        if len(parts) < 2:
            return None
        step_name = parts[0]
        step_data = step_outputs.get(step_name, {})
        obj = step_data
        for part in parts[1:]:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        return obj

    def _check_validation(self, validation: str, artifact: dict) -> str | None:
        """Check a simple validation rule against an artifact."""
        v = validation.strip()

        # "field in enum_name"
        if " in " in v and "[" not in v:
            parts = v.split(" in ", 1)
            if len(parts) == 2:
                field_name = parts[0].strip()
                enum_name = parts[1].strip()
                actual = artifact.get(field_name)
                enum_def = self.enums.get(enum_name)
                if enum_def and actual and actual not in enum_def.values:
                    return f"{field_name}={actual} not in {enum_name}"
        return None

    # ═══════════════════════════════════════════════════════════════
    # Full Run Validation
    # ═══════════════════════════════════════════════════════════════

    def validate_run(self, step_outputs: dict[str, dict]) -> list[Violation]:
        """Run all contract validations against a completed workflow."""
        violations = []

        # I0: Retrieve completeness
        for step_name, output in step_outputs.items():
            spec = self.step_specs.get(step_name)
            if spec and spec.primitive == "retrieve":
                violations.extend(self.check_retrieve_completeness(output))

        # I3: Artifact schemas
        for step_name, output in step_outputs.items():
            spec = self.step_specs.get(step_name)
            if spec and spec.primitive == "generate":
                artifact = output.get("artifact")
                violations.extend(self.check_artifact(step_name, artifact))

        # I4: Confidence floors
        for step_name, output in step_outputs.items():
            confidence = output.get("confidence")
            violations.extend(self.check_confidence(step_name, confidence))

        # I7: Cross-step consistency
        violations.extend(self.check_consistency(step_outputs))

        return violations

    @property
    def has_contracts(self) -> bool:
        """True if this domain has any machine-validatable contracts."""
        return bool(
            self.tool_contracts
            or self.artifact_contracts
            or self.consistency_rules
            or any(s.confidence_floor is not None for s in self.step_specs.values())
        )

    def summary(self) -> str:
        """Human-readable summary of what this contract covers."""
        parts = [f"Domain: {self.domain_name}"]
        if self.tool_contracts:
            parts.append(f"  Tools: {len(self.tool_contracts)} contracts")
        if self.enums:
            parts.append(f"  Enums: {len(self.enums)} definitions")
        if self.artifact_contracts:
            parts.append(f"  Artifacts: {len(self.artifact_contracts)} schemas")
        if self.consistency_rules:
            parts.append(f"  Consistency: {len(self.consistency_rules)} rules")
        floors = [(n, s.confidence_floor) for n, s in self.step_specs.items()
                  if s.confidence_floor is not None]
        if floors:
            parts.append(f"  Confidence floors: {len(floors)} steps")
        return "\n".join(parts)


def _deep_contains(obj: Any, key: str) -> bool:
    """Check if a key appears anywhere in a nested dict/list structure."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_deep_contains(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_deep_contains(v, key) for v in obj)
    if isinstance(obj, str):
        return key.lower() in obj.lower()
    return False
