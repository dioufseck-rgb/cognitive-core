"""
Cognitive Core — Case Input Validation (Sprint 2.1)

Pydantic schema validation at the case input boundary.
Clean, informative errors at the edge rather than cryptic failures
deep in workflow execution.

Scope:
  - case_input validated before coordinator.start() proceeds
  - Required fields checked, type errors caught, size limits enforced
  - Validation errors return a structured response with field + reason
  - Domain scaffold references to missing input fields caught at load time

Usage:
    from cognitive_core.engine.input_validation import validate_case_input, CaseInputError

    try:
        validate_case_input(case_input, workflow_type=workflow_type, domain_config=domain_cfg)
    except CaseInputError as e:
        # e.errors is a list of {field, reason, expected, received}
        raise
"""

from __future__ import annotations

import json
from typing import Any


# ── Max sizes ─────────────────────────────────────────────────────────
MAX_CASE_INPUT_BYTES = 512 * 1024   # 512 KB total
MAX_FIELD_BYTES      = 64 * 1024    # 64 KB per field value
MAX_STRING_LENGTH    = 32_768       # 32 K chars per string field


# ── Validation error ───────────────────────────────────────────────────

class CaseInputError(ValueError):
    """
    Raised when case input fails schema validation.

    Attributes:
        errors: list of {field, reason, expected, received}
    """
    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        msg = "; ".join(f"{e['field']}: {e['reason']}" for e in errors)
        super().__init__(f"Case input validation failed — {msg}")

    def to_dict(self) -> dict[str, Any]:
        return {"validation_errors": self.errors, "error": str(self)}


# ── Core validation logic ──────────────────────────────────────────────

def validate_case_input(
    case_input: Any,
    *,
    workflow_type: str = "",
    domain_config: dict[str, Any] | None = None,
) -> None:
    """
    Validate case_input before passing to coordinator.start().

    Raises CaseInputError with structured error list on failure.
    Raises nothing on success.
    """
    errors: list[dict[str, Any]] = []

    # ── T1: Must be a dict ──
    if not isinstance(case_input, dict):
        raise CaseInputError([{
            "field": "case_input",
            "reason": f"must be a mapping/dict, got {type(case_input).__name__}",
            "expected": "dict",
            "received": type(case_input).__name__,
        }])

    # ── T2: Must not be empty ──
    if not case_input:
        raise CaseInputError([{
            "field": "case_input",
            "reason": "case input must not be empty",
            "expected": "non-empty dict",
            "received": "{}",
        }])

    # ── T3: Payload size check ──
    try:
        raw_bytes = json.dumps(case_input, default=str).encode()
        if len(raw_bytes) > MAX_CASE_INPUT_BYTES:
            raise CaseInputError([{
                "field": "case_input",
                "reason": (
                    f"payload size {len(raw_bytes):,} bytes exceeds limit "
                    f"{MAX_CASE_INPUT_BYTES:,} bytes ({MAX_CASE_INPUT_BYTES // 1024} KB)"
                ),
                "expected": f"<= {MAX_CASE_INPUT_BYTES:,} bytes",
                "received": f"{len(raw_bytes):,} bytes",
            }])
    except (TypeError, ValueError):
        pass  # Non-serialisable — will surface later

    # ── T4: Key type check ──
    for key in case_input:
        if not isinstance(key, str):
            errors.append({
                "field": f"key:{key!r}",
                "reason": f"all keys must be strings, got {type(key).__name__}",
                "expected": "str",
                "received": type(key).__name__,
            })

    # ── T5: Per-field size and type checks ──
    for field, value in case_input.items():
        if not isinstance(field, str):
            continue  # Already caught above

        # String fields: length limit
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            errors.append({
                "field": field,
                "reason": (
                    f"string value length {len(value):,} chars exceeds limit "
                    f"{MAX_STRING_LENGTH:,} chars"
                ),
                "expected": f"<= {MAX_STRING_LENGTH:,} chars",
                "received": f"{len(value):,} chars",
            })

        # Per-field byte size
        try:
            fsize = len(json.dumps(value, default=str).encode())
            if fsize > MAX_FIELD_BYTES:
                errors.append({
                    "field": field,
                    "reason": (
                        f"field size {fsize:,} bytes exceeds limit "
                        f"{MAX_FIELD_BYTES:,} bytes ({MAX_FIELD_BYTES // 1024} KB)"
                    ),
                    "expected": f"<= {MAX_FIELD_BYTES:,} bytes",
                    "received": f"{fsize:,} bytes",
                })
        except (TypeError, ValueError):
            pass

    # ── T6: Domain scaffold required_inputs check ──
    if domain_config:
        errors.extend(_check_domain_required_inputs(case_input, domain_config))

    if errors:
        raise CaseInputError(errors)


def _check_domain_required_inputs(
    case_input: dict[str, Any],
    domain_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Check that all retrieve.specification sources referenced in the domain
    scaffold are present in the case_input.

    This surfaces missing inputs at load time, not mid-workflow.
    """
    errors: list[dict[str, Any]] = []

    # Domain scaffolds list their retrieve inputs under steps[].retrieve.specification
    steps = domain_config.get("steps", [])
    for step in steps:
        retrieve_spec = step.get("retrieve", {})
        if not retrieve_spec:
            # Check alternate structure: step primitive is "retrieve"
            if step.get("primitive") == "retrieve":
                retrieve_spec = step.get("specification", {})

        sources = retrieve_spec.get("sources", []) if isinstance(retrieve_spec, dict) else []
        for source in sources:
            source_name = source if isinstance(source, str) else source.get("name", "")
            if source_name and source_name not in case_input:
                # Only flag as error if the source is marked required
                required = source.get("required", False) if isinstance(source, dict) else False
                if required:
                    errors.append({
                        "field": source_name,
                        "reason": (
                            f"required retrieve source '{source_name}' not present in case_input "
                            f"(referenced by step '{step.get('name', '?')}')"
                        ),
                        "expected": "present in case_input",
                        "received": "missing",
                    })

    return errors


def validate_domain_scaffold_references(
    domain_config: dict[str, Any],
    workflow_config: dict[str, Any],
    domain_name: str = "",
) -> list[dict[str, Any]]:
    """
    Validate that all step names referenced in the domain scaffold
    exist in the workflow definition.

    Called at scaffold load time (not at runtime).
    Returns list of errors (empty = valid).
    """
    errors: list[dict[str, Any]] = []

    workflow_steps = {s.get("name") for s in workflow_config.get("steps", [])}
    domain_steps = domain_config.get("steps", [])

    for step in domain_steps:
        step_name = step.get("name", "")
        if step_name and step_name not in workflow_steps:
            errors.append({
                "field": f"domain.steps.{step_name}",
                "reason": (
                    f"domain scaffold references step '{step_name}' "
                    f"which does not exist in workflow '{workflow_config.get('name', '?')}'. "
                    f"Known steps: {sorted(workflow_steps)}"
                ),
                "expected": f"one of {sorted(workflow_steps)}",
                "received": step_name,
                "domain": domain_name,
            })

    return errors
