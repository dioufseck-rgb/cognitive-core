"""
Cognitive Core - Primitive Registry

Central registry that maps primitive names to their base prompts,
output schemas, and configuration requirements. New primitives are
registered here; new use cases consume from here.
"""

from pathlib import Path
from typing import Any

from registry.schemas import (
    SCHEMA_REGISTRY,
    BaseOutput,
    ClassifyOutput,
    InvestigateOutput,
    VerifyOutput,
    GenerateOutput,
    ChallengeOutput,
    RetrieveOutput,
    ThinkOutput,
    ActOutput,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Primitive configuration specs — what each primitive requires
# ---------------------------------------------------------------------------

PRIMITIVE_CONFIGS: dict[str, dict[str, Any]] = {
    "classify": {
        "required_params": ["categories", "criteria"],
        "optional_params": ["confidence_threshold", "additional_instructions"],
        "defaults": {
            "confidence_threshold": "0.7",
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "classify.txt",
        "schema": ClassifyOutput,
    },
    "investigate": {
        "required_params": ["question", "scope"],
        "optional_params": ["effort_level", "available_evidence", "additional_instructions"],
        "defaults": {
            "effort_level": "moderate",
            "available_evidence": "See context above.",
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "investigate.txt",
        "schema": InvestigateOutput,
    },
    "verify": {
        "required_params": ["rules"],
        "optional_params": ["additional_instructions"],
        "defaults": {
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "verify.txt",
        "schema": VerifyOutput,
    },
    "generate": {
        "required_params": ["requirements", "format", "constraints"],
        "optional_params": ["additional_instructions"],
        "defaults": {
            "format": "text",
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "generate.txt",
        "schema": GenerateOutput,
    },
    "challenge": {
        "required_params": ["perspective", "threat_model"],
        "optional_params": ["additional_instructions"],
        "defaults": {
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "challenge.txt",
        "schema": ChallengeOutput,
    },
    "retrieve": {
        "required_params": ["specification"],
        "optional_params": ["sources", "strategy", "additional_instructions"],
        "defaults": {
            "strategy": "deterministic",
            "sources": "Will be populated from tool registry at runtime.",
            "additional_instructions": "",
            "context": "No additional context provided.",
            "source_results": "No sources queried yet.",
        },
        "prompt_file": "retrieve.txt",
        "schema": RetrieveOutput,
    },
    "think": {
        "required_params": ["instruction"],
        "optional_params": ["focus", "additional_instructions"],
        "defaults": {
            "focus": "Consider all available evidence and prior step outputs.",
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "think.txt",
        "schema": ThinkOutput,
    },
    "act": {
        "required_params": ["actions", "authorization"],
        "optional_params": ["mode", "rollback_strategy", "additional_instructions"],
        "defaults": {
            "mode": "dry_run",
            "rollback_strategy": "Reverse all actions in LIFO order if any action fails.",
            "additional_instructions": "",
            "context": "No additional context provided.",
        },
        "prompt_file": "act.txt",
        "schema": ActOutput,
    },
}


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------

def list_primitives() -> list[str]:
    """List all available primitive names."""
    return list(PRIMITIVE_CONFIGS.keys())


def get_prompt_template(primitive_name: str) -> str:
    """Load the base prompt template for a primitive."""
    name = primitive_name.lower()
    if name not in PRIMITIVE_CONFIGS:
        raise ValueError(f"Unknown primitive: {name}. Available: {list_primitives()}")

    prompt_file = PROMPTS_DIR / PRIMITIVE_CONFIGS[name]["prompt_file"]
    return prompt_file.read_text()


def get_schema_class(primitive_name: str) -> type[BaseOutput]:
    """Get the Pydantic output schema class for a primitive."""
    name = primitive_name.lower()
    if name not in PRIMITIVE_CONFIGS:
        raise ValueError(f"Unknown primitive: {name}. Available: {list_primitives()}")
    return PRIMITIVE_CONFIGS[name]["schema"]


def get_config_spec(primitive_name: str) -> dict[str, Any]:
    """Get the configuration specification for a primitive."""
    name = primitive_name.lower()
    if name not in PRIMITIVE_CONFIGS:
        raise ValueError(f"Unknown primitive: {name}. Available: {list_primitives()}")
    return PRIMITIVE_CONFIGS[name]


def render_prompt(primitive_name: str, params: dict[str, str]) -> str:
    """
    Render a primitive's prompt template with the given parameters.

    Merges provided params with defaults, validates required params,
    and returns the fully rendered prompt string.
    """
    name = primitive_name.lower()
    config = get_config_spec(name)
    template = get_prompt_template(name)

    # Merge defaults with provided params
    merged = {**config["defaults"], **params}

    # Validate required params
    missing = [p for p in config["required_params"] if p not in merged or not merged[p]]
    if missing:
        raise ValueError(
            f"Primitive '{name}' requires parameters: {missing}. "
            f"Provided: {list(params.keys())}"
        )

    # Render template
    try:
        rendered = template.format(**merged)
    except KeyError as e:
        raise ValueError(
            f"Prompt template for '{name}' expects parameter {e} "
            f"which was not provided. Available: {list(merged.keys())}"
        )

    return rendered


def validate_use_case_step(step: dict) -> list[str]:
    """
    Validate a use case step configuration.
    Returns a list of error messages (empty if valid).
    """
    errors = []

    if "primitive" not in step:
        errors.append("Step missing 'primitive' field")
        return errors

    name = step["primitive"].lower()
    if name not in PRIMITIVE_CONFIGS:
        errors.append(f"Unknown primitive: {name}")
        return errors

    config = get_config_spec(name)
    params = step.get("params", {})

    for req in config["required_params"]:
        # Allow params to reference state with ${...} syntax
        if req not in params:
            errors.append(f"Step '{step.get('name', name)}' missing required param: {req}")

    return errors
