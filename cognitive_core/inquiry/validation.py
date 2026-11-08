"""Bounded validation feedback derived from the trusted response schema.

Pydantic's messages, contexts, and location labels can contain model-generated
values. Repair prompts and checkpoints must not echo those values. This module
keeps only schema-owned field names, bounded list indexes, known error codes,
and messages built from constants or trusted schema constraints.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import BaseModel, ValidationError


_MAX_ISSUES = 8
_MAX_DEPTH = 16
_MAX_NODES = 32
_MAX_PATH = 220

_MESSAGES = {
    "missing": "This field is required.",
    "extra_forbidden": "Remove fields not declared for this object.",
    "list_type": "Use an array.",
    "tuple_type": "Use an array matching the tuple schema.",
    "dict_type": "Use an object.",
    "model_type": "Use an object matching the schema.",
    "model_attributes_type": "Use an object matching the schema.",
    "string_type": "Use a string.",
    "string_too_short": "The string is shorter than the schema permits.",
    "string_too_long": "The string is longer than the schema permits.",
    "string_pattern_mismatch": "Use a string matching the schema's required pattern.",
    "int_type": "Use an integer.",
    "int_parsing": "Use an integer.",
    "int_from_float": "Use an integer without a fractional part.",
    "float_type": "Use a number.",
    "float_parsing": "Use a number.",
    "bool_type": "Use a boolean.",
    "bool_parsing": "Use a boolean.",
    "none_required": "Use null.",
    "finite_number": "Use a finite number.",
    "literal_error": "Use one of the values allowed by the schema.",
    "enum": "Use one of the values allowed by the schema.",
    "union_tag_invalid": "Choose a valid discriminator value from the schema.",
    "union_tag_not_found": "Provide the discriminator field required by the schema.",
    "greater_than": "Use a value above the schema's lower bound.",
    "greater_than_equal": "Use a value at or above the schema's lower bound.",
    "less_than": "Use a value below the schema's upper bound.",
    "less_than_equal": "Use a value at or below the schema's upper bound.",
    "too_short": "The collection is shorter than the schema permits.",
    "too_long": "The collection is longer than the schema permits.",
    "value_error": "The value violates a validation constraint; review the schema and task requirements.",
    "assertion_error": "The value violates a validation constraint; review the schema and task requirements.",
    "url_type": "Use a URL string matching the schema.",
    "url_parsing": "Use a valid URL matching the schema.",
    "url_scheme": "Use a URL scheme permitted by the schema.",
    "invalid": "The value does not satisfy the response schema.",
}


def _resolve(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Resolve only local schema references, without following reference loops."""
    seen = set()
    for _ in range(_MAX_DEPTH):
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return node
        seen.add(ref)
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(target, dict) else None
        if not isinstance(target, dict):
            return node
        node = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
    return node


def _views(nodes, root):
    """Yield union wrappers and their alternatives, with bounded traversal."""
    pending = [(node, 0) for node in nodes[:_MAX_NODES]]
    yielded = 0
    while pending and yielded < _MAX_NODES:
        node, depth = pending.pop(0)
        node = _resolve(node, root)
        yield node
        yielded += 1
        if depth < _MAX_DEPTH:
            choices = node.get("oneOf", node.get("anyOf", []))
            if isinstance(choices, list):
                pending.extend((choice, depth + 1) for choice in choices[:_MAX_NODES]
                               if isinstance(choice, dict))


def _field_path(parts):
    result = ""
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]" if 0 <= part <= 9999 else "[*]"
        elif part is None:
            result += "[*]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            result += ("." if result else "") + part
        else:
            result += "[" + json.dumps(part, ensure_ascii=True) + "]"
    result = result or "$"
    return result if len(result) <= _MAX_PATH else result[:_MAX_PATH] + "..."


def _locate(location, root):
    nodes = [root]
    parts = []
    for token in location[:_MAX_DEPTH]:
        views = list(_views(nodes, root))
        # Pydantic inserts discriminator values or model names into union
        # locations. Recognize only labels declared by this union's schema;
        # they select a branch but are not part of the displayed field path.
        selected = []
        if isinstance(token, str):
            for node in views:
                choices = node.get("oneOf", node.get("anyOf", []))
                if not isinstance(choices, list) or not choices:
                    continue
                discriminator = node.get("discriminator", {})
                mapping = discriminator.get("mapping", {}) if isinstance(discriminator, dict) else {}
                if token in mapping and isinstance(mapping[token], str):
                    selected.append({"$ref": mapping[token]})
                elif not mapping:
                    for choice in choices:
                        if isinstance(choice, dict) and _resolve(choice, root).get("title") == token:
                            selected.append(choice)
            if selected:
                nodes = selected[:_MAX_NODES]
                continue
        matched = []
        if type(token) is int:
            for node in views:
                prefixes = node.get("prefixItems", [])
                if isinstance(prefixes, list) and 0 <= token < len(prefixes) and isinstance(prefixes[token], dict):
                    matched.append(prefixes[token])
                elif isinstance(node.get("items"), dict):
                    matched.append(node["items"])
            if matched:
                parts.append(token)
        elif isinstance(token, str):
            for node in views:
                properties = node.get("properties", {})
                if isinstance(properties, dict) and token in properties and isinstance(properties[token], dict):
                    matched.append(properties[token])
            if matched:
                parts.append(token)  # Membership above makes this schema-owned.
        if not matched:
            # Dynamic dictionary keys are legitimate locations, but their text
            # belongs to the model output. Represent them with a wildcard.
            matched = [node["additionalProperties"] for node in views
                       if isinstance(node.get("additionalProperties"), dict)]
            if matched:
                parts.append(None)
            else:
                break
        nodes = matched[:_MAX_NODES]
    return parts, list(_views(nodes, root))


def _constraint(nodes, name):
    values = [node[name] for node in nodes if name in node]
    if not values or any(value != values[0] for value in values[1:]):
        return None
    value = values[0]
    if type(value) not in (int, float) or abs(value) > 1e12 or not math.isfinite(value):
        return None
    return value


def _message(code, nodes):
    constraints = {
        "too_long": ("maxItems", "Must contain at most {} items."),
        "too_short": ("minItems", "Must contain at least {} items."),
        "string_too_long": ("maxLength", "Use at most {} characters."),
        "string_too_short": ("minLength", "Use at least {} characters."),
        "greater_than": ("exclusiveMinimum", "Use a number greater than {}."),
        "greater_than_equal": ("minimum", "Use a number greater than or equal to {}."),
        "less_than": ("exclusiveMaximum", "Use a number less than {}."),
        "less_than_equal": ("maximum", "Use a number less than or equal to {}."),
    }
    if code in constraints:
        name, template = constraints[code]
        value = _constraint(nodes, name)
        if value is not None:
            return template.format(value)
    return _MESSAGES[code]


def validation_issues(error: ValidationError, response_model: type[BaseModel]) -> list[dict[str, str]]:
    """Describe failures without copying model values, unknown keys, or contexts.

    The returned dictionaries contain only path, code, and message strings.
    Unknown error kinds and unresolvable locations receive generic diagnostics;
    the caller must still reject the invalid response using its original schema.
    """
    try:
        schema = response_model.model_json_schema()
    except Exception:
        return [{"path": "$", "code": "invalid", "message": _MESSAGES["invalid"]}]
    issues = []
    seen = set()
    for entry in error.errors(include_url=False, include_input=False, include_context=False)[:64]:
        code = entry["type"] if entry["type"] in _MESSAGES else "invalid"
        parts, nodes = _locate(entry.get("loc", ()), schema)
        message = _message(code, nodes)
        if code in {"union_tag_invalid", "union_tag_not_found"}:
            for node in nodes:
                discriminator = node.get("discriminator", {})
                if not isinstance(discriminator, dict):
                    continue
                name, mapping = discriminator.get("propertyName"), discriminator.get("mapping")
                if isinstance(name, str) and isinstance(mapping, dict) and mapping:
                    parts.append(name)
                    choices = [key for key in mapping if isinstance(key, str)]
                    if len(choices) <= 8:
                        options = ", ".join(json.dumps(key, ensure_ascii=True) for key in choices)
                        if len(options) <= 220:
                            message = "Use one of the schema's discriminator values: " + options + "."
                    break
        issue = {"path": _field_path(parts), "code": code, "message": message[:300]}
        identity = tuple(issue.values())
        if identity not in seen:
            issues.append(issue)
            seen.add(identity)
        if len(issues) >= _MAX_ISSUES:
            break
    return issues or [{"path": "$", "code": "invalid", "message": _MESSAGES["invalid"]}]
