"""
Cognitive Core — PII Redaction Layer

Redacts PII from LLM prompts before API calls, de-redacts in responses.
Designed to sit at the create_llm_node() chokepoint so no PII escapes.

Strategy: Hybrid (Option C)
  - Regex patterns: SSN, account numbers, email, phone, DOB
  - Case-entity mapping: names/addresses from case input mapped to placeholders
  - No ML dependency, deterministic, auditable

Integration: create_llm_node() chokepoint (Option A)
  - Redaction happens in one place
  - Impossible to accidentally send unredacted PII

Usage:
    redactor = PiiRedactor()
    redactor.register_entities_from_case(case_input)  # learn names, addresses
    redacted_prompt = redactor.redact(prompt)
    # ... send to LLM ...
    final_output = redactor.deredact(llm_response)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.pii")


# ═══════════════════════════════════════════════════════════════════
# PII Patterns (Regex)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PiiPattern:
    """A regex pattern that detects a PII type."""
    name: str
    pattern: re.Pattern
    placeholder_prefix: str


# Patterns ordered by specificity (most specific first)
PII_PATTERNS = [
    PiiPattern(
        name="ssn",
        pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        placeholder_prefix="SSN",
    ),
    PiiPattern(
        name="email",
        pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        placeholder_prefix="EMAIL",
    ),
    PiiPattern(
        name="phone",
        pattern=re.compile(
            r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ),
        placeholder_prefix="PHONE",
    ),
    PiiPattern(
        name="credit_card",
        pattern=re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        placeholder_prefix="CC",
    ),
    PiiPattern(
        name="account_number",
        # 10-16 digit numbers that aren't SSNs or phone numbers
        pattern=re.compile(r'\b\d{10,16}\b'),
        placeholder_prefix="ACCT",
    ),
    PiiPattern(
        name="dob",
        pattern=re.compile(
            r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'
        ),
        placeholder_prefix="DOB",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# Redaction Map
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RedactionMap:
    """Bidirectional mapping between PII values and placeholders."""
    _forward: dict[str, str] = field(default_factory=dict)   # original → placeholder
    _reverse: dict[str, str] = field(default_factory=dict)   # placeholder → original
    _counters: dict[str, int] = field(default_factory=dict)  # prefix → next number

    def get_or_create(self, original: str, prefix: str) -> str:
        """Get existing placeholder or create a new one."""
        if original in self._forward:
            return self._forward[original]

        count = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = count
        placeholder = f"[{prefix}_{count}]"

        self._forward[original] = placeholder
        self._reverse[placeholder] = original
        return placeholder

    def deredact(self, text: str) -> str:
        """Replace all placeholders with original values."""
        result = text
        # Sort by length descending to avoid partial replacements
        for placeholder, original in sorted(
            self._reverse.items(), key=lambda x: len(x[0]), reverse=True
        ):
            result = result.replace(placeholder, original)
        return result

    @property
    def redaction_count(self) -> int:
        return len(self._forward)

    @property
    def summary(self) -> dict[str, int]:
        """Count of redactions by type (for audit log)."""
        counts: dict[str, int] = {}
        for placeholder in self._reverse:
            prefix = placeholder.split("_")[0].strip("[")
            counts[prefix] = counts.get(prefix, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════
# PII Redactor
# ═══════════════════════════════════════════════════════════════════

class PiiRedactor:
    """
    Redacts PII from text using regex patterns + case-entity mapping.

    Lifecycle:
      1. Create redactor (once per workflow execution)
      2. register_entities_from_case() — learn names/addresses from case input
      3. redact() — mask PII in prompt before LLM call
      4. deredact() — restore PII in LLM response
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._map = RedactionMap()
        self._entity_patterns: list[tuple[str, str]] = []  # (original, prefix)

    def register_entities_from_case(self, case_input: dict[str, Any]) -> None:
        """
        Extract names and addresses from case input and register them.

        Scans all string values in the case input recursively. Registers
        values that look like names (2-4 word sequences starting with caps)
        or addresses (contain street/city/state patterns).
        """
        if not self.enabled:
            return

        strings = _extract_strings(case_input)

        for key, value in strings:
            key_lower = key.lower()

            # Name fields
            if any(term in key_lower for term in [
                "name", "customer_name", "member_name", "account_holder",
                "cardholder", "first_name", "last_name", "full_name",
            ]):
                if value and len(value) > 1:
                    self._register_entity(value, "NAME")

            # Address fields
            elif any(term in key_lower for term in [
                "address", "street", "city", "zip", "postal",
            ]):
                if value and len(value) > 1:
                    self._register_entity(value, "ADDR")

            # Membership / account IDs
            elif any(term in key_lower for term in [
                "member_id", "account_id", "customer_id",
            ]):
                if value and len(value) > 1:
                    self._register_entity(value, "ID")

    def _register_entity(self, value: str, prefix: str) -> None:
        """Register an entity value for redaction."""
        self._entity_patterns.append((value, prefix))
        # Also register individual name parts for partial matches
        if prefix == "NAME" and " " in value:
            parts = value.split()
            for part in parts:
                if len(part) > 2:  # Skip initials
                    self._entity_patterns.append((part, "NAME"))

    def redact(self, text: str) -> str:
        """
        Redact all PII from text. Returns redacted text.

        Order:
          1. Entity patterns (longest first to avoid partial matches)
          2. Regex patterns
        """
        if not self.enabled:
            return text

        result = text

        # Entity patterns — longest first
        sorted_entities = sorted(
            self._entity_patterns, key=lambda x: len(x[0]), reverse=True
        )
        for original, prefix in sorted_entities:
            if original in result:
                placeholder = self._map.get_or_create(original, prefix)
                result = result.replace(original, placeholder)

        # Regex patterns
        for pp in PII_PATTERNS:
            def _replacer(match, _pp=pp):
                return self._map.get_or_create(match.group(0), _pp.placeholder_prefix)
            result = pp.pattern.sub(_replacer, result)

        return result

    def deredact(self, text: str) -> str:
        """Restore all placeholders to original PII values."""
        if not self.enabled:
            return text
        return self._map.deredact(text)

    @property
    def redaction_count(self) -> int:
        """Total number of unique PII values redacted."""
        return self._map.redaction_count

    @property
    def audit_summary(self) -> dict[str, Any]:
        """Audit-safe summary (no actual PII)."""
        return {
            "total_redacted": self._map.redaction_count,
            "by_type": self._map.summary,
            "enabled": self.enabled,
        }


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_strings(
    obj: Any,
    prefix: str = "",
    results: list | None = None,
) -> list[tuple[str, str]]:
    """Recursively extract (key, string_value) pairs from nested dicts/lists."""
    if results is None:
        results = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            _extract_strings(value, full_key, results)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _extract_strings(item, f"{prefix}[{i}]", results)
    elif isinstance(obj, str) and obj.strip():
        results.append((prefix, obj.strip()))

    return results


def create_redactor_from_domain(domain_config: dict[str, Any]) -> PiiRedactor:
    """
    Create a PiiRedactor based on domain configuration.

    If domain has pii_redaction: false, returns disabled redactor.
    """
    enabled = domain_config.get("pii_redaction", True)
    return PiiRedactor(enabled=enabled)
