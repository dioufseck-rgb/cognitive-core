"""
Cognitive Core — Tier Escalation Invariant (H-001)

Enforces upward-only governance tier resolution as a hard invariant.
No code path can produce an effective tier lower than the declared tier.

The tier ordering is: auto < spot_check < gate < hold
This function is the single point through which all tier decisions flow.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cognitive_core.tier")

# Canonical tier ordering — higher number = more restrictive
TIER_ORDER = {
    "auto": 0,
    "spot_check": 1,
    "gate": 2,
    "hold": 3,
}

TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}


class TierInvariantViolation(Exception):
    """Raised when code attempts to downgrade a governance tier."""
    pass


def resolve_effective_tier(
    declared_tier: str,
    *override_candidates: str,
    source_labels: list[str] | None = None,
) -> tuple[str, str]:
    """
    Resolve the effective governance tier from declared + overrides.

    INVARIANT: effective_tier >= declared_tier (always).
    No override can produce a downward result.

    Args:
        declared_tier: The tier from the domain YAML
        *override_candidates: Additional tier values from circuit breakers,
            kill switches, config reload, etc.
        source_labels: Optional labels for each override (for logging)

    Returns:
        (effective_tier, override_source) where override_source is
        "declared" if no override raised it, or the label of the
        winning override.
    """
    declared_val = TIER_ORDER.get(declared_tier)
    if declared_val is None:
        raise ValueError(f"Unknown tier: {declared_tier!r}. Valid: {list(TIER_ORDER.keys())}")

    best_val = declared_val
    best_source = "declared"

    labels = source_labels or [f"override_{i}" for i in range(len(override_candidates))]

    for i, candidate in enumerate(override_candidates):
        if candidate is None or candidate == "":
            continue
        cand_val = TIER_ORDER.get(candidate)
        if cand_val is None:
            logger.warning("Ignoring unknown tier override: %r from %s", candidate, labels[i] if i < len(labels) else "unknown")
            continue
        if cand_val > best_val:
            best_val = cand_val
            best_source = labels[i] if i < len(labels) else f"override_{i}"

    effective_tier = TIER_NAMES[best_val]

    # Hard invariant check — should be mathematically impossible to violate
    # given the max() logic above, but defense in depth
    if TIER_ORDER[effective_tier] < declared_val:
        raise TierInvariantViolation(
            f"INVARIANT VIOLATION: effective_tier={effective_tier} "
            f"< declared_tier={declared_tier}. This should never happen."
        )

    if effective_tier != declared_tier:
        logger.info(
            "Tier escalated: declared=%s → effective=%s (source: %s)",
            declared_tier, effective_tier, best_source,
        )

    return effective_tier, best_source


def validate_tier(tier: str) -> bool:
    """Check if a tier name is valid."""
    return tier in TIER_ORDER


def tier_at_least(tier: str, minimum: str) -> bool:
    """Check if tier meets or exceeds the minimum."""
    return TIER_ORDER.get(tier, -1) >= TIER_ORDER.get(minimum, -1)
