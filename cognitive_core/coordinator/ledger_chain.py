"""
Cognitive Core — Ledger Hash Chain (Sprint 4.1)

Each ledger entry hashes the prior entry. Modification of any record
is detectable. Turns "append-only by convention" into "append-only by
verification."

Spec:
  - Each entry: entry_hash = sha256(prior_hash + canonical_content)
  - Genesis entry hashes a fixed GENESIS constant
  - verify_ledger(instance_id) → {valid: bool, first_invalid_entry: int | None}
  - Exposed via GET /instances/{id}/verify
  - HTML trace page shows ✓ verified / ✗ tampered

This module provides:
  - compute_entry_hash()       — deterministic hash for one ledger row
  - chain_hash()               — apply hash chain to a sequence of entries
  - verify_ledger()            — verify chain integrity for an instance
  - patch_store_for_hashing()  — monkey-patch CoordinatorStore.log_action
                                  to write entry_hash automatically

Usage:
    from cognitive_core.coordinator.ledger_chain import verify_ledger, patch_store_for_hashing

    patch_store_for_hashing(store)   # call once at startup
    result = verify_ledger(store, instance_id="wf_abc123")
    # result = {"valid": True, "first_invalid_entry": None, "entries_checked": 12}
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

# The genesis constant — every chain starts by hashing this
GENESIS_CONSTANT = "cognitive-core-genesis-v1"


def _canonical(content: dict[str, Any]) -> bytes:
    """
    Deterministic canonical encoding of a ledger entry's content.

    Uses sorted keys + no extra whitespace so the hash is reproducible
    across Python versions and dict ordering.
    """
    return json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def compute_entry_hash(prior_hash: str, entry_content: dict[str, Any]) -> str:
    """
    Compute the hash for one ledger entry.

    hash = sha256( prior_hash_hex + canonical_json(content) )
    """
    payload = prior_hash.encode("utf-8") + _canonical(entry_content)
    return hashlib.sha256(payload).hexdigest()


def _entry_canonical_content(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract the stable fields of a ledger entry for hashing."""
    return {
        "id":              entry.get("id"),
        "instance_id":     entry.get("instance_id"),
        "correlation_id":  entry.get("correlation_id"),
        "action_type":     entry.get("action_type"),
        "details":         entry.get("details"),
        "idempotency_key": entry.get("idempotency_key"),
        "created_at":      entry.get("created_at"),
    }


def verify_ledger(store, instance_id: str) -> dict[str, Any]:
    """
    Verify the hash chain integrity for all ledger entries of an instance.

    Returns:
        {
          "valid": bool,
          "first_invalid_entry": int | None,   # ledger row id
          "entries_checked": int,
          "instance_id": str,
        }

    An entry is invalid if its stored entry_hash does not match the
    recomputed hash from prior_hash + content.
    """
    entries = store.get_ledger(instance_id=instance_id)

    if not entries:
        return {
            "valid": True,
            "first_invalid_entry": None,
            "entries_checked": 0,
            "instance_id": instance_id,
        }

    prior_hash = GENESIS_CONSTANT
    for i, entry in enumerate(entries):
        stored_hash = entry.get("entry_hash")
        content = _entry_canonical_content(entry)
        expected_hash = compute_entry_hash(prior_hash, content)

        if stored_hash is None:
            # Entry predates hash chain — treat as valid but skip
            prior_hash = expected_hash
            continue

        if stored_hash != expected_hash:
            return {
                "valid": False,
                "first_invalid_entry": entry.get("id"),
                "first_invalid_position": i,
                "entries_checked": i + 1,
                "instance_id": instance_id,
            }

        prior_hash = stored_hash

    return {
        "valid": True,
        "first_invalid_entry": None,
        "entries_checked": len(entries),
        "instance_id": instance_id,
    }


def verify_all_instances(store) -> dict[str, Any]:
    """
    Verify hash chain integrity across all instances.

    Returns summary with per-instance results and overall valid flag.
    """
    try:
        instances = store.list_instances(limit=10_000)
    except Exception:
        return {"valid": False, "error": "could not list instances"}

    results = []
    all_valid = True
    for inst in instances:
        r = verify_ledger(store, inst.instance_id)
        results.append(r)
        if not r["valid"]:
            all_valid = False

    return {
        "valid": all_valid,
        "instances_checked": len(results),
        "invalid_instances": [r["instance_id"] for r in results if not r["valid"]],
        "results": results,
    }


def patch_store_for_hashing(store) -> None:
    """
    Monkey-patch store.log_action to automatically compute and write
    entry_hash for every new ledger entry.

    Call once after store initialisation:
        patch_store_for_hashing(coordinator.store)

    Also runs a schema migration to add the entry_hash column if absent.
    """
    # ── Schema migration ──────────────────────────────────────────────
    try:
        store.db.execute(
            "ALTER TABLE action_ledger ADD COLUMN entry_hash TEXT"
        )
        store._commit()
    except Exception:
        pass  # Column already exists

    # ── Override log_action ──────────────────────────────────────────
    original_log_action = store.log_action

    def hashed_log_action(
        instance_id: str,
        correlation_id: str,
        action_type: str,
        details: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> bool:
        """log_action patched to write entry_hash."""
        # Get the current tail of the chain for this instance
        prior_hash = _get_prior_hash(store, instance_id)

        # We need the content we're about to insert to compute the hash.
        # Build the content dict exactly as it will be stored.
        created_at = time.time()
        entry_content = {
            "id":              None,   # autoincrement — not known yet; excluded from hash
            "instance_id":     instance_id,
            "correlation_id":  correlation_id,
            "action_type":     action_type,
            "details":         details,
            "idempotency_key": idempotency_key,
            "created_at":      created_at,
        }
        entry_hash = compute_entry_hash(prior_hash, entry_content)

        # Write directly (bypass original to control INSERT)
        try:
            store.db.execute("""
                INSERT INTO action_ledger
                (instance_id, correlation_id, action_type, details,
                 idempotency_key, created_at, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                instance_id, correlation_id, action_type,
                json.dumps(details, default=str),
                idempotency_key, created_at, entry_hash,
            ))
            store._commit()
            return True
        except Exception:
            # Idempotency key collision or other error — fall back to original
            return original_log_action(
                instance_id, correlation_id, action_type, details, idempotency_key
            )

    store.log_action = hashed_log_action


def _get_prior_hash(store, instance_id: str) -> str:
    """
    Fetch the entry_hash of the most recent ledger entry for this instance,
    or GENESIS_CONSTANT if no entries yet.
    """
    try:
        rows = store.db.fetchall(
            "SELECT entry_hash FROM action_ledger WHERE instance_id = ? ORDER BY id DESC LIMIT 1",
            (instance_id,)
        )
        if rows and rows[0].get("entry_hash"):
            return rows[0]["entry_hash"]
    except Exception:
        pass
    return GENESIS_CONSTANT
