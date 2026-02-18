"""
Cognitive Core — Immutable Audit Trail

Append-only event store that records every automated decision for
regulatory reconstruction. Separate database/schema from coordinator
(Design: Option B).

Features:
  - Append-only: no UPDATE, no DELETE exposed
  - SHA-256 hash chain: each event includes hash of previous event
  - Tamper detection: verify chain integrity on demand
  - Query by trace_id for full decision reconstruction

Events recorded:
  - primitive_complete: classification, investigation, generation outputs
  - governance_decision: tier evaluation, quality gate results
  - escalation: escalation trigger, reason, routing
  - delegation: parent→child dispatch and completion

Usage:
    trail = AuditTrail("audit.db")
    trail.record_primitive(trace_id, step, primitive, output, model, prompt_hash, confidence)
    trail.record_governance(trace_id, domain, declared_tier, applied_tier, gate_result)

    events = trail.get_trail(trace_id)
    integrity = trail.verify_chain()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.audit")


# ═══════════════════════════════════════════════════════════════════
# Audit Event
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    """A single audit trail entry."""
    id: int
    trace_id: str
    event_type: str
    timestamp: float
    payload: dict[str, Any]
    event_hash: str
    previous_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "event_hash": self.event_hash,
            "previous_hash": self.previous_hash,
        }


# ═══════════════════════════════════════════════════════════════════
# Hash Chain
# ═══════════════════════════════════════════════════════════════════

GENESIS_HASH = "0" * 64  # SHA-256 of nothing — chain anchor


def compute_event_hash(
    previous_hash: str,
    trace_id: str,
    event_type: str,
    timestamp: float,
    payload_json: str,
) -> str:
    """Compute SHA-256 hash for an audit event."""
    content = f"{previous_hash}|{trace_id}|{event_type}|{timestamp}|{payload_json}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════
# Audit Trail Store
# ═══════════════════════════════════════════════════════════════════

class AuditTrail:
    """
    Append-only audit event store with hash chain integrity.

    Uses a SEPARATE SQLite database from the coordinator store
    (Design Option B: isolated audit writes).
    """

    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                payload TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                previous_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_audit_trace
                ON audit_events(trace_id);
            CREATE INDEX IF NOT EXISTS idx_audit_type
                ON audit_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_events(timestamp);
        """)
        self._conn.commit()

    def _get_last_hash(self) -> str:
        """Get the hash of the most recent event, or GENESIS_HASH."""
        row = self._conn.execute(
            "SELECT event_hash FROM audit_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_HASH

    def _append(self, trace_id: str, event_type: str, payload: dict[str, Any]) -> AuditEvent:
        """Append a single event to the audit trail."""
        with self._lock:
            timestamp = time.time()
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            previous_hash = self._get_last_hash()

            event_hash = compute_event_hash(
                previous_hash, trace_id, event_type, timestamp, payload_json
            )

            cursor = self._conn.execute(
                """INSERT INTO audit_events
                   (trace_id, event_type, timestamp, payload, event_hash, previous_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (trace_id, event_type, timestamp, payload_json, event_hash, previous_hash),
            )
            self._conn.commit()

            return AuditEvent(
                id=cursor.lastrowid,
                trace_id=trace_id,
                event_type=event_type,
                timestamp=timestamp,
                payload=payload,
                event_hash=event_hash,
                previous_hash=previous_hash,
            )

    # ── Recording Methods ───────────────────────────────────────

    def record_primitive(
        self,
        trace_id: str,
        step_name: str,
        primitive: str,
        output_hash: str,
        model_version: str,
        prompt_hash: str,
        confidence: float | None = None,
        decision: str = "",
    ) -> AuditEvent:
        """Record a primitive completion (classify, investigate, generate, etc.)."""
        payload = {
            "step_name": step_name,
            "primitive": primitive,
            "output_hash": output_hash,
            "model_version": model_version,
            "prompt_hash": prompt_hash,
        }
        if confidence is not None:
            payload["confidence"] = confidence
        if decision:
            payload["decision"] = decision
        return self._append(trace_id, "primitive_complete", payload)

    def record_governance(
        self,
        trace_id: str,
        domain: str,
        declared_tier: str,
        applied_tier: str,
        quality_gate_result: str | None = None,
        reason: str = "",
    ) -> AuditEvent:
        """Record a governance tier decision."""
        payload = {
            "domain": domain,
            "declared_tier": declared_tier,
            "applied_tier": applied_tier,
        }
        if quality_gate_result:
            payload["quality_gate_result"] = quality_gate_result
        if reason:
            payload["reason"] = reason
        return self._append(trace_id, "governance_decision", payload)

    def record_escalation(
        self,
        trace_id: str,
        step_name: str,
        trigger: str,
        reason: str,
        routed_to: str = "",
    ) -> AuditEvent:
        """Record an escalation event."""
        return self._append(trace_id, "escalation", {
            "step_name": step_name,
            "trigger": trigger,
            "reason": reason,
            "routed_to": routed_to,
        })

    def record_delegation(
        self,
        trace_id: str,
        child_trace_id: str,
        parent_workflow: str,
        child_workflow: str,
        policy: str,
        status: str,
    ) -> AuditEvent:
        """Record a delegation event."""
        return self._append(trace_id, "delegation", {
            "child_trace_id": child_trace_id,
            "parent_workflow": parent_workflow,
            "child_workflow": child_workflow,
            "policy": policy,
            "status": status,
        })

    # ── Query Methods ───────────────────────────────────────────

    def get_trail(self, trace_id: str) -> list[AuditEvent]:
        """Get full audit trail for a trace_id, ordered by time."""
        rows = self._conn.execute(
            """SELECT id, trace_id, event_type, timestamp, payload,
                      event_hash, previous_hash
               FROM audit_events
               WHERE trace_id = ?
               ORDER BY id ASC""",
            (trace_id,),
        ).fetchall()
        return [
            AuditEvent(
                id=r[0], trace_id=r[1], event_type=r[2],
                timestamp=r[3], payload=json.loads(r[4]),
                event_hash=r[5], previous_hash=r[6],
            )
            for r in rows
        ]

    def get_events_by_type(self, event_type: str, limit: int = 100) -> list[AuditEvent]:
        """Get recent events of a specific type."""
        rows = self._conn.execute(
            """SELECT id, trace_id, event_type, timestamp, payload,
                      event_hash, previous_hash
               FROM audit_events
               WHERE event_type = ?
               ORDER BY id DESC LIMIT ?""",
            (event_type, limit),
        ).fetchall()
        return [
            AuditEvent(
                id=r[0], trace_id=r[1], event_type=r[2],
                timestamp=r[3], payload=json.loads(r[4]),
                event_hash=r[5], previous_hash=r[6],
            )
            for r in rows
        ]

    def count_events(self) -> int:
        """Total event count."""
        row = self._conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        return row[0]

    # ── Integrity Verification ──────────────────────────────────

    def verify_chain(self, limit: int | None = None) -> tuple[bool, str]:
        """
        Verify the hash chain integrity of the entire audit trail.

        Returns (is_valid, message).
        """
        query = "SELECT id, trace_id, event_type, timestamp, payload, event_hash, previous_hash FROM audit_events ORDER BY id ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        rows = self._conn.execute(query).fetchall()
        if not rows:
            return True, "Empty trail — nothing to verify"

        expected_prev = GENESIS_HASH
        for row in rows:
            event_id, trace_id, event_type, timestamp, payload_json, stored_hash, stored_prev = row

            # Check previous hash linkage
            if stored_prev != expected_prev:
                return False, (
                    f"Chain broken at event {event_id}: "
                    f"expected previous_hash={expected_prev[:16]}..., "
                    f"got {stored_prev[:16]}..."
                )

            # Recompute hash
            computed = compute_event_hash(
                stored_prev, trace_id, event_type, timestamp, payload_json
            )
            if computed != stored_hash:
                return False, (
                    f"Tampered event {event_id}: "
                    f"computed hash={computed[:16]}..., "
                    f"stored hash={stored_hash[:16]}..."
                )

            expected_prev = stored_hash

        return True, f"Chain verified: {len(rows)} events, integrity intact"

    def close(self):
        self._conn.close()
