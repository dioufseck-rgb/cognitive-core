"""
Cognitive Core — Compensation Ledger (H-009)

Single-workflow compensation for Act primitive failures.
Before Act executes: register what it will do and how to reverse it.
On failure after Act: walk ledger backward and fire compensations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.compensation")


class CompensationStatus(str, Enum):
    PENDING = "pending"        # Registered, Act not yet executed
    CONFIRMED = "confirmed"    # Act succeeded
    COMPENSATED = "compensated"  # Compensation fired successfully
    FAILED = "failed"          # Compensation attempted but failed
    SKIPPED = "skipped"        # No handler registered — escalated to HITL


@dataclass
class CompensationEntry:
    id: int
    instance_id: str
    step_name: str
    idempotency_key: str
    action_description: str
    compensation_data: dict[str, Any]
    status: CompensationStatus
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class CompensationLedger:
    """
    Tracks Act executions and their compensation data.

    Flow:
    1. Before Act: register(instance_id, step, key, description, data)
    2. After Act success: confirm(key)
    3. On workflow failure post-Act: compensate(instance_id, handler_fn)
       - Walks entries in reverse order
       - Calls handler for each confirmed entry
       - Marks compensated or failed
    """

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS compensations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                action_description TEXT NOT NULL,
                compensation_data TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_comp_instance
                ON compensations(instance_id);
            CREATE INDEX IF NOT EXISTS idx_comp_key
                ON compensations(idempotency_key);
        """)

    def register(
        self,
        instance_id: str,
        step_name: str,
        idempotency_key: str,
        action_description: str,
        compensation_data: dict[str, Any],
    ) -> int:
        """
        Register a compensation entry before Act executes.

        Returns the entry ID.
        """
        now = time.time()
        with self._lock:
            cursor = self._conn.execute("""
                INSERT INTO compensations
                (instance_id, step_name, idempotency_key, action_description,
                 compensation_data, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instance_id, step_name, idempotency_key,
                action_description,
                json.dumps(compensation_data, default=str),
                CompensationStatus.PENDING.value,
                now, now,
            ))
            self._conn.commit()
            entry_id = cursor.lastrowid
            logger.info(
                "Compensation registered: instance=%s step=%s key=%s",
                instance_id, step_name, idempotency_key[:16],
            )
            return entry_id

    def confirm(self, idempotency_key: str) -> None:
        """Mark Act as successfully executed."""
        with self._lock:
            self._conn.execute("""
                UPDATE compensations SET status = ?, updated_at = ?
                WHERE idempotency_key = ? AND status = ?
            """, (CompensationStatus.CONFIRMED.value, time.time(),
                  idempotency_key, CompensationStatus.PENDING.value))
            self._conn.commit()

    def compensate(
        self,
        instance_id: str,
        handler: Callable[[CompensationEntry], bool] | None = None,
    ) -> list[CompensationEntry]:
        """
        Execute compensations for an instance in reverse order.

        Only confirmed entries are compensated (pending = Act never ran).

        Args:
            instance_id: The instance to compensate
            handler: Function that takes a CompensationEntry and returns
                     True if compensation succeeded. If None, all entries
                     are marked SKIPPED (escalate to HITL).

        Returns:
            List of entries with updated statuses.
        """
        with self._lock:
            rows = self._conn.execute("""
                SELECT * FROM compensations
                WHERE instance_id = ? AND status = ?
                ORDER BY created_at DESC
            """, (instance_id, CompensationStatus.CONFIRMED.value)).fetchall()

            results = []
            for row in rows:
                entry = self._row_to_entry(row)

                if handler is None:
                    # No handler — escalate to HITL
                    self._update_status(
                        entry.idempotency_key,
                        CompensationStatus.SKIPPED,
                        "No compensation handler — requires human intervention",
                    )
                    entry.status = CompensationStatus.SKIPPED
                    results.append(entry)
                    logger.warning(
                        "Compensation SKIPPED (no handler): instance=%s step=%s",
                        instance_id, entry.step_name,
                    )
                    continue

                try:
                    success = handler(entry)
                    if success:
                        self._update_status(
                            entry.idempotency_key,
                            CompensationStatus.COMPENSATED,
                        )
                        entry.status = CompensationStatus.COMPENSATED
                        logger.info(
                            "Compensation SUCCESS: instance=%s step=%s",
                            instance_id, entry.step_name,
                        )
                    else:
                        self._update_status(
                            entry.idempotency_key,
                            CompensationStatus.FAILED,
                            "Handler returned False",
                        )
                        entry.status = CompensationStatus.FAILED
                except Exception as e:
                    self._update_status(
                        entry.idempotency_key,
                        CompensationStatus.FAILED,
                        str(e),
                    )
                    entry.status = CompensationStatus.FAILED
                    entry.error = str(e)
                    logger.error(
                        "Compensation FAILED: instance=%s step=%s error=%s",
                        instance_id, entry.step_name, e,
                    )
                results.append(entry)

            return results

    def get_entries(self, instance_id: str) -> list[CompensationEntry]:
        """Get all compensation entries for an instance."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM compensations WHERE instance_id = ? ORDER BY created_at",
                (instance_id,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def _update_status(self, key: str, status: CompensationStatus, error: str = ""):
        self._conn.execute("""
            UPDATE compensations SET status = ?, error = ?, updated_at = ?
            WHERE idempotency_key = ?
        """, (status.value, error, time.time(), key))
        self._conn.commit()

    def _row_to_entry(self, row) -> CompensationEntry:
        return CompensationEntry(
            id=row["id"],
            instance_id=row["instance_id"],
            step_name=row["step_name"],
            idempotency_key=row["idempotency_key"],
            action_description=row["action_description"],
            compensation_data=json.loads(row["compensation_data"]),
            status=CompensationStatus(row["status"]),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self):
        self._conn.close()
