"""
Cognitive Core — State Replay (S-013)

Checkpoint snapshots stored after each workflow step completion.
Enables debugging by reconstructing workflow state at any step
and resuming execution from that point.

Two modes:
  - replay: Re-run from step N with the same inputs
  - override: Re-run from step N with modified inputs

Each replay creates a new trace with parent_trace_id linkage.

Usage:
    from engine.replay import ReplayManager

    mgr = ReplayManager(db_path="coordinator.db")
    mgr.save_checkpoint(trace_id, step_name, step_index, state_snapshot)

    # List available checkpoints
    checkpoints = mgr.list_checkpoints(trace_id)

    # Reconstruct state at a specific step
    state = mgr.get_checkpoint(trace_id, step_name)

    # Replay from that step
    replay_id = mgr.create_replay(
        trace_id, step_name, override_input={"complaint": "updated text"}
    )
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.replay")


@dataclass
class Checkpoint:
    """Snapshot of workflow state at a specific step."""
    checkpoint_id: str
    trace_id: str
    step_name: str
    step_index: int
    state_snapshot: dict[str, Any]
    created_at: float
    size_bytes: int = 0


@dataclass
class ReplayRecord:
    """Record of a replay execution."""
    replay_id: str
    parent_trace_id: str
    from_step: str
    from_step_index: int
    override_input: dict[str, Any] | None
    created_at: float
    status: str = "created"  # created, running, completed, failed


class ReplayManager:
    """
    Manages checkpoint storage and replay creation.

    Checkpoints are stored in the coordinator database (same SQLite file).
    """

    def __init__(self, db_path: str = "coordinator.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                state_snapshot TEXT NOT NULL,
                created_at REAL NOT NULL,
                size_bytes INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoint_trace
                ON checkpoints(trace_id);
            CREATE INDEX IF NOT EXISTS idx_checkpoint_step
                ON checkpoints(trace_id, step_name);

            CREATE TABLE IF NOT EXISTS replays (
                replay_id TEXT PRIMARY KEY,
                parent_trace_id TEXT NOT NULL,
                from_step TEXT NOT NULL,
                from_step_index INTEGER NOT NULL,
                override_input TEXT,
                created_at REAL NOT NULL,
                status TEXT DEFAULT 'created'
            );

            CREATE INDEX IF NOT EXISTS idx_replay_parent
                ON replays(parent_trace_id);
        """)
        self._conn.commit()

    def save_checkpoint(
        self,
        trace_id: str,
        step_name: str,
        step_index: int,
        state_snapshot: dict[str, Any],
    ) -> Checkpoint:
        """
        Save a checkpoint after a step completes.

        Call this from the engine after each successful step execution.
        The state_snapshot should include:
          - steps_completed: list of completed step names
          - step_outputs: dict mapping step_name → output
          - current_input: the case_input (possibly enriched by prior steps)
        """
        with self._lock:
            checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
            snapshot_json = json.dumps(state_snapshot, sort_keys=True, default=str)
            size = len(snapshot_json.encode("utf-8"))
            now = time.time()

            self._conn.execute(
                """INSERT INTO checkpoints
                   (checkpoint_id, trace_id, step_name, step_index,
                    state_snapshot, created_at, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (checkpoint_id, trace_id, step_name, step_index,
                 snapshot_json, now, size),
            )
            self._conn.commit()

            return Checkpoint(
                checkpoint_id=checkpoint_id,
                trace_id=trace_id,
                step_name=step_name,
                step_index=step_index,
                state_snapshot=state_snapshot,
                created_at=now,
                size_bytes=size,
            )

    def get_checkpoint(
        self,
        trace_id: str,
        step_name: str,
    ) -> Checkpoint | None:
        """Get the most recent checkpoint for a specific step."""
        row = self._conn.execute(
            """SELECT checkpoint_id, trace_id, step_name, step_index,
                      state_snapshot, created_at, size_bytes
               FROM checkpoints
               WHERE trace_id = ? AND step_name = ?
               ORDER BY created_at DESC LIMIT 1""",
            (trace_id, step_name),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            checkpoint_id=row[0],
            trace_id=row[1],
            step_name=row[2],
            step_index=row[3],
            state_snapshot=json.loads(row[4]),
            created_at=row[5],
            size_bytes=row[6],
        )

    def get_checkpoint_by_index(
        self,
        trace_id: str,
        step_index: int,
    ) -> Checkpoint | None:
        """Get checkpoint by step index."""
        row = self._conn.execute(
            """SELECT checkpoint_id, trace_id, step_name, step_index,
                      state_snapshot, created_at, size_bytes
               FROM checkpoints
               WHERE trace_id = ? AND step_index = ?
               ORDER BY created_at DESC LIMIT 1""",
            (trace_id, step_index),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            checkpoint_id=row[0], trace_id=row[1], step_name=row[2],
            step_index=row[3], state_snapshot=json.loads(row[4]),
            created_at=row[5], size_bytes=row[6],
        )

    def list_checkpoints(self, trace_id: str) -> list[Checkpoint]:
        """List all checkpoints for a trace, ordered by step index."""
        rows = self._conn.execute(
            """SELECT checkpoint_id, trace_id, step_name, step_index,
                      state_snapshot, created_at, size_bytes
               FROM checkpoints
               WHERE trace_id = ?
               ORDER BY step_index ASC""",
            (trace_id,),
        ).fetchall()
        return [
            Checkpoint(
                checkpoint_id=r[0], trace_id=r[1], step_name=r[2],
                step_index=r[3], state_snapshot=json.loads(r[4]),
                created_at=r[5], size_bytes=r[6],
            )
            for r in rows
        ]

    def create_replay(
        self,
        parent_trace_id: str,
        from_step: str,
        override_input: dict[str, Any] | None = None,
    ) -> ReplayRecord | None:
        """
        Create a replay record. The caller is responsible for actually
        executing the replay using the checkpoint state.

        Returns None if no checkpoint exists for the specified step.
        """
        checkpoint = self.get_checkpoint(parent_trace_id, from_step)
        if checkpoint is None:
            logger.warning(
                "No checkpoint for trace=%s step=%s — cannot create replay",
                parent_trace_id, from_step,
            )
            return None

        replay_id = f"replay_{uuid.uuid4().hex[:12]}"
        now = time.time()

        with self._lock:
            override_json = json.dumps(override_input, default=str) if override_input else None
            self._conn.execute(
                """INSERT INTO replays
                   (replay_id, parent_trace_id, from_step, from_step_index,
                    override_input, created_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (replay_id, parent_trace_id, from_step,
                 checkpoint.step_index, override_json, now, "created"),
            )
            self._conn.commit()

        return ReplayRecord(
            replay_id=replay_id,
            parent_trace_id=parent_trace_id,
            from_step=from_step,
            from_step_index=checkpoint.step_index,
            override_input=override_input,
            created_at=now,
        )

    def update_replay_status(self, replay_id: str, status: str):
        """Update replay execution status."""
        with self._lock:
            self._conn.execute(
                "UPDATE replays SET status = ? WHERE replay_id = ?",
                (status, replay_id),
            )
            self._conn.commit()

    def list_replays(self, parent_trace_id: str) -> list[ReplayRecord]:
        """List all replays for a parent trace."""
        rows = self._conn.execute(
            """SELECT replay_id, parent_trace_id, from_step, from_step_index,
                      override_input, created_at, status
               FROM replays
               WHERE parent_trace_id = ?
               ORDER BY created_at DESC""",
            (parent_trace_id,),
        ).fetchall()
        return [
            ReplayRecord(
                replay_id=r[0], parent_trace_id=r[1], from_step=r[2],
                from_step_index=r[3],
                override_input=json.loads(r[4]) if r[4] else None,
                created_at=r[5], status=r[6],
            )
            for r in rows
        ]

    def delete_checkpoints(self, trace_id: str) -> int:
        """Delete all checkpoints for a trace (cleanup)."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM checkpoints WHERE trace_id = ?",
                (trace_id,),
            )
            self._conn.commit()
            return cursor.rowcount

    def checkpoint_stats(self) -> dict[str, Any]:
        """Storage statistics."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM checkpoints"
        ).fetchone()
        replay_row = self._conn.execute(
            "SELECT COUNT(*) FROM replays"
        ).fetchone()
        return {
            "total_checkpoints": row[0],
            "total_bytes": row[1],
            "total_replays": replay_row[0],
        }

    def close(self):
        self._conn.close()
