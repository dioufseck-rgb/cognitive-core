"""
Cognitive Core — Coordinator State Store

Database-backed persistence for workflow instances, work orders,
suspensions, and the action ledger. Uses engine.db.DatabaseBackend
for portable SQLite (dev) and PostgreSQL (prod) support. Supports checkpoint/resume
across process restarts.

Backend auto-selected via CC_DB_BACKEND env var (default: sqlite).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from coordinator.types import (
    InstanceState,
    InstanceStatus,
    WorkOrder,
    WorkOrderStatus,
    WorkOrderResult,
    Suspension,
)


class _Transaction:
    """
    Transaction context manager.

    While active, individual save_*/commit() calls become no-ops.
    The real COMMIT happens when the context manager exits cleanly.
    Works with both SQLite and Postgres via DatabaseBackend.
    """
    def __init__(self, db, store):
        self.db = db
        self.store = store
        self._ctx = None

    def __enter__(self):
        self._ctx = self.db.transaction()
        self._ctx.__enter__()
        self.store._in_transaction = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store._in_transaction = False
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


class CoordinatorStore:
    """Database-backed store for coordinator state.
    
    Uses engine.db.DatabaseBackend for portable SQLite/Postgres support.
    Backend is selected via CC_DB_BACKEND env var (default: sqlite).
    """

    def __init__(self, db_path: str | Path = "coordinator.db", db_backend=None):
        self.db_path = str(db_path)
        if db_backend is not None:
            self.db = db_backend
        else:
            from engine.db import create_backend
            self.db = create_backend(path=str(db_path))
        self._in_transaction = False
        self._create_tables()

    def _commit(self):
        """Commit unless inside an explicit transaction block."""
        if not self._in_transaction:
            self.db.commit()

    def transaction(self):
        """
        Context manager for explicit transaction boundaries.

        Usage:
            with store.transaction():
                store.save_instance(inst)
                store.save_suspension(sus)
                # Both committed atomically, or both rolled back

        Without this, each save_instance/save_work_order commits
        independently. A crash between two saves leaves corrupt state.
        """
        return _Transaction(self.db, self)

    def _create_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS instances (
                instance_id TEXT PRIMARY KEY,
                workflow_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'created',
                governance_tier TEXT NOT NULL DEFAULT 'gate',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                lineage TEXT DEFAULT '[]',
                correlation_id TEXT DEFAULT '',
                current_step TEXT DEFAULT '',
                step_count INTEGER DEFAULT 0,
                elapsed_seconds REAL DEFAULT 0.0,
                pending_work_orders TEXT DEFAULT '[]',
                resume_nonce TEXT DEFAULT '',
                result TEXT,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS work_orders (
                work_order_id TEXT PRIMARY KEY,
                requester_instance_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                contract_name TEXT NOT NULL,
                contract_version INTEGER DEFAULT 1,
                inputs TEXT DEFAULT '{}',
                sla_seconds REAL,
                urgency TEXT DEFAULT 'routine',
                handler_workflow_type TEXT DEFAULT '',
                handler_domain TEXT DEFAULT '',
                handler_instance_id TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'created',
                created_at REAL NOT NULL,
                dispatched_at REAL,
                completed_at REAL,
                result TEXT
            );

            CREATE TABLE IF NOT EXISTS suspensions (
                instance_id TEXT PRIMARY KEY,
                suspended_at_step TEXT NOT NULL,
                state_snapshot TEXT NOT NULL,
                unresolved_needs TEXT DEFAULT '[]',
                work_order_ids TEXT DEFAULT '[]',
                resume_nonce TEXT NOT NULL,
                suspended_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS action_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                details TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_instances_status ON instances(status);
            CREATE INDEX IF NOT EXISTS idx_instances_correlation ON instances(correlation_id);
            CREATE INDEX IF NOT EXISTS idx_work_orders_status ON work_orders(status);
            CREATE INDEX IF NOT EXISTS idx_work_orders_requester ON work_orders(requester_instance_id);
            CREATE INDEX IF NOT EXISTS idx_ledger_instance ON action_ledger(instance_id);
        """)
        self._commit()

    # ─── Instance CRUD ───────────────────────────────────────────────

    def save_instance(self, inst: InstanceState):
        self.db.fetchall("""
            INSERT OR REPLACE INTO instances
            (instance_id, workflow_type, domain, status, governance_tier,
             created_at, updated_at, lineage, correlation_id,
             current_step, step_count, elapsed_seconds,
             pending_work_orders, resume_nonce, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inst.instance_id, inst.workflow_type, inst.domain,
            inst.status.value, inst.governance_tier,
            inst.created_at, inst.updated_at,
            json.dumps(inst.lineage), inst.correlation_id,
            inst.current_step, inst.step_count, inst.elapsed_seconds,
            json.dumps(inst.pending_work_orders), inst.resume_nonce,
            json.dumps(inst.result) if inst.result else None,
            inst.error,
        ))
        self._commit()

    def get_instance(self, instance_id: str) -> InstanceState | None:
        row = self.db.fetchone("SELECT * FROM instances WHERE instance_id = ?", (instance_id,))
        if not row:
            return None
        return self._row_to_instance(row)

    def list_instances(
        self,
        status: InstanceStatus | None = None,
        correlation_id: str | None = None,
        limit: int = 500,
    ) -> list[InstanceState]:
        query = "SELECT * FROM instances WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if correlation_id:
            query += " AND correlation_id = ?"
            params.append(correlation_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.db.execute(query, params).fetchall()
        return [self._row_to_instance(r) for r in rows]

    def _row_to_instance(self, row) -> InstanceState:
        return InstanceState(
            instance_id=row["instance_id"],
            workflow_type=row["workflow_type"],
            domain=row["domain"],
            status=InstanceStatus(row["status"]),
            governance_tier=row["governance_tier"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            lineage=json.loads(row["lineage"]),
            correlation_id=row["correlation_id"],
            current_step=row["current_step"],
            step_count=row["step_count"],
            elapsed_seconds=row["elapsed_seconds"],
            pending_work_orders=json.loads(row["pending_work_orders"]),
            resume_nonce=row["resume_nonce"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
        )

    # ─── Work Order CRUD ─────────────────────────────────────────────

    def save_work_order(self, wo: WorkOrder):
        result_json = None
        if wo.result:
            result_json = json.dumps({
                "work_order_id": wo.result.work_order_id,
                "status": wo.result.status,
                "outputs": wo.result.outputs,
                "error": wo.result.error,
                "completed_at": wo.result.completed_at,
            })
        self.db.execute("""
            INSERT OR REPLACE INTO work_orders
            (work_order_id, requester_instance_id, correlation_id,
             contract_name, contract_version, inputs,
             sla_seconds, urgency,
             handler_workflow_type, handler_domain, handler_instance_id,
             status, created_at, dispatched_at, completed_at, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            wo.work_order_id, wo.requester_instance_id, wo.correlation_id,
            wo.contract_name, wo.contract_version, json.dumps(wo.inputs),
            wo.sla_seconds, wo.urgency,
            wo.handler_workflow_type, wo.handler_domain, wo.handler_instance_id,
            wo.status.value, wo.created_at, wo.dispatched_at, wo.completed_at,
            result_json,
        ))
        self._commit()

    def get_work_order(self, work_order_id: str) -> WorkOrder | None:
        row = self.db.fetchone("SELECT * FROM work_orders WHERE work_order_id = ?", (work_order_id,))
        if not row:
            return None
        return self._row_to_work_order(row)

    def get_work_orders_for_instance(self, instance_id: str) -> list[WorkOrder]:
        rows = self.db.fetchall("SELECT * FROM work_orders WHERE requester_instance_id = ? ORDER BY created_at", (instance_id,))
        return [self._row_to_work_order(r) for r in rows]

    def get_work_orders_for_requester_or_handler(self, instance_id: str) -> list[WorkOrder]:
        """Find work orders where instance_id is either requester or handler."""
        rows = self.db.fetchall("SELECT * FROM work_orders WHERE requester_instance_id = ? OR handler_instance_id = ? ORDER BY created_at", (instance_id, instance_id))
        return [self._row_to_work_order(r) for r in rows]

    def _row_to_work_order(self, row) -> WorkOrder:
        result = None
        if row["result"]:
            rd = json.loads(row["result"])
            result = WorkOrderResult(
                work_order_id=rd["work_order_id"],
                status=rd["status"],
                outputs=rd.get("outputs", {}),
                error=rd.get("error"),
                completed_at=rd.get("completed_at", 0),
            )
        return WorkOrder(
            work_order_id=row["work_order_id"],
            requester_instance_id=row["requester_instance_id"],
            correlation_id=row["correlation_id"],
            contract_name=row["contract_name"],
            contract_version=row["contract_version"],
            inputs=json.loads(row["inputs"]),
            sla_seconds=row["sla_seconds"],
            urgency=row["urgency"],
            handler_workflow_type=row["handler_workflow_type"],
            handler_domain=row["handler_domain"],
            handler_instance_id=row["handler_instance_id"],
            status=WorkOrderStatus(row["status"]),
            created_at=row["created_at"],
            dispatched_at=row["dispatched_at"],
            completed_at=row["completed_at"],
            result=result,
        )

    # ─── Suspension CRUD ─────────────────────────────────────────────

    def save_suspension(self, sus: Suspension):
        # M2: In strict mode, fail fast on non-serializable state
        # instead of silently converting objects to their str() repr.
        _strict = "COGNITIVE_CORE_STRICT" in __import__("os").environ
        if _strict:
            try:
                snapshot_json = json.dumps(sus.state_snapshot)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"State snapshot contains non-serializable data: {e}. "
                    f"This would silently corrupt state on resume. "
                    f"Fix the upstream code that put a non-JSON-native object into the state."
                ) from e
        else:
            snapshot_json = json.dumps(sus.state_snapshot, default=str)

        self.db.execute("""
            INSERT OR REPLACE INTO suspensions
            (instance_id, suspended_at_step, state_snapshot,
             unresolved_needs, work_order_ids, resume_nonce, suspended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            sus.instance_id, sus.suspended_at_step,
            snapshot_json,
            json.dumps(sus.unresolved_needs),
            json.dumps({
                "work_order_ids": sus.work_order_ids,
                "wo_need_map": sus.wo_need_map,
                "deferred_needs": sus.deferred_needs,
            }),
            sus.resume_nonce, sus.suspended_at,
        ))
        self._commit()

    def get_suspension(self, instance_id: str) -> Suspension | None:
        row = self.db.fetchone("SELECT * FROM suspensions WHERE instance_id = ?", (instance_id,))
        if not row:
            return None
        # work_order_ids column now contains a dict with wo_need_map and deferred_needs
        wo_data_raw = json.loads(row["work_order_ids"])
        if isinstance(wo_data_raw, dict):
            wo_ids = wo_data_raw.get("work_order_ids", [])
            wo_need_map = wo_data_raw.get("wo_need_map", {})
            deferred_needs = wo_data_raw.get("deferred_needs", [])
        else:
            # Backward compat: old format was just a list
            wo_ids = wo_data_raw
            wo_need_map = {}
            deferred_needs = []
        return Suspension(
            instance_id=row["instance_id"],
            suspended_at_step=row["suspended_at_step"],
            state_snapshot=json.loads(row["state_snapshot"]),
            unresolved_needs=json.loads(row["unresolved_needs"]),
            work_order_ids=wo_ids,
            wo_need_map=wo_need_map,
            deferred_needs=deferred_needs,
            resume_nonce=row["resume_nonce"],
            suspended_at=row["suspended_at"],
        )

    def delete_suspension(self, instance_id: str):
        self.db.execute(
            "DELETE FROM suspensions WHERE instance_id = ?", (instance_id,)
        )
        self._commit()

    # ─── Action Ledger ───────────────────────────────────────────────

    def log_action(
        self,
        instance_id: str,
        correlation_id: str,
        action_type: str,
        details: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> bool:
        """
        Log an action to the ledger. Returns False if the idempotency
        key already exists (preventing duplicate execution).
        """
        try:
            self.db.execute("""
                INSERT INTO action_ledger
                (instance_id, correlation_id, action_type, details,
                 idempotency_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                instance_id, correlation_id, action_type,
                json.dumps(details, default=str),
                idempotency_key, time.time(),
            ))
            self._commit()
            return True
        except Exception:  # IntegrityError (SQLite) or UniqueViolation (Postgres)
            # Idempotency key already exists
            return False

    def get_ledger(
        self,
        instance_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM action_ledger WHERE 1=1"
        params = []
        if instance_id:
            query += " AND instance_id = ?"
            params.append(instance_id)
        if correlation_id:
            query += " AND correlation_id = ?"
            params.append(correlation_id)
        query += " ORDER BY created_at"
        rows = self.db.execute(query, params).fetchall()
        return [
            {
                "id": r["id"],
                "instance_id": r["instance_id"],
                "correlation_id": r["correlation_id"],
                "action_type": r["action_type"],
                "details": json.loads(r["details"]),
                "idempotency_key": r["idempotency_key"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ─── Statistics ──────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the coordinator."""
        instances = self.db.execute(
            "SELECT status, COUNT(*) as cnt FROM instances GROUP BY status"
        ).fetchall()
        work_orders = self.db.execute(
            "SELECT status, COUNT(*) as cnt FROM work_orders GROUP BY status"
        ).fetchall()
        ledger_count = self.db.fetchone("SELECT COUNT(*) as cnt FROM action_ledger")["cnt"]

        return {
            "instances": {r["status"]: r["cnt"] for r in instances},
            "work_orders": {r["status"]: r["cnt"] for r in work_orders},
            "action_ledger_entries": ledger_count,
        }

    def close(self):
        self.db.close()

    # ─── Reliability Checks ─────────────────────────────────────────

    def find_stuck_instances(self, max_running_seconds: int = 3600) -> list[InstanceState]:
        """
        M3: Find instances stuck in RUNNING state beyond the timeout.

        In normal operation, instances transition through RUNNING in
        seconds to minutes. An instance stuck in RUNNING for > 1 hour
        likely crashed mid-execution.
        """
        cutoff = time.time() - max_running_seconds
        rows = self.db.fetchall("SELECT * FROM instances WHERE status = 'running' AND updated_at < ?", (cutoff,))
        return [self._row_to_instance(r) for r in rows]

    def find_orphaned_suspensions(self) -> list[dict[str, Any]]:
        """
        Find suspensions whose work orders are all COMPLETED or FAILED
        but the instance is still SUSPENDED (resume never happened).
        """
        rows = self.db.execute("""
            SELECT s.instance_id, s.suspended_at_step, s.work_order_ids
            FROM suspensions s
            JOIN instances i ON s.instance_id = i.instance_id
            WHERE i.status = 'suspended'
        """)

        orphans = []
        for r in rows:
            wo_ids = json.loads(r["work_order_ids"])
            all_resolved = True
            for wo_id in wo_ids:
                wo = self.get_work_order(wo_id)
                if wo and wo.status in (WorkOrderStatus.DISPATCHED, WorkOrderStatus.RUNNING):
                    all_resolved = False
                    break
            if all_resolved and wo_ids:
                orphans.append({
                    "instance_id": r["instance_id"],
                    "suspended_at": r["suspended_at_step"],
                    "work_order_ids": wo_ids,
                })
        return orphans
