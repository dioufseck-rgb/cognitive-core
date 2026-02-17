"""
Cognitive Core — Task Queue

Abstract interface for routing governance approvals, human reviews,
and work orders to external consumers. The coordinator publishes tasks;
consumers (CLI, REST API, queue workers, frontend apps) subscribe.

The interface is transport-agnostic:
  - InMemoryTaskQueue: dev/test, same process
  - SQLiteTaskQueue:   Phase 2, polling from DB
  - ServiceBusAdapter: Phase 4, Azure Service Bus
  - WebhookAdapter:    Phase 4, HTTP POST to external systems

Every task has a typed payload, a queue name, and a callback contract.
Consumers process tasks by calling coordinator.approve() / reject() /
resume() — the same methods regardless of how the task arrived.
"""

from __future__ import annotations

import abc
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ─── Task Types ──────────────────────────────────────────────────────

class TaskType:
    """Well-known task types published by the coordinator."""
    GOVERNANCE_APPROVAL = "governance_approval"
    SPOT_CHECK_REVIEW = "spot_check_review"
    HUMAN_DECISION = "human_decision"
    WORK_ORDER = "work_order"
    ESCALATION = "escalation"
    NOTIFICATION = "notification"


class TaskStatus:
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    A unit of work routed to a queue for external processing.

    The coordinator creates tasks; consumers claim and resolve them.
    Resolution calls back to the coordinator via the callback contract.
    """
    task_id: str
    task_type: str
    queue: str

    # What this task is about
    instance_id: str
    correlation_id: str
    workflow_type: str
    domain: str

    # Typed payload — everything the consumer needs to act
    payload: dict[str, Any]

    # Callback contract — how to resolve this task
    # In production: coordinator endpoint + method
    # In dev: coordinator.approve(instance_id) etc.
    callback: TaskCallback

    # Lifecycle
    status: str = TaskStatus.PENDING
    priority: int = 0  # 0=routine, 1=elevated, 2=urgent, 3=critical
    created_at: float = 0.0
    claimed_at: float | None = None
    claimed_by: str = ""
    resolved_at: float | None = None
    sla_seconds: float | None = None
    expires_at: float | None = None

    @staticmethod
    def create(
        task_type: str,
        queue: str,
        instance_id: str,
        correlation_id: str,
        workflow_type: str,
        domain: str,
        payload: dict[str, Any],
        callback: TaskCallback,
        priority: int = 0,
        sla_seconds: float | None = None,
    ) -> Task:
        now = time.time()
        return Task(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            task_type=task_type,
            queue=queue,
            instance_id=instance_id,
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            domain=domain,
            payload=payload,
            callback=callback,
            priority=priority,
            created_at=now,
            sla_seconds=sla_seconds,
            expires_at=now + sla_seconds if sla_seconds else None,
        )


@dataclass
class TaskCallback:
    """
    How to resolve a task. Transport-agnostic.

    In-process: method name + args pattern
    REST API: endpoint URL + method
    Service Bus: reply topic + correlation
    """
    method: str  # "approve", "reject", "resume", "complete"
    instance_id: str
    resume_nonce: str = ""
    # For REST/webhook integrations
    endpoint: str = ""
    # Additional context the consumer might need
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResolution:
    """Result of processing a task."""
    task_id: str
    action: str  # "approve", "reject", "defer", "escalate"
    resolved_by: str = ""
    notes: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    resolved_at: float = 0.0


# ─── Abstract Queue Interface ────────────────────────────────────────

class TaskQueue(abc.ABC):
    """
    Abstract task queue. Implementations handle transport.
    The coordinator calls publish(). Consumers call claim() + resolve().
    """

    @abc.abstractmethod
    def publish(self, task: Task) -> str:
        """Publish a task to a queue. Returns task_id."""
        ...

    @abc.abstractmethod
    def claim(self, queue: str, claimed_by: str = "") -> Task | None:
        """Claim the next pending task from a queue. Returns None if empty."""
        ...

    @abc.abstractmethod
    def resolve(self, task_id: str, resolution: TaskResolution) -> bool:
        """Resolve a claimed task. Returns True if successful."""
        ...

    @abc.abstractmethod
    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        ...

    @abc.abstractmethod
    def list_tasks(
        self,
        queue: str | None = None,
        status: str | None = None,
        instance_id: str | None = None,
    ) -> list[Task]:
        """List tasks with optional filters."""
        ...

    @abc.abstractmethod
    def expire_overdue(self) -> int:
        """Expire tasks past their SLA. Returns count expired."""
        ...


# ─── In-Memory Implementation ────────────────────────────────────────

class InMemoryTaskQueue(TaskQueue):
    """In-process task queue for dev/test."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def publish(self, task: Task) -> str:
        self._tasks[task.task_id] = task
        return task.task_id

    def claim(self, queue: str, claimed_by: str = "") -> Task | None:
        for task in sorted(
            self._tasks.values(),
            key=lambda t: (-t.priority, t.created_at),
        ):
            if task.queue == queue and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CLAIMED
                task.claimed_at = time.time()
                task.claimed_by = claimed_by
                return task
        return None

    def resolve(self, task_id: str, resolution: TaskResolution) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.CLAIMED:
            return False
        if resolution.action in ("approve", "complete"):
            task.status = TaskStatus.COMPLETED
        elif resolution.action == "reject":
            task.status = TaskStatus.REJECTED
        task.resolved_at = resolution.resolved_at or time.time()
        return True

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        queue: str | None = None,
        status: str | None = None,
        instance_id: str | None = None,
    ) -> list[Task]:
        tasks = list(self._tasks.values())
        if queue:
            tasks = [t for t in tasks if t.queue == queue]
        if status:
            tasks = [t for t in tasks if t.status == status]
        if instance_id:
            tasks = [t for t in tasks if t.instance_id == instance_id]
        return sorted(tasks, key=lambda t: (-t.priority, t.created_at))

    def expire_overdue(self) -> int:
        now = time.time()
        count = 0
        for task in self._tasks.values():
            if (task.status == TaskStatus.PENDING
                    and task.expires_at
                    and now > task.expires_at):
                task.status = TaskStatus.EXPIRED
                count += 1
        return count


# ─── SQLite Implementation ───────────────────────────────────────────

class SQLiteTaskQueue(TaskQueue):
    """SQLite-backed task queue. Polling-based for Phase 2."""

    def __init__(self, conn):
        """Takes a sqlite3 connection (shared with CoordinatorStore)."""
        self.conn = conn
        self._create_table()

    def _create_table(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_queue (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                queue TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                workflow_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                payload TEXT DEFAULT '{}',
                callback TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                claimed_at REAL,
                claimed_by TEXT DEFAULT '',
                resolved_at REAL,
                sla_seconds REAL,
                expires_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_taskq_queue_status
                ON task_queue(queue, status);
            CREATE INDEX IF NOT EXISTS idx_taskq_instance
                ON task_queue(instance_id);
        """)
        self.conn.commit()

    def publish(self, task: Task) -> str:
        self.conn.execute("""
            INSERT INTO task_queue
            (task_id, task_type, queue, instance_id, correlation_id,
             workflow_type, domain, payload, callback, status, priority,
             created_at, sla_seconds, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.task_type, task.queue,
            task.instance_id, task.correlation_id,
            task.workflow_type, task.domain,
            json.dumps(task.payload, default=str),
            json.dumps({
                "method": task.callback.method,
                "instance_id": task.callback.instance_id,
                "resume_nonce": task.callback.resume_nonce,
                "endpoint": task.callback.endpoint,
                "context": task.callback.context,
            }, default=str),
            task.status, task.priority,
            task.created_at, task.sla_seconds, task.expires_at,
        ))
        self.conn.commit()
        return task.task_id

    def claim(self, queue: str, claimed_by: str = "") -> Task | None:
        row = self.conn.execute("""
            SELECT * FROM task_queue
            WHERE queue = ? AND status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """, (queue,)).fetchone()
        if not row:
            return None
        task = self._row_to_task(row)
        now = time.time()
        self.conn.execute("""
            UPDATE task_queue SET status = 'claimed',
                claimed_at = ?, claimed_by = ?
            WHERE task_id = ?
        """, (now, claimed_by, task.task_id))
        self.conn.commit()
        task.status = TaskStatus.CLAIMED
        task.claimed_at = now
        task.claimed_by = claimed_by
        return task

    def resolve(self, task_id: str, resolution: TaskResolution) -> bool:
        row = self.conn.execute(
            "SELECT status FROM task_queue WHERE task_id = ?", (task_id,)
        ).fetchone()
        if not row or row["status"] != "claimed":
            return False
        status = TaskStatus.COMPLETED if resolution.action in ("approve", "complete") else TaskStatus.REJECTED
        now = resolution.resolved_at or time.time()
        self.conn.execute("""
            UPDATE task_queue SET status = ?, resolved_at = ?
            WHERE task_id = ?
        """, (status, now, task_id))
        self.conn.commit()
        return True

    def get_task(self, task_id: str) -> Task | None:
        row = self.conn.execute(
            "SELECT * FROM task_queue WHERE task_id = ?", (task_id,)
        ).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(
        self,
        queue: str | None = None,
        status: str | None = None,
        instance_id: str | None = None,
    ) -> list[Task]:
        query = "SELECT * FROM task_queue WHERE 1=1"
        params = []
        if queue:
            query += " AND queue = ?"
            params.append(queue)
        if status:
            query += " AND status = ?"
            params.append(status)
        if instance_id:
            query += " AND instance_id = ?"
            params.append(instance_id)
        query += " ORDER BY priority DESC, created_at ASC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    def expire_overdue(self) -> int:
        now = time.time()
        cursor = self.conn.execute("""
            UPDATE task_queue SET status = 'expired'
            WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?
        """, (now,))
        self.conn.commit()
        return cursor.rowcount

    def _row_to_task(self, row) -> Task:
        cb_data = json.loads(row["callback"])
        return Task(
            task_id=row["task_id"],
            task_type=row["task_type"],
            queue=row["queue"],
            instance_id=row["instance_id"],
            correlation_id=row["correlation_id"],
            workflow_type=row["workflow_type"],
            domain=row["domain"],
            payload=json.loads(row["payload"]),
            callback=TaskCallback(
                method=cb_data.get("method", "approve"),
                instance_id=cb_data.get("instance_id", ""),
                resume_nonce=cb_data.get("resume_nonce", ""),
                endpoint=cb_data.get("endpoint", ""),
                context=cb_data.get("context", {}),
            ),
            status=row["status"],
            priority=row["priority"],
            created_at=row["created_at"],
            claimed_at=row["claimed_at"],
            claimed_by=row["claimed_by"] or "",
            resolved_at=row["resolved_at"],
            sla_seconds=row["sla_seconds"],
            expires_at=row["expires_at"],
        )
