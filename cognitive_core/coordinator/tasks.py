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
    RESOURCE_REQUEST = "resource_request"


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

    @abc.abstractmethod
    def force_complete(self, task_id: str) -> bool:
        """Mark a task as completed directly, bypassing the claim requirement.
        Used when approve/reject is called directly on the coordinator."""
        ...

    @abc.abstractmethod
    def unclaim(self, task_id: str) -> bool:
        """Return a claimed task to pending state (for defer actions)."""
        ...

    @property
    def backend_name(self) -> str:
        """Return the backend identifier for proof ledger recording."""
        return self.__class__.__name__


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

    def force_complete(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.COMPLETED
        task.resolved_at = time.time()
        return True

    def unclaim(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.PENDING
        task.claimed_at = None
        task.claimed_by = ""
        return True


# ─── SQLite Implementation ───────────────────────────────────────────

class SQLiteTaskQueue(TaskQueue):
    """Database-backed task queue. Uses engine.db.DatabaseBackend."""

    def __init__(self, db):
        """Takes a DatabaseBackend (shared with CoordinatorStore)."""
        self.db = db
        self._create_table()

    def _create_table(self):
        self.db.executescript("""
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
        self.db.commit()

    def publish(self, task: Task) -> str:
        self.db.fetchone("""
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
        self.db.commit()
        return task.task_id

    def claim(self, queue: str, claimed_by: str = "") -> Task | None:
        row = self.db.fetchone("""
            SELECT * FROM task_queue
            WHERE queue = ? AND status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """, (queue,))
        if not row:
            return None
        task = self._row_to_task(row)
        now = time.time()
        self.db.execute("""
            UPDATE task_queue SET status = 'claimed',
                claimed_at = ?, claimed_by = ?
            WHERE task_id = ?
        """, (now, claimed_by, task.task_id))
        self.db.commit()
        task.status = TaskStatus.CLAIMED
        task.claimed_at = now
        task.claimed_by = claimed_by
        return task

    def resolve(self, task_id: str, resolution: TaskResolution) -> bool:
        row = self.db.fetchone(
            "SELECT status FROM task_queue WHERE task_id = ?", (task_id,)
        )
        if not row or row["status"] != "claimed":
            return False
        status = TaskStatus.COMPLETED if resolution.action in ("approve", "complete") else TaskStatus.REJECTED
        now = resolution.resolved_at or time.time()
        self.db.execute("""
            UPDATE task_queue SET status = ?, resolved_at = ?
            WHERE task_id = ?
        """, (status, now, task_id))
        self.db.commit()
        return True

    def get_task(self, task_id: str) -> Task | None:
        row = self.db.fetchone(
            "SELECT * FROM task_queue WHERE task_id = ?", (task_id,)
        )
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
        rows = self.db.fetchall(query, tuple(params))
        return [self._row_to_task(r) for r in rows]

    def expire_overdue(self) -> int:
        now = time.time()
        cursor = self.db.execute("""
            UPDATE task_queue SET status = 'expired'
            WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?
        """, (now,))
        self.db.commit()
        return cursor.rowcount

    def force_complete(self, task_id: str) -> bool:
        self.db.execute(
            "UPDATE task_queue SET status = 'completed', resolved_at = ? WHERE task_id = ?",
            (time.time(), task_id),
        )
        self.db.commit()
        return True

    def unclaim(self, task_id: str) -> bool:
        self.db.execute("""
            UPDATE task_queue SET status = 'pending',
                claimed_at = NULL, claimed_by = ''
            WHERE task_id = ?
        """, (task_id,))
        self.db.commit()
        return True

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


# ─── Azure Service Bus Adapter ───────────────────────────────────────

try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage
    _SERVICE_BUS_AVAILABLE = True
except ImportError:
    ServiceBusClient = None  # type: ignore[assignment,misc]
    ServiceBusMessage = None  # type: ignore[assignment,misc]
    _SERVICE_BUS_AVAILABLE = False

import logging as _logging
_logger = _logging.getLogger("cognitive_core.coordinator.tasks.servicebus")


class ServiceBusTaskQueueAdapter(TaskQueue):
    """
    Azure Service Bus-backed task queue adapter.

    Sends governance tasks to a Service Bus queue on suspension.
    Guarantees at-least-once delivery with idempotency key:
      instance_id + resume_nonce

    When DATA_SERVICE_BUS_URL is not set, use create_task_queue() which
    falls back to SQLiteTaskQueue automatically.
    """

    def __init__(self, connection_string: str, queue_name: str = "governance-tasks"):
        if not _SERVICE_BUS_AVAILABLE:
            raise RuntimeError(
                "azure-servicebus package is not installed. "
                "Install it with: pip install azure-servicebus"
            )
        self._connection_string = connection_string
        self._queue_name = queue_name
        # Local task registry — tasks are also stored here for get_task/list_tasks
        self._tasks: dict[str, Task] = {}
        # Map task_id → received ServiceBusMessage (for settlement)
        self._pending_messages: dict[str, Any] = {}
        # Idempotency set: set of "instance_id:resume_nonce" already processed
        self._processed_keys: set[str] = set()

    @property
    def backend_name(self) -> str:
        return "ServiceBusTaskQueueAdapter"

    def _idempotency_key(self, task: Task) -> str:
        return f"{task.instance_id}:{task.callback.resume_nonce}"

    def publish(self, task: Task) -> str:
        """Send a task to the Service Bus queue and record it locally."""
        idem_key = self._idempotency_key(task)
        if idem_key in self._processed_keys:
            _logger.warning(
                "Duplicate publish suppressed for idempotency key %s (task_id=%s)",
                idem_key, task.task_id,
            )
            return task.task_id

        body = json.dumps({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "queue": task.queue,
            "instance_id": task.instance_id,
            "correlation_id": task.correlation_id,
            "workflow_type": task.workflow_type,
            "domain": task.domain,
            "payload": task.payload,
            "callback": {
                "method": task.callback.method,
                "instance_id": task.callback.instance_id,
                "resume_nonce": task.callback.resume_nonce,
                "endpoint": task.callback.endpoint,
                "context": task.callback.context,
            },
            "status": task.status,
            "priority": task.priority,
            "created_at": task.created_at,
            "sla_seconds": task.sla_seconds,
            "expires_at": task.expires_at,
        }, default=str)

        with ServiceBusClient.from_connection_string(self._connection_string) as client:
            with client.get_queue_sender(self._queue_name) as sender:
                sb_message = ServiceBusMessage(
                    body=body,
                    message_id=task.task_id,
                    subject=task.task_type,
                    application_properties={
                        "instance_id": task.instance_id,
                        "queue": task.queue,
                        "priority": task.priority,
                    },
                )
                sender.send_messages(sb_message)

        # Record locally for list/get operations
        self._tasks[task.task_id] = task
        _logger.info("Published task %s (type=%s) to Service Bus", task.task_id, task.task_type)
        return task.task_id

    def claim(self, queue: str, claimed_by: str = "") -> Task | None:
        """Receive the next message from Service Bus and return it as a Task."""
        with ServiceBusClient.from_connection_string(self._connection_string) as client:
            with client.get_queue_receiver(
                self._queue_name, max_wait_time=5
            ) as receiver:
                messages = receiver.receive_messages(max_message_count=1, max_wait_time=5)
                if not messages:
                    return None
                msg = messages[0]
                try:
                    data = json.loads(str(msg))
                except Exception:
                    body_bytes = b"".join(msg.body)
                    data = json.loads(body_bytes.decode("utf-8"))

                if data.get("queue") != queue:
                    # Wrong queue — abandon and skip
                    receiver.abandon_message(msg)
                    return None

                task_id = data["task_id"]

                # Idempotency check
                idem_key = f"{data['instance_id']}:{data.get('callback', {}).get('resume_nonce', '')}"
                if idem_key in self._processed_keys:
                    _logger.info(
                        "Idempotency: suppressing duplicate delivery for key %s", idem_key
                    )
                    receiver.complete_message(msg)
                    return None

                cb_data = data.get("callback", {})
                task = Task(
                    task_id=task_id,
                    task_type=data["task_type"],
                    queue=data["queue"],
                    instance_id=data["instance_id"],
                    correlation_id=data["correlation_id"],
                    workflow_type=data["workflow_type"],
                    domain=data["domain"],
                    payload=data.get("payload", {}),
                    callback=TaskCallback(
                        method=cb_data.get("method", "approve"),
                        instance_id=cb_data.get("instance_id", ""),
                        resume_nonce=cb_data.get("resume_nonce", ""),
                        endpoint=cb_data.get("endpoint", ""),
                        context=cb_data.get("context", {}),
                    ),
                    status=TaskStatus.CLAIMED,
                    priority=data.get("priority", 0),
                    created_at=data.get("created_at", time.time()),
                    claimed_at=time.time(),
                    claimed_by=claimed_by,
                    sla_seconds=data.get("sla_seconds"),
                    expires_at=data.get("expires_at"),
                )

                self._tasks[task_id] = task
                # Keep the receiver context alive by storing the message for settlement
                # Note: in production you'd use a peek-lock receiver with longer lifetime
                self._pending_messages[task_id] = (receiver, msg)
                return task

    def resolve(self, task_id: str, resolution: TaskResolution) -> bool:
        """Complete or abandon the Service Bus message."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.CLAIMED:
            return False

        idem_key = self._idempotency_key(task)

        if resolution.action in ("approve", "complete"):
            task.status = TaskStatus.COMPLETED
            # Mark idempotency key as processed
            self._processed_keys.add(idem_key)
        elif resolution.action == "reject":
            task.status = TaskStatus.REJECTED
            self._processed_keys.add(idem_key)
        else:
            # Abandon — let it be redelivered
            pass

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

    def force_complete(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        idem_key = self._idempotency_key(task)
        task.status = TaskStatus.COMPLETED
        task.resolved_at = time.time()
        self._processed_keys.add(idem_key)
        return True

    def unclaim(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.PENDING
        task.claimed_at = None
        task.claimed_by = ""
        return True


# ─── Factory ─────────────────────────────────────────────────────────

import os as _os


def create_task_queue(db=None, queue_name: str = "governance-tasks") -> TaskQueue:
    """
    Factory that returns ServiceBusTaskQueueAdapter when DATA_SERVICE_BUS_URL
    is set, otherwise falls back to SQLiteTaskQueue (or InMemoryTaskQueue if
    no db is provided).
    """
    sb_url = _os.environ.get("DATA_SERVICE_BUS_URL", "")
    if sb_url:
        if not _SERVICE_BUS_AVAILABLE:
            _logger.warning(
                "DATA_SERVICE_BUS_URL is set but azure-servicebus is not installed. "
                "Falling back to SQLite task queue."
            )
        else:
            _logger.info("Using Azure Service Bus task queue (queue=%s)", queue_name)
            return ServiceBusTaskQueueAdapter(
                connection_string=sb_url,
                queue_name=queue_name,
            )
    if db is not None:
        return SQLiteTaskQueue(db)
    return InMemoryTaskQueue()
