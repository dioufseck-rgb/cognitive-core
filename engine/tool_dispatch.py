"""
Cognitive Core — Tool Dispatch + Act Safety (H-005, H-006, H-007, H-008)

Runtime enforcement of the read/write boundary, step timeouts,
input integrity checksums, and idempotency key management.

H-005: ToolDispatcher — all external calls route through it.
       Write tools blocked outside Act primitive.
H-006: StepTimeout — per-step execution deadlines.
H-007: IntegrityChecker — SHA-256 checksums on retrieved documents.
H-008: IdempotencyManager — prevent duplicate Act execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.tool_dispatch")


# ═══════════════════════════════════════════════════════════════
# H-005: Tool Registry + Write Boundary
# ═══════════════════════════════════════════════════════════════

class ToolMode(str, Enum):
    READ = "read"
    WRITE = "write"


class WriteBoundaryViolation(Exception):
    """Raised when a non-Act primitive attempts to call a write tool."""
    pass


class UnknownToolError(Exception):
    """Raised when calling an unregistered tool."""
    pass


@dataclass
class RegisteredTool:
    """A tool registered with the dispatcher."""
    name: str
    fn: Callable
    mode: ToolMode
    description: str = ""


class ToolDispatcher:
    """
    Central dispatcher for all external tool calls.

    Enforces:
    - Only registered tools can be called
    - Write tools can only be called from Act primitives
    - All calls are logged with timing
    """

    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}
        self._call_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        fn: Callable,
        mode: ToolMode = ToolMode.READ,
        description: str = "",
    ) -> None:
        """Register a tool with the dispatcher."""
        with self._lock:
            self._tools[name] = RegisteredTool(
                name=name, fn=fn, mode=mode, description=description,
            )
            logger.info("Tool registered: %s (mode=%s)", name, mode.value)

    def dispatch(
        self,
        tool_name: str,
        args: dict[str, Any],
        current_primitive: str,
        instance_id: str = "",
        step_name: str = "",
    ) -> Any:
        """
        Dispatch a tool call with boundary enforcement.

        Raises:
            UnknownToolError: Tool not registered
            WriteBoundaryViolation: Write tool called outside Act
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise UnknownToolError(f"Tool not registered: {tool_name!r}")

        # CORE INVARIANT: write tools only from Act
        if tool.mode == ToolMode.WRITE and current_primitive != "act":
            raise WriteBoundaryViolation(
                f"Write tool {tool_name!r} called from {current_primitive!r} primitive. "
                f"Only the Act primitive (primitive 8) can call write tools. "
                f"This is a read/write boundary violation."
            )

        start = time.time()
        try:
            result = tool.fn(**args)
            elapsed = time.time() - start
            self._log_call(tool_name, current_primitive, instance_id, step_name, elapsed, True)
            return result
        except Exception as e:
            elapsed = time.time() - start
            self._log_call(tool_name, current_primitive, instance_id, step_name, elapsed, False, str(e))
            raise

    def get_tool(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self, mode: ToolMode | None = None) -> list[RegisteredTool]:
        tools = list(self._tools.values())
        if mode:
            tools = [t for t in tools if t.mode == mode]
        return tools

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def _log_call(self, tool_name, primitive, instance_id, step_name, elapsed, success, error=None):
        entry = {
            "tool": tool_name,
            "primitive": primitive,
            "instance_id": instance_id,
            "step": step_name,
            "elapsed_s": round(elapsed, 4),
            "success": success,
            "timestamp": time.time(),
        }
        if error:
            entry["error"] = error
        self._call_log.append(entry)


# ═══════════════════════════════════════════════════════════════
# H-006: Step-Level Hard Timeouts
# ═══════════════════════════════════════════════════════════════

class StepTimeoutError(Exception):
    """Raised when a step exceeds its execution deadline."""
    def __init__(self, step_name: str, timeout_seconds: float, elapsed: float):
        self.step_name = step_name
        self.timeout_seconds = timeout_seconds
        self.elapsed = elapsed
        super().__init__(
            f"Step {step_name!r} timed out after {elapsed:.1f}s "
            f"(deadline: {timeout_seconds}s)"
        )


def run_with_timeout(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: float = 30.0,
    step_name: str = "unknown",
) -> Any:
    """
    Run a function with a hard timeout.

    Uses threading to enforce the deadline. If the function exceeds
    the timeout, StepTimeoutError is raised in the calling thread.
    The worker thread is abandoned (daemon) — it may continue running
    but its result is discarded.

    Args:
        fn: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Maximum execution time
        step_name: For error messages

    Returns:
        Function result

    Raises:
        StepTimeoutError: If timeout exceeded
    """
    kwargs = kwargs or {}
    result = [None]
    error = [None]
    start = time.time()

    def worker():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        elapsed = time.time() - start
        raise StepTimeoutError(step_name, timeout_seconds, elapsed)

    if error[0] is not None:
        raise error[0]

    return result[0]


# ═══════════════════════════════════════════════════════════════
# H-007: Input Integrity Checksums
# ═══════════════════════════════════════════════════════════════

@dataclass
class IntegrityRecord:
    """Record of a document's integrity checksum."""
    source_name: str
    content_hash: str
    byte_count: int
    algorithm: str = "sha256"
    timestamp: float = field(default_factory=time.time)


class IntegrityChecker:
    """
    Compute and verify SHA-256 checksums on retrieved documents.

    At Retrieve time: hash the raw bytes, record in audit trail.
    After the fact: verify original document matches recorded hash.
    """

    def __init__(self, audit_trail: Any = None):
        self._records: list[IntegrityRecord] = []
        self._audit = audit_trail

    def hash_content(
        self,
        content: bytes,
        source_name: str,
        trace_id: str = "",
    ) -> IntegrityRecord:
        """
        Hash document content and record the checksum.

        Args:
            content: Raw document bytes
            source_name: Filename or identifier
            trace_id: For audit trail linkage

        Returns:
            IntegrityRecord with hash
        """
        h = hashlib.sha256(content).hexdigest()
        record = IntegrityRecord(
            source_name=source_name,
            content_hash=h,
            byte_count=len(content),
        )
        self._records.append(record)

        logger.info(
            "Input integrity: source=%s hash=%s bytes=%d",
            source_name, h[:16] + "...", len(content),
            extra={
                "event_type": "input_integrity",
                "source_name": source_name,
                "content_hash": h,
                "byte_count": len(content),
            },
        )

        if self._audit and trace_id:
            try:
                self._audit.record(
                    trace_id=trace_id,
                    event_type="input_integrity",
                    payload={
                        "source_name": source_name,
                        "content_hash": h,
                        "byte_count": len(content),
                        "algorithm": "sha256",
                    },
                )
            except Exception as e:
                logger.warning("Failed to record integrity to audit: %s", e)

        return record

    def verify(self, content: bytes, expected_hash: str) -> bool:
        """Verify content matches expected hash."""
        actual = hashlib.sha256(content).hexdigest()
        return actual == expected_hash

    def get_records(self, source_name: str | None = None) -> list[IntegrityRecord]:
        if source_name:
            return [r for r in self._records if r.source_name == source_name]
        return list(self._records)


# ═══════════════════════════════════════════════════════════════
# H-008: Idempotency Key Management
# ═══════════════════════════════════════════════════════════════

class IdempotencyStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IdempotencyRecord:
    key: str
    instance_id: str
    step_name: str
    status: IdempotencyStatus
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class IdempotencyManager:
    """
    Prevent duplicate Act execution via idempotency keys.

    Key = SHA-256(instance_id + step_name + input_hash)

    If key exists with status=completed → return cached result.
    If key exists with status=in_progress → raise (or wait).
    If key not found → create with in_progress, execute, mark completed.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS act_executions (
                idempotency_key TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_act_instance
                ON act_executions(instance_id);
        """)

    @staticmethod
    def compute_key(instance_id: str, step_name: str, input_data: Any) -> str:
        """Compute deterministic idempotency key."""
        raw = f"{instance_id}:{step_name}:{json.dumps(input_data, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def check(self, key: str) -> IdempotencyRecord | None:
        """Check if an idempotency key exists."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM act_executions WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            return IdempotencyRecord(
                key=row["idempotency_key"],
                instance_id=row["instance_id"],
                step_name=row["step_name"],
                status=IdempotencyStatus(row["status"]),
                result=json.loads(row["result_json"]) if row["result_json"] else None,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def acquire(self, key: str, instance_id: str, step_name: str) -> IdempotencyRecord | None:
        """
        Attempt to acquire the idempotency key for execution.

        Returns:
            None if acquired (proceed with execution)
            IdempotencyRecord if key exists (use cached result or wait)
        """
        with self._lock:
            existing = self.check(key)
            if existing:
                return existing

            now = time.time()
            self._conn.execute("""
                INSERT INTO act_executions
                (idempotency_key, instance_id, step_name, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key, instance_id, step_name, IdempotencyStatus.IN_PROGRESS.value, now, now))
            self._conn.commit()
            return None  # Acquired — proceed

    def complete(self, key: str, result: dict[str, Any]) -> None:
        """Mark execution as completed with result."""
        with self._lock:
            self._conn.execute("""
                UPDATE act_executions
                SET status = ?, result_json = ?, updated_at = ?
                WHERE idempotency_key = ?
            """, (IdempotencyStatus.COMPLETED.value, json.dumps(result, default=str), time.time(), key))
            self._conn.commit()

    def fail(self, key: str, error: str = "") -> None:
        """Mark execution as failed."""
        with self._lock:
            self._conn.execute("""
                UPDATE act_executions
                SET status = ?, result_json = ?, updated_at = ?
                WHERE idempotency_key = ?
            """, (IdempotencyStatus.FAILED.value, json.dumps({"error": error}), time.time(), key))
            self._conn.commit()

    def release(self, key: str) -> None:
        """Remove a key (for retry after failure)."""
        with self._lock:
            self._conn.execute("DELETE FROM act_executions WHERE idempotency_key = ?", (key,))
            self._conn.commit()

    def get_executions(self, instance_id: str) -> list[IdempotencyRecord]:
        """Get all executions for an instance."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM act_executions WHERE instance_id = ? ORDER BY created_at",
                (instance_id,),
            ).fetchall()
            return [
                IdempotencyRecord(
                    key=r["idempotency_key"],
                    instance_id=r["instance_id"],
                    step_name=r["step_name"],
                    status=IdempotencyStatus(r["status"]),
                    result=json.loads(r["result_json"]) if r["result_json"] else None,
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def close(self):
        self._conn.close()
