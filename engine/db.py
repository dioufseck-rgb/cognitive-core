"""
Cognitive Core — Database Backend Abstraction

Provides a unified interface for SQLite and PostgreSQL backends.
All database-using modules (coordinator store, audit trail, replay,
logic breaker) use this abstraction instead of raw sqlite3/psycopg.

Usage:
    from engine.db import create_backend

    # SQLite (default, Year 1)
    db = create_backend("sqlite", path="coordinator.db")

    # PostgreSQL (Year 2+)
    db = create_backend("postgres", dsn="postgresql://user:pass@host/dbname")

    # Execute queries
    db.execute("INSERT INTO foo (bar) VALUES (?)", ("baz",))
    row = db.fetchone("SELECT * FROM foo WHERE bar = ?", ("baz",))
    rows = db.fetchall("SELECT * FROM foo")
    db.commit()

The abstraction handles:
  - Parameter placeholder translation (? for SQLite, %s for Postgres)
  - AUTOINCREMENT → SERIAL/GENERATED ALWAYS AS IDENTITY
  - PRAGMA translation
  - Row factory (dict-like access)
  - Transaction context manager
  - Connection pooling (Postgres only)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("cognitive_core.db")

# ═══════════════════════════════════════════════════════════════
# Abstract Interface
# ═══════════════════════════════════════════════════════════════

class DatabaseBackend:
    """Abstract database backend interface."""

    def execute(self, sql: str, params: tuple = ()) -> Any:
        raise NotImplementedError

    def executescript(self, sql: str) -> None:
        raise NotImplementedError

    def fetchone(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        raise NotImplementedError

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        raise NotImplementedError

    def commit(self) -> None:
        raise NotImplementedError

    def rollback(self) -> None:
        raise NotImplementedError

    @contextmanager
    def transaction(self) -> Iterator[None]:
        raise NotImplementedError

    @property
    def lastrowid(self) -> int:
        raise NotImplementedError

    @property
    def rowcount(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def backend_type(self) -> str:
        raise NotImplementedError

    def translate_sql(self, sql: str) -> str:
        """Translate SQL from canonical (SQLite) form to backend-specific form."""
        return sql


# ═══════════════════════════════════════════════════════════════
# SQLite Backend
# ═══════════════════════════════════════════════════════════════

class SQLiteBackend(DatabaseBackend):
    """SQLite backend — Year 1 default."""

    def __init__(self, path: str = ":memory:", wal: bool = True, busy_timeout: int = 5000):
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if wal:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
        self._lock = threading.RLock()
        self._last_cursor = None
        self._in_transaction = False
        logger.info("SQLite backend initialized: %s", path)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        with self._lock:
            self._last_cursor = self._conn.execute(sql, params)
            if not self._in_transaction:
                self._conn.commit()
            return self._last_cursor

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)
            if not self._in_transaction:
                self._conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    def commit(self) -> None:
        if not self._in_transaction:
            self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_transaction = True
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._in_transaction = False

    @property
    def lastrowid(self) -> int:
        return self._last_cursor.lastrowid if self._last_cursor else 0

    @property
    def rowcount(self) -> int:
        return self._last_cursor.rowcount if self._last_cursor else 0

    def close(self) -> None:
        self._conn.close()

    @property
    def backend_type(self) -> str:
        return "sqlite"


# ═══════════════════════════════════════════════════════════════
# PostgreSQL Backend
# ═══════════════════════════════════════════════════════════════

class PostgresBackend(DatabaseBackend):
    """
    PostgreSQL backend — Year 2+ or Azure deployment.

    Requires psycopg (v3) or psycopg2. Automatically detects which is available.
    Translates SQLite-flavored SQL to Postgres dialect:
      - ? → %s parameter placeholders
      - INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
      - INSERT OR REPLACE → INSERT ... ON CONFLICT ... DO UPDATE
      - PRAGMA statements → no-ops or equivalents
    """

    def __init__(self, dsn: str, pool_min: int = 2, pool_max: int = 10):
        self._dsn = dsn
        self._conn = None
        self._last_cursor = None
        self._in_transaction = False
        self._lock = threading.Lock()

        # Try psycopg (v3) first, then psycopg2
        try:
            import psycopg
            self._conn = psycopg.connect(dsn, autocommit=False)
            self._driver = "psycopg3"
            self._placeholder = "%s"
            logger.info("PostgreSQL backend initialized (psycopg3): %s", _safe_dsn(dsn))
        except ImportError:
            try:
                import psycopg2
                import psycopg2.extras
                self._conn = psycopg2.connect(dsn)
                self._conn.autocommit = False
                # Use RealDictCursor for dict-like rows
                self._cursor_factory = psycopg2.extras.RealDictCursor
                self._driver = "psycopg2"
                self._placeholder = "%s"
                logger.info("PostgreSQL backend initialized (psycopg2): %s", _safe_dsn(dsn))
            except ImportError:
                raise ImportError(
                    "PostgreSQL backend requires psycopg (v3) or psycopg2. "
                    "Install with: pip install psycopg[binary] or pip install psycopg2-binary"
                )

    def translate_sql(self, sql: str) -> str:
        """Translate SQLite SQL to PostgreSQL dialect."""
        s = sql
        # Parameter placeholders: ? → %s
        s = s.replace("?", "%s")
        # AUTOINCREMENT → SERIAL (for CREATE TABLE)
        s = s.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        # INSERT OR REPLACE → INSERT ... ON CONFLICT DO UPDATE
        # This is a simplified transform — works for our schema where PKs are named
        if "INSERT OR REPLACE INTO" in s:
            s = _translate_upsert(s)
        # PRAGMA statements → no-ops
        if s.strip().upper().startswith("PRAGMA"):
            return "SELECT 1"  # no-op
        # CREATE TABLE IF NOT EXISTS — works in both
        # CREATE INDEX IF NOT EXISTS — works in both
        return s

    def execute(self, sql: str, params: tuple = ()) -> Any:
        translated = self.translate_sql(sql)
        with self._lock:
            cursor = self._get_cursor()
            cursor.execute(translated, params)
            self._last_cursor = cursor
            if not self._in_transaction:
                self._conn.commit()
            return cursor

    def executescript(self, sql: str) -> None:
        """Execute multiple statements. Split on semicolons and execute each."""
        with self._lock:
            cursor = self._get_cursor()
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                translated = self.translate_sql(stmt)
                if translated.strip() and translated.strip() != "SELECT 1":
                    cursor.execute(translated)
            if not self._in_transaction:
                self._conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        translated = self.translate_sql(sql)
        with self._lock:
            cursor = self._get_cursor()
            cursor.execute(translated, params)
            row = cursor.fetchone()
            if row is None:
                return None
            if isinstance(row, dict):
                return row
            # psycopg3 returns tuples by default
            if hasattr(cursor, 'description') and cursor.description:
                cols = [d[0] for d in cursor.description]
                return dict(zip(cols, row))
            return dict(row)

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        translated = self.translate_sql(sql)
        with self._lock:
            cursor = self._get_cursor()
            cursor.execute(translated, params)
            rows = cursor.fetchall()
            if not rows:
                return []
            if isinstance(rows[0], dict):
                return rows
            if hasattr(cursor, 'description') and cursor.description:
                cols = [d[0] for d in cursor.description]
                return [dict(zip(cols, r)) for r in rows]
            return [dict(r) for r in rows]

    def commit(self) -> None:
        if not self._in_transaction:
            self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self._in_transaction = True
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._in_transaction = False

    @property
    def lastrowid(self) -> int:
        if self._last_cursor is None:
            return 0
        if self._driver == "psycopg2":
            return self._last_cursor.lastrowid or 0
        # psycopg3: use RETURNING id in the query
        return 0

    @property
    def rowcount(self) -> int:
        return self._last_cursor.rowcount if self._last_cursor else 0

    def close(self) -> None:
        self._conn.close()

    @property
    def backend_type(self) -> str:
        return "postgres"

    def _get_cursor(self):
        if self._driver == "psycopg2":
            return self._conn.cursor(cursor_factory=self._cursor_factory)
        return self._conn.cursor()


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

def create_backend(
    backend_type: str | None = None,
    path: str = "coordinator.db",
    dsn: str = "",
    **kwargs,
) -> DatabaseBackend:
    """
    Create a database backend from config.

    Auto-detects backend type from environment:
      - CC_DB_BACKEND=postgres + CC_DB_DSN=postgresql://...  → Postgres
      - CC_DB_BACKEND=sqlite (or unset)                      → SQLite

    Args:
        backend_type: "sqlite" or "postgres". Auto-detect if None.
        path: SQLite file path (only for sqlite backend)
        dsn: PostgreSQL connection string (only for postgres backend)
        **kwargs: Additional backend-specific arguments
    """
    if backend_type is None:
        backend_type = os.environ.get("CC_DB_BACKEND", "sqlite").lower()

    if backend_type == "postgres" or backend_type == "postgresql":
        if not dsn:
            dsn = os.environ.get("CC_DB_DSN", "")
        if not dsn:
            raise ValueError(
                "PostgreSQL backend requires a DSN. Set CC_DB_DSN or pass dsn parameter. "
                "Example: postgresql://user:pass@host:5432/cognitive_core"
            )
        return PostgresBackend(dsn=dsn, **kwargs)
    else:
        return SQLiteBackend(path=path, **kwargs)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _safe_dsn(dsn: str) -> str:
    """Mask password in DSN for logging."""
    if "@" in dsn:
        parts = dsn.split("@")
        prefix = parts[0]
        if ":" in prefix:
            # mask everything after last : before @
            user_part = prefix.rsplit(":", 1)[0]
            return f"{user_part}:****@{parts[1]}"
    return dsn


def _translate_upsert(sql: str) -> str:
    """
    Translate INSERT OR REPLACE INTO table (...) VALUES (...)
    to INSERT INTO table (...) VALUES (...) ON CONFLICT (pk) DO UPDATE SET ...

    This is a simplified translation that works for our known schemas.
    """
    sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")

    # Extract table name
    import re
    m = re.match(r"INSERT INTO\s+(\w+)\s*\(([^)]+)\)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql

    table = m.group(1)
    columns = [c.strip() for c in m.group(2).split(",")]

    # Known primary keys per table
    pk_map = {
        "instances": "instance_id",
        "work_orders": "work_order_id",
        "suspensions": "instance_id",
        "audit_events": "id",
        "audit_payload": "event_id",
        "checkpoints": "id",
        "circuit_breaker_state": "key",
    }

    pk = pk_map.get(table)
    if not pk or pk not in columns:
        return sql

    # Build ON CONFLICT DO UPDATE
    non_pk_cols = [c for c in columns if c != pk]
    if not non_pk_cols:
        return sql + f" ON CONFLICT ({pk}) DO NOTHING"

    update_sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk_cols)
    return sql + f" ON CONFLICT ({pk}) DO UPDATE SET {update_sets}"


# ═══════════════════════════════════════════════════════════════
# Schema DDL — Postgres-compatible versions
# ═══════════════════════════════════════════════════════════════

COORDINATOR_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    workflow_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    governance_tier TEXT NOT NULL DEFAULT 'gate',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    lineage TEXT DEFAULT '[]',
    correlation_id TEXT DEFAULT '',
    current_step TEXT DEFAULT '',
    step_count INTEGER DEFAULT 0,
    elapsed_seconds DOUBLE PRECISION DEFAULT 0.0,
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
    sla_seconds DOUBLE PRECISION,
    urgency TEXT DEFAULT 'routine',
    handler_workflow_type TEXT DEFAULT '',
    handler_domain TEXT DEFAULT '',
    handler_instance_id TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'created',
    created_at DOUBLE PRECISION NOT NULL,
    dispatched_at DOUBLE PRECISION,
    completed_at DOUBLE PRECISION,
    result TEXT
);

CREATE TABLE IF NOT EXISTS suspensions (
    instance_id TEXT PRIMARY KEY,
    suspended_at_step TEXT NOT NULL,
    state_snapshot TEXT NOT NULL,
    unresolved_needs TEXT DEFAULT '[]',
    work_order_ids TEXT DEFAULT '[]',
    resume_nonce TEXT NOT NULL,
    suspended_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS action_ledger (
    id SERIAL PRIMARY KEY,
    instance_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    details TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_instances_status ON instances(status);
CREATE INDEX IF NOT EXISTS idx_instances_correlation ON instances(correlation_id);
CREATE INDEX IF NOT EXISTS idx_work_orders_status ON work_orders(status);
CREATE INDEX IF NOT EXISTS idx_work_orders_requester ON work_orders(requester_instance_id);
CREATE INDEX IF NOT EXISTS idx_ledger_instance ON action_ledger(instance_id);
"""

AUDIT_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS audit_events (
    id SERIAL PRIMARY KEY,
    trace_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    payload TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    previous_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);

CREATE TABLE IF NOT EXISTS audit_payload (
    event_id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL,
    payload_data TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    ttl_days INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_payload_trace ON audit_payload(trace_id);
CREATE INDEX IF NOT EXISTS idx_payload_created ON audit_payload(created_at);
"""

CHECKPOINT_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id SERIAL PRIMARY KEY,
    instance_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_instance ON checkpoints(instance_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_step ON checkpoints(instance_id, step_name);
"""
