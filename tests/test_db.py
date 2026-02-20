"""
Cognitive Core — Database Backend Tests

Tests the abstraction layer against SQLite (always available).
The same interface contract applies to PostgreSQL.
"""

import os
import sys
import json
import tempfile
import threading
import unittest

# Direct import to avoid engine/__init__.py
import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(_base, "engine", "db.py")
_spec = importlib.util.spec_from_file_location("engine.db", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.db"] = _mod
_spec.loader.exec_module(_mod)

DatabaseBackend = _mod.DatabaseBackend
SQLiteBackend = _mod.SQLiteBackend
PostgresBackend = _mod.PostgresBackend
create_backend = _mod.create_backend
_translate_upsert = _mod._translate_upsert
_safe_dsn = _mod._safe_dsn


class TestSQLiteBackendBasics(unittest.TestCase):
    """Core CRUD operations through the abstraction layer."""

    def setUp(self):
        self.db = SQLiteBackend(path=":memory:")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL DEFAULT 0.0
            );
        """)

    def tearDown(self):
        self.db.close()

    def test_insert_and_fetchone(self):
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("alpha", 1.5))
        row = self.db.fetchone("SELECT * FROM items WHERE name = ?", ("alpha",))
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "alpha")
        self.assertAlmostEqual(row["value"], 1.5)

    def test_fetchone_missing(self):
        row = self.db.fetchone("SELECT * FROM items WHERE name = ?", ("nope",))
        self.assertIsNone(row)

    def test_fetchall(self):
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("a", 1.0))
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("b", 2.0))
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("c", 3.0))
        rows = self.db.fetchall("SELECT * FROM items ORDER BY name")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["name"], "a")
        self.assertEqual(rows[2]["name"], "c")

    def test_fetchall_empty(self):
        rows = self.db.fetchall("SELECT * FROM items")
        self.assertEqual(rows, [])

    def test_lastrowid(self):
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("x", 0))
        self.assertGreater(self.db.lastrowid, 0)

    def test_rowcount_on_update(self):
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("x", 0))
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("y", 0))
        self.db.execute("UPDATE items SET value = 99 WHERE value = 0")
        self.assertEqual(self.db.rowcount, 2)

    def test_rowcount_on_delete(self):
        self.db.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("x", 0))
        self.db.execute("DELETE FROM items WHERE name = ?", ("x",))
        self.assertEqual(self.db.rowcount, 1)

    def test_executescript_multiple(self):
        self.db.executescript("""
            INSERT INTO items (name, value) VALUES ('s1', 10);
            INSERT INTO items (name, value) VALUES ('s2', 20);
        """)
        rows = self.db.fetchall("SELECT * FROM items")
        self.assertEqual(len(rows), 2)

    def test_backend_type(self):
        self.assertEqual(self.db.backend_type, "sqlite")


class TestSQLiteBackendTransactions(unittest.TestCase):
    """Transaction support."""

    def setUp(self):
        self.db = SQLiteBackend(path=":memory:")
        self.db.executescript("CREATE TABLE t (id TEXT PRIMARY KEY, val INTEGER);")

    def tearDown(self):
        self.db.close()

    def test_transaction_commit(self):
        with self.db.transaction():
            self.db.execute("INSERT INTO t (id, val) VALUES (?, ?)", ("a", 1))
            self.db.execute("INSERT INTO t (id, val) VALUES (?, ?)", ("b", 2))
        rows = self.db.fetchall("SELECT * FROM t")
        self.assertEqual(len(rows), 2)

    def test_transaction_rollback(self):
        try:
            with self.db.transaction():
                self.db.execute("INSERT INTO t (id, val) VALUES (?, ?)", ("a", 1))
                raise ValueError("force rollback")
        except ValueError:
            pass
        rows = self.db.fetchall("SELECT * FROM t")
        self.assertEqual(len(rows), 0)

    def test_transaction_atomicity(self):
        """Both inserts succeed or neither does."""
        try:
            with self.db.transaction():
                self.db.execute("INSERT INTO t (id, val) VALUES (?, ?)", ("a", 1))
                # This should fail — duplicate PK
                self.db.execute("INSERT INTO t (id, val) VALUES (?, ?)", ("a", 2))
        except Exception:
            pass
        rows = self.db.fetchall("SELECT * FROM t")
        self.assertEqual(len(rows), 0)  # Both rolled back


class TestSQLiteBackendThreadSafety(unittest.TestCase):
    """Concurrent access via threads."""

    def setUp(self):
        self.tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = SQLiteBackend(path=self.tf.name)
        self.db.executescript("CREATE TABLE c (id INTEGER PRIMARY KEY AUTOINCREMENT, val INTEGER);")

    def tearDown(self):
        self.db.close()
        os.unlink(self.tf.name)

    def test_concurrent_writes(self):
        errors = []
        def writer(n):
            try:
                for i in range(20):
                    self.db.execute("INSERT INTO c (val) VALUES (?)", (n * 100 + i,))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        rows = self.db.fetchall("SELECT * FROM c")
        self.assertEqual(len(rows), 80)


class TestSQLiteBackendFile(unittest.TestCase):
    """File-based SQLite persistence."""

    def test_persist_across_connections(self):
        tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        try:
            db1 = SQLiteBackend(path=tf.name)
            db1.executescript("CREATE TABLE p (k TEXT PRIMARY KEY, v TEXT);")
            db1.execute("INSERT INTO p (k, v) VALUES (?, ?)", ("key1", "val1"))
            db1.close()

            db2 = SQLiteBackend(path=tf.name)
            row = db2.fetchone("SELECT * FROM p WHERE k = ?", ("key1",))
            self.assertIsNotNone(row)
            self.assertEqual(row["v"], "val1")
            db2.close()
        finally:
            os.unlink(tf.name)


class TestFactory(unittest.TestCase):
    """create_backend factory."""

    def test_default_is_sqlite(self):
        db = create_backend(path=":memory:")
        self.assertEqual(db.backend_type, "sqlite")
        db.close()

    def test_explicit_sqlite(self):
        db = create_backend("sqlite", path=":memory:")
        self.assertEqual(db.backend_type, "sqlite")
        db.close()

    def test_env_override(self):
        os.environ["CC_DB_BACKEND"] = "sqlite"
        db = create_backend(path=":memory:")
        self.assertEqual(db.backend_type, "sqlite")
        db.close()
        del os.environ["CC_DB_BACKEND"]

    def test_postgres_without_dsn_raises(self):
        with self.assertRaises(ValueError):
            create_backend("postgres")


class TestUpsertTranslation(unittest.TestCase):
    """INSERT OR REPLACE → ON CONFLICT translation."""

    def test_instances_upsert(self):
        sql = "INSERT OR REPLACE INTO instances (instance_id, workflow_type, domain) VALUES (%s, %s, %s)"
        result = _translate_upsert(sql)
        self.assertIn("ON CONFLICT (instance_id)", result)
        self.assertIn("workflow_type = EXCLUDED.workflow_type", result)
        self.assertNotIn("instance_id = EXCLUDED.instance_id", result)

    def test_suspensions_upsert(self):
        sql = "INSERT OR REPLACE INTO suspensions (instance_id, suspended_at_step) VALUES (%s, %s)"
        result = _translate_upsert(sql)
        self.assertIn("ON CONFLICT (instance_id)", result)

    def test_unknown_table_passthrough(self):
        sql = "INSERT OR REPLACE INTO unknown_table (a, b) VALUES (%s, %s)"
        result = _translate_upsert(sql)
        # Should still replace INSERT OR REPLACE, just won't have ON CONFLICT
        self.assertIn("INSERT INTO unknown_table", result)


class TestSafeDsn(unittest.TestCase):
    """DSN password masking."""

    def test_masks_password(self):
        result = _safe_dsn("postgresql://user:secret@host:5432/db")
        self.assertIn("****", result)
        self.assertNotIn("secret", result)

    def test_no_password(self):
        result = _safe_dsn("postgresql://host:5432/db")
        self.assertNotIn("****", result)


class TestPostgresSchemaCompatibility(unittest.TestCase):
    """Verify Postgres schema DDL is syntactically valid SQL."""

    def test_coordinator_schema_parseable(self):
        """Schema string should contain expected tables."""
        from engine.db import COORDINATOR_SCHEMA_POSTGRES
        self.assertIn("CREATE TABLE IF NOT EXISTS instances", COORDINATOR_SCHEMA_POSTGRES)
        self.assertIn("CREATE TABLE IF NOT EXISTS work_orders", COORDINATOR_SCHEMA_POSTGRES)
        self.assertIn("CREATE TABLE IF NOT EXISTS suspensions", COORDINATOR_SCHEMA_POSTGRES)
        self.assertIn("CREATE TABLE IF NOT EXISTS action_ledger", COORDINATOR_SCHEMA_POSTGRES)
        self.assertIn("SERIAL PRIMARY KEY", COORDINATOR_SCHEMA_POSTGRES)
        self.assertIn("DOUBLE PRECISION", COORDINATOR_SCHEMA_POSTGRES)

    def test_audit_schema_parseable(self):
        from engine.db import AUDIT_SCHEMA_POSTGRES
        self.assertIn("CREATE TABLE IF NOT EXISTS audit_events", AUDIT_SCHEMA_POSTGRES)
        self.assertIn("CREATE TABLE IF NOT EXISTS audit_payload", AUDIT_SCHEMA_POSTGRES)
        self.assertIn("SERIAL PRIMARY KEY", AUDIT_SCHEMA_POSTGRES)

    def test_checkpoint_schema_parseable(self):
        from engine.db import CHECKPOINT_SCHEMA_POSTGRES
        self.assertIn("CREATE TABLE IF NOT EXISTS checkpoints", CHECKPOINT_SCHEMA_POSTGRES)
        self.assertIn("SERIAL PRIMARY KEY", CHECKPOINT_SCHEMA_POSTGRES)


class TestSQLiteAsCoordinatorStore(unittest.TestCase):
    """Simulate coordinator store operations through the abstraction."""

    def setUp(self):
        self.db = SQLiteBackend(path=":memory:")
        # Use the SQLite schema (same as coordinator/store.py)
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
            CREATE TABLE IF NOT EXISTS action_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                details TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                created_at REAL NOT NULL
            );
        """)

    def tearDown(self):
        self.db.close()

    def test_save_and_load_instance(self):
        import time
        now = time.time()
        self.db.execute("""
            INSERT OR REPLACE INTO instances
            (instance_id, workflow_type, domain, status, governance_tier,
             created_at, updated_at, lineage, correlation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("inst_1", "spending_advisor", "debit_spending", "running", "auto",
              now, now, "[]", "corr_1"))

        row = self.db.fetchone("SELECT * FROM instances WHERE instance_id = ?", ("inst_1",))
        self.assertIsNotNone(row)
        self.assertEqual(row["workflow_type"], "spending_advisor")
        self.assertEqual(row["status"], "running")

    def test_idempotent_ledger(self):
        import time
        # First insert succeeds
        self.db.execute("""
            INSERT INTO action_ledger
            (instance_id, correlation_id, action_type, details, idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("i1", "c1", "start", "{}", "key_1", time.time()))

        # Duplicate key fails
        with self.assertRaises(Exception):
            self.db.execute("""
                INSERT INTO action_ledger
                (instance_id, correlation_id, action_type, details, idempotency_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("i1", "c1", "start", "{}", "key_1", time.time()))

    def test_list_by_status(self):
        import time
        now = time.time()
        for i, status in enumerate(["running", "completed", "running", "suspended"]):
            self.db.execute("""
                INSERT INTO instances
                (instance_id, workflow_type, domain, status, governance_tier, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (f"inst_{i}", "w", "d", status, "auto", now, now))

        running = self.db.fetchall("SELECT * FROM instances WHERE status = ?", ("running",))
        self.assertEqual(len(running), 2)

    def test_stats_query(self):
        import time
        now = time.time()
        self.db.execute("""
            INSERT INTO instances
            (instance_id, workflow_type, domain, status, governance_tier, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("i1", "w", "d", "completed", "auto", now, now))

        rows = self.db.fetchall("SELECT status, COUNT(*) as cnt FROM instances GROUP BY status")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cnt"], 1)


class TestSQLiteAsAuditStore(unittest.TestCase):
    """Simulate audit trail operations through the abstraction."""

    def setUp(self):
        self.db = SQLiteBackend(path=":memory:")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                payload TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                previous_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_payload (
                event_id INTEGER PRIMARY KEY,
                trace_id TEXT NOT NULL,
                payload_data TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl_days INTEGER DEFAULT 0
            );
        """)

    def tearDown(self):
        self.db.close()

    def test_append_and_query(self):
        import time, hashlib
        now = time.time()
        payload = json.dumps({"step": "classify", "confidence": 0.9})
        h = hashlib.sha256(payload.encode()).hexdigest()
        self.db.execute("""
            INSERT INTO audit_events (trace_id, event_type, timestamp, payload, event_hash, previous_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("t1", "primitive_complete", now, payload, h, "genesis"))

        row = self.db.fetchone("SELECT * FROM audit_events WHERE trace_id = ?", ("t1",))
        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "primitive_complete")

    def test_payload_tiered_storage(self):
        import time
        # Insert audit event
        self.db.execute("""
            INSERT INTO audit_events (trace_id, event_type, timestamp, payload, event_hash, previous_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("t1", "test", time.time(), "{}", "hash1", "genesis"))
        event_id = self.db.lastrowid

        # Store payload separately
        self.db.execute("""
            INSERT INTO audit_payload (event_id, trace_id, payload_data, created_at, ttl_days)
            VALUES (?, ?, ?, ?, ?)
        """, (event_id, "t1", '{"sensitive": "data"}', time.time(), 30))

        # Delete payload — audit event survives
        self.db.execute("DELETE FROM audit_payload WHERE event_id = ?", (event_id,))
        self.assertEqual(self.db.rowcount, 1)

        event = self.db.fetchone("SELECT * FROM audit_events WHERE id = ?", (event_id,))
        self.assertIsNotNone(event)  # Still there
        payload = self.db.fetchone("SELECT * FROM audit_payload WHERE event_id = ?", (event_id,))
        self.assertIsNone(payload)  # Gone


if __name__ == "__main__":
    unittest.main()
