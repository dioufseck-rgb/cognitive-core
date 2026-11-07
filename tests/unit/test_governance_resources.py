"""Database lifecycle regressions, including Python 3.13 ResourceWarnings.

Tracking explicit close() calls makes the regressions observable on earlier
Python versions too; garbage collection must not be the cleanup mechanism.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from cognitive_core.engine.compensation import CompensationLedger
from cognitive_core.engine.governance import (
    GovernancePipeline,
    get_governance,
    reset_governance,
)
from cognitive_core.engine.logic_breaker import BreakerState, LogicCircuitBreaker
from cognitive_core.engine.replay import ReplayManager


class TrackedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_count = 0

    def close(self):
        super().close()
        self.close_count += 1


class GovernanceResourceTests(unittest.TestCase):
    def setUp(self):
        self.connections = []
        real_connect = sqlite3.connect

        def tracked_connect(*args, **kwargs):
            kwargs["factory"] = TrackedConnection
            connection = real_connect(*args, **kwargs)
            self.connections.append(connection)
            return connection

        self.addCleanup(self._close_remaining_connections)
        connect_patch = patch("sqlite3.connect", side_effect=tracked_connect)
        connect_patch.start()
        self.addCleanup(connect_patch.stop)

    def _close_remaining_connections(self):
        # Keep a failed regression from leaking into the rest of the suite.
        for connection in self.connections:
            if not connection.close_count:
                connection.close()

    def _initialize(self, pipeline):
        with patch.dict(os.environ, {"CC_REPLAY_ENABLED": "true", "CC_WEBHOOK_URL": ""}):
            with patch(
                "cognitive_core.engine.replay.ReplayManager",
                side_effect=lambda: ReplayManager(":memory:"),
            ):
                pipeline.initialize()
        self.assertIsNotNone(pipeline._compensation)
        self.assertIsNotNone(pipeline._replay_manager)

    def test_close_is_idempotent_and_pipeline_can_reinitialize(self):
        pipeline = GovernancePipeline()
        self.addCleanup(pipeline.close)
        pipeline.close()  # Closing an unused pipeline must also be safe.
        self._initialize(pipeline)
        self._initialize(pipeline)
        self.assertEqual(len(self.connections), 2)
        self.assertTrue(all(conn.close_count == 0 for conn in self.connections))

        pipeline.close()
        pipeline.close()
        self.assertTrue(all(conn.close_count == 1 for conn in self.connections))

        self._initialize(pipeline)
        self.assertEqual(len(self.connections), 4)
        pipeline.close()
        self.assertTrue(all(conn.close_count == 1 for conn in self.connections))

    def test_reset_closes_singleton_resources(self):
        reset_governance()
        self.addCleanup(reset_governance)
        pipeline = get_governance()
        self._initialize(pipeline)

        reset_governance()
        reset_governance()
        self.assertEqual(len(self.connections), 2)
        self.assertTrue(all(conn.close_count == 1 for conn in self.connections))
        self.assertIsNot(get_governance(), pipeline)

    def test_singleton_closes_at_normal_interpreter_exit(self):
        script = textwrap.dedent("""
            import atexit
            import json
            import sqlite3
            from unittest.mock import patch

            connections = []
            real_connect = sqlite3.connect

            class TrackedConnection(sqlite3.Connection):
                close_count = 0

                def close(self):
                    super().close()
                    self.close_count += 1

            def tracked_connect(*args, **kwargs):
                kwargs['factory'] = TrackedConnection
                connection = real_connect(*args, **kwargs)
                connections.append(connection)
                return connection

            # Registered first, so this runs after governance's exit cleanup.
            atexit.register(lambda: print(json.dumps([c.close_count for c in connections])))

            from cognitive_core.engine.governance import get_governance
            from cognitive_core.engine.replay import ReplayManager

            with patch('sqlite3.connect', side_effect=tracked_connect):
                with patch('cognitive_core.engine.replay.ReplayManager',
                           side_effect=lambda: ReplayManager(':memory:')):
                    get_governance().initialize()
        """)
        environment = dict(os.environ, CC_REPLAY_ENABLED="true", CC_WEBHOOK_URL="")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [1, 1])

    def test_close_attempts_both_resources_when_one_fails(self):
        pipeline = GovernancePipeline()
        self.addCleanup(pipeline.close)
        self._initialize(pipeline)
        compensation_connection = pipeline._compensation._conn
        replay_connection = pipeline._replay_manager._conn

        with patch.object(pipeline._compensation, "close", side_effect=RuntimeError("close failed")):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                pipeline.close()
        self.assertEqual(replay_connection.close_count, 1)
        self.assertEqual(compensation_connection.close_count, 0)
        pipeline.close()
        self.assertEqual(compensation_connection.close_count, 1)

    def test_failed_component_initialization_closes_connection(self):
        for component, setup_method in (
            (CompensationLedger, "_init_db"),
            (ReplayManager, "_create_tables"),
        ):
            with self.subTest(component=component.__name__):
                with patch.object(component, setup_method, side_effect=sqlite3.OperationalError("setup failed")):
                    with self.assertRaisesRegex(sqlite3.OperationalError, "setup failed"):
                        component(":memory:")
                self.assertEqual(self.connections[-1].close_count, 1)

    def test_breaker_persistence_closes_every_connection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "breaker.sqlite")
            breaker = LogicCircuitBreaker(db_path=db_path)
            breaker.record("test", "classify", confidence=0.2)
            restored = LogicCircuitBreaker(db_path=db_path)
            self.assertEqual(restored._breakers[("test", "classify")].window_fill, 1)
            restored.reset("test", "classify")
            breaker.reset_all()
            self.assertTrue(self.connections)
            self.assertTrue(all(conn.close_count == 1 for conn in self.connections))

    def test_breaker_sql_errors_close_connections(self):
        breaker = LogicCircuitBreaker()
        breaker.db_path = ":memory:"
        state = BreakerState("test", "classify", deque(), breaker.window_size)
        operations = (
            ("initialize", breaker._init_db),
            ("save", lambda: breaker._save_state(state)),
            ("load", breaker._load_state),
            ("delete", lambda: breaker._delete_state("test", "classify")),
            ("clear", breaker._clear_all_state),
        )
        for name, operation in operations:
            with self.subTest(operation=name):
                previous_count = len(self.connections)
                with patch.object(TrackedConnection, "execute", side_effect=sqlite3.OperationalError("SQL failed")):
                    with self.assertRaisesRegex(sqlite3.OperationalError, "SQL failed"):
                        operation()
                self.assertEqual(len(self.connections), previous_count + 1)
                self.assertEqual(self.connections[-1].close_count, 1)


if __name__ == "__main__":
    unittest.main()
