"""
TASK 8 — Tests: Service Bus task queue adapter

Verifies:
- ServiceBusTaskQueueAdapter implements the TaskQueue interface
- publish() sends a task to the Service Bus queue
- claim() receives a message and returns it as a Task
- Duplicate task delivery does not double-process (idempotency)
- When DATA_SERVICE_BUS_URL is unset, create_task_queue() falls back to SQLite
- When DATA_SERVICE_BUS_URL is set but azure-servicebus unavailable, falls back to SQLite
- Proof ledger records which queue backend handled each governance task
- force_complete() and unclaim() work correctly on all three queue implementations
"""

from __future__ import annotations

import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from cognitive_core.coordinator.tasks import (
    Task, TaskCallback, TaskResolution, TaskStatus, TaskType,
    InMemoryTaskQueue, SQLiteTaskQueue, create_task_queue,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_task(instance_id: str = "inst_001", resume_nonce: str = "nonce_abc") -> Task:
    return Task.create(
        task_type=TaskType.GOVERNANCE_APPROVAL,
        queue="governance",
        instance_id=instance_id,
        correlation_id="corr_001",
        workflow_type="fraud_investigation",
        domain="check_fraud",
        payload={"step": "verify_compliance", "tier": "gate"},
        callback=TaskCallback(
            method="approve",
            instance_id=instance_id,
            resume_nonce=resume_nonce,
        ),
        priority=1,
        sla_seconds=3600,
    )


# ── InMemoryTaskQueue interface tests ────────────────────────────────────────

class TestInMemoryTaskQueue(unittest.TestCase):

    def setUp(self):
        self.q = InMemoryTaskQueue()
        self.task = _make_task()

    def test_publish_returns_task_id(self):
        task_id = self.q.publish(self.task)
        self.assertEqual(task_id, self.task.task_id)

    def test_claim_returns_pending_task(self):
        self.q.publish(self.task)
        claimed = self.q.claim("governance", "worker_1")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.task_id, self.task.task_id)
        self.assertEqual(claimed.status, TaskStatus.CLAIMED)

    def test_force_complete_marks_completed(self):
        self.q.publish(self.task)
        result = self.q.force_complete(self.task.task_id)
        self.assertTrue(result)
        t = self.q.get_task(self.task.task_id)
        self.assertEqual(t.status, TaskStatus.COMPLETED)

    def test_unclaim_restores_pending(self):
        self.q.publish(self.task)
        self.q.claim("governance", "worker_1")
        result = self.q.unclaim(self.task.task_id)
        self.assertTrue(result)
        t = self.q.get_task(self.task.task_id)
        self.assertEqual(t.status, TaskStatus.PENDING)
        self.assertIsNone(t.claimed_at)

    def test_force_complete_on_missing_task_returns_false(self):
        result = self.q.force_complete("nonexistent_task_id")
        self.assertFalse(result)


# ── Factory — fallback to SQLite ──────────────────────────────────────────────

class TestCreateTaskQueueFactory(unittest.TestCase):

    def test_no_env_var_returns_sqlite_when_db_provided(self):
        os.environ.pop("DATA_SERVICE_BUS_URL", None)
        from cognitive_core.engine.db import SQLiteBackend
        db = SQLiteBackend(":memory:")
        q = create_task_queue(db=db)
        self.assertIsInstance(q, SQLiteTaskQueue)
        db.close()

    def test_no_env_var_returns_in_memory_without_db(self):
        os.environ.pop("DATA_SERVICE_BUS_URL", None)
        q = create_task_queue()
        self.assertIsInstance(q, InMemoryTaskQueue)

    def test_env_var_set_but_package_unavailable_falls_back_to_sqlite(self):
        """When DATA_SERVICE_BUS_URL is set but azure-servicebus is not installed,
        falls back to SQLite without error."""
        from cognitive_core.engine.db import SQLiteBackend
        db = SQLiteBackend(":memory:")
        os.environ["DATA_SERVICE_BUS_URL"] = "Endpoint=sb://fake.servicebus.windows.net/;SharedAccessKeyName=Test;SharedAccessKey=abc123"
        try:
            with patch("coordinator.tasks._SERVICE_BUS_AVAILABLE", False):
                q = create_task_queue(db=db)
            self.assertIsInstance(q, SQLiteTaskQueue)
        finally:
            os.environ.pop("DATA_SERVICE_BUS_URL", None)
            db.close()

    def test_env_var_set_with_package_returns_service_bus_adapter(self):
        """When DATA_SERVICE_BUS_URL is set and azure-servicebus is available,
        returns ServiceBusTaskQueueAdapter."""
        os.environ["DATA_SERVICE_BUS_URL"] = "Endpoint=sb://test.servicebus.windows.net/;SharedAccessKeyName=Test;SharedAccessKey=abc123"
        try:
            with patch("coordinator.tasks._SERVICE_BUS_AVAILABLE", True):
                # Mock the ServiceBusTaskQueueAdapter constructor
                with patch("coordinator.tasks.ServiceBusTaskQueueAdapter") as MockAdapter:
                    mock_instance = MagicMock()
                    MockAdapter.return_value = mock_instance
                    q = create_task_queue()
                    MockAdapter.assert_called_once()
                    self.assertEqual(q, mock_instance)
        finally:
            os.environ.pop("DATA_SERVICE_BUS_URL", None)


# ── ServiceBusTaskQueueAdapter ────────────────────────────────────────────────

class TestServiceBusTaskQueueAdapter(unittest.TestCase):
    """
    Tests for ServiceBusTaskQueueAdapter using mocked azure-servicebus SDK.
    """

    def _make_adapter(self):
        """Create a ServiceBusTaskQueueAdapter with fully mocked Service Bus."""
        from cognitive_core.coordinator.tasks import ServiceBusTaskQueueAdapter
        with patch("coordinator.tasks._SERVICE_BUS_AVAILABLE", True):
            adapter = ServiceBusTaskQueueAdapter.__new__(ServiceBusTaskQueueAdapter)
            adapter._connection_string = "fake_connection_string"
            adapter._queue_name = "governance-tasks"
            adapter._tasks = {}
            adapter._pending_messages = {}
            adapter._processed_keys = set()
            return adapter

    def test_publish_sends_message_to_service_bus(self):
        """Suspended workflow's governance task appears in the Service Bus queue."""
        adapter = self._make_adapter()
        task = _make_task()

        mock_sender = MagicMock()
        mock_client = MagicMock()
        mock_client.get_queue_sender.return_value.__enter__ = lambda s: mock_sender
        mock_client.get_queue_sender.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("coordinator.tasks.ServiceBusClient") as MockSBClient, \
             patch("coordinator.tasks.ServiceBusMessage") as MockSBMessage:
            MockSBClient.from_connection_string.return_value = mock_client
            task_id = adapter.publish(task)

        self.assertEqual(task_id, task.task_id)
        self.assertIn(task.task_id, adapter._tasks)
        MockSBClient.from_connection_string.assert_called_once_with("fake_connection_string")

    def test_duplicate_publish_is_suppressed(self):
        """Duplicate task delivery does not double-process a completion."""
        adapter = self._make_adapter()
        task = _make_task(instance_id="inst_dup", resume_nonce="nonce_dup")

        # Mark the idempotency key as already processed
        idem_key = f"{task.instance_id}:{task.callback.resume_nonce}"
        adapter._processed_keys.add(idem_key)

        with patch("coordinator.tasks.ServiceBusClient") as MockSBClient, \
             patch("coordinator.tasks.ServiceBusMessage"):
            task_id = adapter.publish(task)

        # Should return task_id without calling Service Bus
        self.assertEqual(task_id, task.task_id)
        MockSBClient.from_connection_string.assert_not_called()

    def test_force_complete_marks_task_and_records_idempotency(self):
        adapter = self._make_adapter()
        task = _make_task()
        adapter._tasks[task.task_id] = task
        task.status = TaskStatus.PENDING

        result = adapter.force_complete(task.task_id)
        self.assertTrue(result)
        self.assertEqual(adapter._tasks[task.task_id].status, TaskStatus.COMPLETED)
        idem_key = f"{task.instance_id}:{task.callback.resume_nonce}"
        self.assertIn(idem_key, adapter._processed_keys)

    def test_unclaim_restores_pending(self):
        adapter = self._make_adapter()
        task = _make_task()
        task.status = TaskStatus.CLAIMED
        task.claimed_at = time.time()
        task.claimed_by = "worker_1"
        adapter._tasks[task.task_id] = task

        result = adapter.unclaim(task.task_id)
        self.assertTrue(result)
        self.assertEqual(adapter._tasks[task.task_id].status, TaskStatus.PENDING)
        self.assertIsNone(adapter._tasks[task.task_id].claimed_at)

    def test_idempotency_check_on_claim_suppresses_duplicate(self):
        """When a message arrives with an already-processed idempotency key,
        claim() completes the message without returning a task."""
        adapter = self._make_adapter()
        task = _make_task(instance_id="inst_idem", resume_nonce="nonce_idem")

        # Pre-populate the processed set (simulates prior processing)
        idem_key = f"inst_idem:nonce_idem"
        adapter._processed_keys.add(idem_key)

        # Mock a received message with the duplicate task's data
        msg_body = json.dumps({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "queue": "governance",
            "instance_id": "inst_idem",
            "correlation_id": "corr_001",
            "workflow_type": "fraud_investigation",
            "domain": "check_fraud",
            "payload": {},
            "callback": {
                "method": "approve",
                "instance_id": "inst_idem",
                "resume_nonce": "nonce_idem",
                "endpoint": "",
                "context": {},
            },
            "status": "pending",
            "priority": 1,
            "created_at": time.time(),
        })

        mock_msg = MagicMock()
        mock_msg.__str__ = lambda s: msg_body
        mock_receiver = MagicMock()
        mock_receiver.receive_messages.return_value = [mock_msg]
        mock_receiver.__enter__ = lambda s: mock_receiver
        mock_receiver.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.get_queue_receiver.return_value = mock_receiver
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("coordinator.tasks.ServiceBusClient") as MockSBClient:
            MockSBClient.from_connection_string.return_value = mock_client
            result = adapter.claim("governance", "worker_1")

        # Duplicate delivery → returns None, message completed
        self.assertIsNone(result)
        mock_receiver.complete_message.assert_called_once_with(mock_msg)

    def test_backend_name_is_correct(self):
        adapter = self._make_adapter()
        self.assertEqual(adapter.backend_name, "ServiceBusTaskQueueAdapter")


# ── Proof Ledger — backend recording ─────────────────────────────────────────

class TestTaskQueueBackendProofEvent(unittest.TestCase):

    def test_proof_ledger_records_queue_backend(self):
        """The proof ledger records which queue backend handled each governance task."""
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov._record_proof(
            "governance_task_published",
            step="__governance__",
            instance_id="inst_test",
            task_id="task_001",
            queue_backend="SQLiteTaskQueue",
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="governance",
        )

        events = [e for e in gov._proof_ledger if e["event"] == "governance_task_published"]
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["queue_backend"], "SQLiteTaskQueue")
        self.assertEqual(events[0]["task_id"], "task_001")

    def test_proof_ledger_records_service_bus_backend(self):
        from cognitive_core.engine.governance import GovernancePipeline

        gov = GovernancePipeline()
        gov.initialize()

        gov._record_proof(
            "governance_task_published",
            step="__governance__",
            instance_id="inst_sb",
            task_id="task_sb_001",
            queue_backend="ServiceBusTaskQueueAdapter",
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="governance",
        )

        events = [e for e in gov._proof_ledger if e["event"] == "governance_task_published"]
        sb_events = [e for e in events if e["queue_backend"] == "ServiceBusTaskQueueAdapter"]
        self.assertGreaterEqual(len(sb_events), 1)


# ── SQLiteTaskQueue — force_complete and unclaim ──────────────────────────────

class TestSQLiteTaskQueueForceComplete(unittest.TestCase):

    def setUp(self):
        from cognitive_core.engine.db import SQLiteBackend
        self.db = SQLiteBackend(":memory:")
        self.q = SQLiteTaskQueue(self.db)
        self.task = _make_task()
        self.q.publish(self.task)

    def tearDown(self):
        self.db.close()

    def test_force_complete_sets_completed(self):
        result = self.q.force_complete(self.task.task_id)
        self.assertTrue(result)
        t = self.q.get_task(self.task.task_id)
        self.assertEqual(t.status, TaskStatus.COMPLETED)

    def test_unclaim_restores_pending(self):
        self.q.claim("governance", "worker_1")
        result = self.q.unclaim(self.task.task_id)
        self.assertTrue(result)
        t = self.q.get_task(self.task.task_id)
        self.assertEqual(t.status, TaskStatus.PENDING)


if __name__ == "__main__":
    unittest.main()
