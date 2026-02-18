"""
Cognitive Core — Worker Backends (S-001 + S-002)

Pluggable backends for case execution:
  - InlineBackend: synchronous in-process (dev/testing)
  - ThreadPoolBackend: ThreadPoolExecutor in-process (dev, bounded concurrency)
  - ArqBackend: arq + Redis async dispatch (production)

The active backend is selected by CC_WORKER_MODE env var:
  inline    → InlineBackend
  thread    → ThreadPoolBackend
  arq       → ArqBackend (default if arq+redis available)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.worker")


# ═══════════════════════════════════════════════════════════════════
# Job Tracking
# ═══════════════════════════════════════════════════════════════════

@dataclass
class JobRecord:
    """In-memory record for tracking job lifecycle."""
    job_id: str
    instance_id: str
    status: str = "queued"      # queued | running | completed | failed
    enqueued_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""


class JobTracker:
    """Thread-safe in-memory job status tracker."""

    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, instance_id: str) -> JobRecord:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        record = JobRecord(
            job_id=job_id,
            instance_id=instance_id,
            status="queued",
            enqueued_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_by_instance(self, instance_id: str) -> JobRecord | None:
        with self._lock:
            for j in self._jobs.values():
                if j.instance_id == instance_id:
                    return j
        return None

    def mark_running(self, job_id: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "running"
                self._jobs[job_id].started_at = time.time()

    def mark_completed(self, job_id: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "completed"
                self._jobs[job_id].completed_at = time.time()

    def mark_failed(self, job_id: str, error: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].completed_at = time.time()
                self._jobs[job_id].error = error[:500]

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for j in self._jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
            return counts


# ═══════════════════════════════════════════════════════════════════
# Worker Backend Interface
# ═══════════════════════════════════════════════════════════════════

class WorkerBackend:
    """Abstract interface for job dispatch."""

    def enqueue(
        self,
        instance_id: str,
        workflow: str,
        domain: str,
        case_input: dict[str, Any],
        model: str = "default",
        temperature: float = 0.1,
        correlation_id: str = "",
    ) -> str:
        """Enqueue a case for execution. Returns job_id."""
        raise NotImplementedError

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get job status."""
        raise NotImplementedError

    def shutdown(self):
        """Graceful shutdown."""
        pass


# ═══════════════════════════════════════════════════════════════════
# Inline Backend (synchronous, dev/test)
# ═══════════════════════════════════════════════════════════════════

class InlineBackend(WorkerBackend):
    """
    Synchronous in-process execution. Blocks until workflow completes.
    """

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.tracker = JobTracker()

    def enqueue(self, instance_id, workflow, domain, case_input,
                model="default", temperature=0.1, correlation_id=""):
        record = self.tracker.create(instance_id)
        self.tracker.mark_running(record.job_id)

        try:
            self._run_coordinator(
                instance_id, workflow, domain, case_input,
                model, temperature, correlation_id,
            )
            self.tracker.mark_completed(record.job_id)
        except Exception as e:
            self.tracker.mark_failed(record.job_id, str(e))
            logger.error("Inline execution failed for %s: %s", instance_id, e)

        return record.job_id

    def _run_coordinator(self, instance_id, workflow, domain, case_input,
                         model, temperature, correlation_id):
        from coordinator.runtime import Coordinator
        coord = Coordinator(project_root=self.project_root, verbose=False)
        coord.start(
            workflow_type=workflow,
            domain=domain,
            case_input=case_input,
            correlation_id=correlation_id or instance_id,
            model=model,
            temperature=temperature,
        )

    def get_job(self, job_id):
        return self.tracker.get(job_id)


# ═══════════════════════════════════════════════════════════════════
# Thread Pool Backend (S-002 Option B)
# ═══════════════════════════════════════════════════════════════════

class ThreadPoolBackend(WorkerBackend):
    """
    Async execution via ThreadPoolExecutor.
    Bounded concurrency. Coordinator runs synchronously in worker threads.
    Checkpoint/resume handles crash recovery.
    """

    def __init__(
        self,
        project_root: str = ".",
        max_workers: int = 4,
    ):
        self.project_root = project_root
        self.tracker = JobTracker()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cc_worker",
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        logger.info("ThreadPoolBackend started: max_workers=%d", max_workers)

    def enqueue(self, instance_id, workflow, domain, case_input,
                model="default", temperature=0.1, correlation_id=""):
        record = self.tracker.create(instance_id)

        future = self._pool.submit(
            self._execute,
            record.job_id,
            instance_id, workflow, domain, case_input,
            model, temperature, correlation_id,
        )

        with self._lock:
            self._futures[record.job_id] = future

        logger.info("Enqueued job %s for instance %s", record.job_id, instance_id)
        return record.job_id

    def _execute(self, job_id, instance_id, workflow, domain, case_input,
                 model, temperature, correlation_id):
        """Run in worker thread."""
        self.tracker.mark_running(job_id)
        try:
            from coordinator.runtime import Coordinator
            coord = Coordinator(project_root=self.project_root, verbose=False)
            coord.start(
                workflow_type=workflow,
                domain=domain,
                case_input=case_input,
                correlation_id=correlation_id or instance_id,
                model=model,
                temperature=temperature,
            )
            self.tracker.mark_completed(job_id)
            logger.info("Job %s completed for instance %s", job_id, instance_id)
        except Exception as e:
            self.tracker.mark_failed(job_id, str(e))
            logger.error("Job %s failed for instance %s: %s", job_id, instance_id, e)

    def get_job(self, job_id):
        return self.tracker.get(job_id)

    def shutdown(self):
        logger.info("Shutting down ThreadPoolBackend...")
        self._pool.shutdown(wait=True, cancel_futures=False)


# ═══════════════════════════════════════════════════════════════════
# Arq Backend (production — Redis)
# ═══════════════════════════════════════════════════════════════════

class ArqBackend(WorkerBackend):
    """
    Async execution via arq + Redis.

    Enqueues jobs to Redis. The arq worker (api/worker.py) picks them
    up and runs the coordinator in a thread pool.

    Requires: pip install arq redis
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.tracker = JobTracker()
        self._arq_pool = None

    def _ensure_pool(self):
        """Lazy-init arq Redis pool."""
        if self._arq_pool is not None:
            return
        try:
            import asyncio
            from arq import create_pool
            from arq.connections import RedisSettings

            settings = RedisSettings.from_dsn(self.redis_url)
            loop = asyncio.new_event_loop()
            self._arq_pool = loop.run_until_complete(create_pool(settings))
        except ImportError:
            raise RuntimeError(
                "arq not installed. pip install arq redis\n"
                "Or set CC_WORKER_MODE=inline for in-process execution."
            )
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Redis at {self.redis_url}: {e}")

    def enqueue(self, instance_id, workflow, domain, case_input,
                model="default", temperature=0.1, correlation_id=""):
        import asyncio

        self._ensure_pool()
        record = self.tracker.create(instance_id)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self._arq_pool.enqueue_job(
                "run_case",
                _job_id=record.job_id,
                instance_id=instance_id,
                workflow=workflow,
                domain=domain,
                case_input=case_input,
                model=model,
                temperature=temperature,
                correlation_id=correlation_id or instance_id,
            )
        )

        logger.info("Enqueued arq job %s for instance %s", record.job_id, instance_id)
        return record.job_id

    def get_job(self, job_id):
        return self.tracker.get(job_id)

    def shutdown(self):
        if self._arq_pool:
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._arq_pool.close())


# ═══════════════════════════════════════════════════════════════════
# Backend Factory
# ═══════════════════════════════════════════════════════════════════

def create_backend(
    mode: str | None = None,
    project_root: str = ".",
    max_workers: int = 4,
    redis_url: str = "redis://localhost:6379",
) -> WorkerBackend:
    """
    Create the appropriate worker backend.

    Mode selection:
      CC_WORKER_MODE env var or explicit mode parameter.
      - "inline": InlineBackend (synchronous)
      - "thread": ThreadPoolBackend (async, in-process)
      - "arq": ArqBackend (async, Redis)
      - None/auto: try arq, fall back to thread
    """
    mode = mode or os.environ.get("CC_WORKER_MODE", "auto")

    if mode == "inline":
        logger.info("Worker backend: InlineBackend (synchronous)")
        return InlineBackend(project_root=project_root)

    elif mode == "thread":
        logger.info("Worker backend: ThreadPoolBackend (max_workers=%d)", max_workers)
        return ThreadPoolBackend(project_root=project_root, max_workers=max_workers)

    elif mode == "arq":
        logger.info("Worker backend: ArqBackend (redis=%s)", redis_url)
        return ArqBackend(redis_url=redis_url)

    else:  # auto
        # Try arq first, fall back to thread
        try:
            import arq  # noqa: F401
            import redis as redis_lib  # noqa: F401
            logger.info("Worker backend: ArqBackend (auto-detected)")
            return ArqBackend(redis_url=redis_url)
        except ImportError:
            logger.info("Worker backend: ThreadPoolBackend (arq not available)")
            return ThreadPoolBackend(project_root=project_root, max_workers=max_workers)
