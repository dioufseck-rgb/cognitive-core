"""
Cognitive Core — arq Worker Entry Point

This module is the CMD target for the worker container.
It defines the arq worker class that processes cases from Redis.

Usage:
    python -m api.arq_worker

    # Or via arq CLI:
    arq api.arq_worker.WorkerSettings
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("cognitive_core.arq_worker")


async def run_case(
    ctx: dict,
    *,
    instance_id: str,
    workflow: str,
    domain: str,
    case_input: dict,
    model: str = "default",
    temperature: float = 0.1,
    correlation_id: str = "",
):
    """
    arq task function. Runs the coordinator in a thread pool
    to avoid blocking the async event loop.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    pool: ThreadPoolExecutor = ctx.get("pool")

    def _execute():
        from coordinator.runtime import Coordinator
        project_root = os.environ.get("CC_PROJECT_ROOT", ".")
        coord = Coordinator(project_root=project_root, verbose=False)
        coord.start(
            workflow_type=workflow,
            domain=domain,
            case_input=case_input,
            correlation_id=correlation_id or instance_id,
            model=model,
            temperature=temperature,
        )
        return instance_id

    result = await loop.run_in_executor(pool, _execute)
    logger.info("Completed case %s (%s/%s)", instance_id, workflow, domain)
    return result


async def startup(ctx: dict):
    """arq startup hook — initialize thread pool."""
    max_workers = int(os.environ.get("CC_MAX_WORKERS", "4"))
    ctx["pool"] = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cc_worker",
    )
    logger.info("arq worker started: max_workers=%d", max_workers)


async def shutdown(ctx: dict):
    """arq shutdown hook — clean up thread pool."""
    pool = ctx.get("pool")
    if pool:
        pool.shutdown(wait=True)
    logger.info("arq worker shutdown complete")


class WorkerSettings:
    """arq worker configuration."""
    functions = [run_case]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = int(os.environ.get("CC_MAX_WORKERS", "4"))
    job_timeout = int(os.environ.get("CC_JOB_TIMEOUT", "300"))  # 5 min default
    redis_settings = None  # Set from REDIS_URL at import time

    @classmethod
    def _init_redis(cls):
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            from arq.connections import RedisSettings
            cls.redis_settings = RedisSettings.from_dsn(redis_url)
        except ImportError:
            logger.warning("arq not installed — worker cannot start")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._init_redis()


# Initialize on import
try:
    WorkerSettings._init_redis()
except Exception:
    pass


if __name__ == "__main__":
    try:
        from arq import run_worker
        run_worker(WorkerSettings)
    except ImportError:
        print("arq not installed. Install with: pip install arq redis")
        print("Or use CC_WORKER_MODE=thread for in-process execution.")
