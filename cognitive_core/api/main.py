"""
Cognitive Core — Fraud Operations Center API

FastAPI server for the fraud examiner web UI.
Coordinator singleton, thread pool for sync workflows, static SPA mount.

Run:
    uvicorn api.main:app --reload --port 8080
"""

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import cognitive_core.api.deps as deps

COORD_CONFIG = "demos/fraud-operations/coordinator_config.yaml"
FRAUD_DATA_DB = "demos/fraud-operations/fraud_data.db"
FIXTURES_DIR = "demos/fraud-operations/cases/fixtures"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Seed fraud data DB from fixture files ──────────────────────
    try:
        from demos.fraud_operations.fraud_db import import_fixtures_dir
        rows = import_fixtures_dir(FRAUD_DATA_DB, FIXTURES_DIR)
        print(f"[startup] Seeded {rows} tool data rows from {FIXTURES_DIR}", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: Could not seed fraud data: {e}", flush=True)

    # ── Coordinator singleton ──────────────────────────────────────
    from cognitive_core.coordinator.runtime import Coordinator
    coordinator = Coordinator(config_path=COORD_CONFIG, verbose=False)
    deps.set_coordinator(coordinator)
    print(f"[startup] Coordinator ready (config: {COORD_CONFIG})", flush=True)

    # ── Terminate stale "running" instances from the previous process ─
    # Any instance left in "running" state by a previous server process
    # will never complete; terminate them so the kanban stays clean.
    try:
        stale = [i for i in coordinator.store.list_instances(limit=1000)
                 if i.status.value == "running"]
        if stale:
            for inst in stale:
                try:
                    coordinator.terminate(instance_id=inst.instance_id,
                                          reason="Stale: server restarted")
                except Exception:
                    pass
            print(f"[startup] Terminated {len(stale)} stale running instance(s)", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: Could not clean stale instances: {e}", flush=True)

    # ── Thread pool for sync workflow execution ────────────────────
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="workflow")
    deps.set_executor(executor)
    print("[startup] Thread pool ready (4 workers)", flush=True)

    yield

    executor.shutdown(wait=False)
    print("[shutdown] Thread pool stopped", flush=True)


app = FastAPI(
    title="NFCU Fraud Operations Center",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ────────────────────────────────────────────────────
from cognitive_core.api.routers import cases, instances, tasks, hitl, trace  # noqa: E402

app.include_router(cases.router, prefix="/api")
app.include_router(instances.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")
app.include_router(hitl.router, prefix="/api")
app.include_router(trace.router, prefix="/api")

# ── SPA (mount after API routes so /api/* routes win) ─────────────
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
