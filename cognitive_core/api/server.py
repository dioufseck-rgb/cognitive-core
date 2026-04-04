"""
Cognitive Core — General API Server

A domain-agnostic FastAPI server that hosts the governance loop endpoints
and serves the HTML trace page. Configured via environment variables or
command-line arguments so it works with any domain pack.

Usage:
    # From repo root, pointed at the consumer-lending pack:
    CC_COORD_CONFIG=library/domain-packs/consumer-lending/coordinator_config.yaml \\
    CC_COORD_BASE=library/domain-packs/consumer-lending \\
    uvicorn cognitive_core.api.server:app --port 8000

    # Or via the CLI helper:
    python -m cognitive_core.api.server --config library/domain-packs/consumer-lending/coordinator_config.yaml

Environment variables:
    CC_COORD_CONFIG   path to coordinator_config.yaml  (default: coordinator_config.yaml)
    CC_COORD_BASE     working directory for the coordinator (default: .)
    CC_DB_PATH        path to SQLite database (default: cognitive_core.db)
    CC_WORKERS        thread pool size (default: 4)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import cognitive_core.api.deps as deps

# ── Configuration from environment ────────────────────────────────────
COORD_CONFIG = os.environ.get("CC_COORD_CONFIG", "coordinator_config.yaml")
COORD_BASE   = os.environ.get("CC_COORD_BASE", ".")
DB_PATH      = os.environ.get("CC_DB_PATH", "cognitive_core.db")
WORKERS      = int(os.environ.get("CC_WORKERS", "4"))
API_KEY      = os.environ.get("CC_API_KEY", "")  # If set, all API requests require Authorization: Bearer <key>


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Resolve config path before any chdir ──────────────────────
    # COORD_CONFIG may be relative to the launch cwd (repo root).
    # Resolve it to absolute now, before chdir changes the cwd.
    config_path_abs = str(Path(COORD_CONFIG).resolve())

    # ── Change to coordinator base dir if specified ────────────────
    original_cwd = Path.cwd()
    base = Path(COORD_BASE).resolve()
    if base != original_cwd:
        os.chdir(base)
        print(f"[startup] Working directory: {base}", flush=True)

    # ── Coordinator singleton ──────────────────────────────────────
    from cognitive_core.coordinator.runtime import Coordinator
    coordinator = Coordinator(config_path=config_path_abs, verbose=False)
    deps.set_coordinator(coordinator)
    print(f"[startup] Coordinator ready (config: {COORD_CONFIG})", flush=True)

    # ── Hash chain on action ledger ────────────────────────────────
    # The CoordinatorStore already has hash chain built into log_action.
    # No patching needed — verify with store.verify_ledger(instance_id).
    print("[startup] Action ledger hash chain enabled", flush=True)

    # ── Clean stale state from previous process ───────────────────
    # Running instances will never complete (the thread is gone).
    # Pending tasks may reference instance IDs that no longer exist
    # or will never be actioned — cancel them all on restart.
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
            print(f"[startup] Terminated {len(stale)} stale instance(s)", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: Could not clean stale instances: {e}", flush=True)

    # Cancel all pending tasks — they belong to the previous process.
    # Leaving them causes old decisions to be applied to new instances.
    try:
        from cognitive_core.coordinator.tasks import TaskStatus, TaskResolution
        import time as _time
        pending_tasks = coordinator.tasks.list_tasks(status=TaskStatus.PENDING)
        cancelled = 0
        for task in pending_tasks:
            try:
                coordinator.tasks.resolve(task.task_id, TaskResolution(
                    task_id=task.task_id,
                    action="cancel",
                    resolved_by="server_startup",
                    notes="Stale task cancelled on server restart",
                    resolved_at=_time.time(),
                ))
                cancelled += 1
            except Exception:
                pass
        if cancelled:
            print(f"[startup] Cancelled {cancelled} stale pending task(s)", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: Could not cancel stale tasks: {e}", flush=True)

    # ── Thread pool ────────────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="workflow")
    deps.set_executor(executor)
    print(f"[startup] Thread pool ready ({WORKERS} workers)", flush=True)

    # ── Auth status ────────────────────────────────────────────────
    if API_KEY:
        print(f"[startup] API key auth enabled (CC_API_KEY)", flush=True)
    else:
        print(f"[startup] WARNING: No auth configured — API is open. "
              f"For production, deploy behind an API gateway or set CC_API_KEY.", flush=True)
    if not os.environ.get("X_CALLER_IDENTITY_HEADER"):
        print(f"[startup] Caller identity: read from X-Caller-Identity header (injected by gateway)", flush=True)

    yield

    executor.shutdown(wait=False)
    os.chdir(original_cwd)
    print("[shutdown] Server stopped", flush=True)


app = FastAPI(
    title="Cognitive Core",
    description="Governed institutional AI framework",
    version="0.1.0-technical-preview",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API key auth middleware ────────────────────────────────────────
# Enabled only when CC_API_KEY is set. Open by default for dev mode.
# Exempt: GET / (landing page) and GET /instances/*/trace (browser-facing).
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request                    # noqa: E402
from starlette.responses import JSONResponse              # noqa: E402

class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Minimal API key auth. Enabled only when CC_API_KEY is set.

    For production, deploy behind an API gateway (Apigee, Kong, AWS API Gateway)
    that handles authentication and injects caller identity via:
      X-Caller-Identity: alice@institution.com
      X-Caller-Role: underwriter

    The framework reads these headers and passes them to the action ledger
    so reviewer identity is recorded without the caller needing to supply it
    in the request body.
    """
    async def dispatch(self, request: Request, call_next):
        # Propagate gateway-injected identity into request state
        # so endpoints can read it without re-parsing headers
        caller = (
            request.headers.get("X-Caller-Identity") or
            request.headers.get("X-User-ID") or
            request.headers.get("X-Authenticated-User") or
            ""
        )
        request.state.caller_identity = caller

        if not API_KEY:
            return await call_next(request)

        path = request.url.path
        # Exempt browser-facing routes — auth handled at gateway
        if path == "/" or path.startswith("/instances/"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {API_KEY}":
            return await call_next(request)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

app.add_middleware(ApiKeyMiddleware)

# ── API Routers ────────────────────────────────────────────────────
from cognitive_core.api.routers import instances, tasks, hitl  # noqa: E402

app.include_router(instances.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")
app.include_router(hitl.router, prefix="/api")


# ── Instance submission endpoint ───────────────────────────────────
from fastapi import HTTPException  # noqa: E402
from pydantic import BaseModel     # noqa: E402
from typing import Any             # noqa: E402


class StartRequest(BaseModel):
    workflow_type: str
    domain: str
    case_input: dict[str, Any]


@app.post("/api/start")
async def start_instance(req: StartRequest) -> dict[str, Any]:
    """Submit a new case for processing. Returns instance_id immediately.
    The workflow runs in a background thread — open the trace_url to watch progress."""
    import asyncio
    import threading
    coord = deps.get_coordinator()

    # Validate required fields
    if not req.workflow_type:
        raise HTTPException(422, "workflow_type is required")
    if not req.domain:
        raise HTTPException(422, "domain is required")
    if not req.case_input:
        raise HTTPException(422, "case_input must not be empty")

    # Run start() in a background thread — returns immediately.
    # The workflow executes asynchronously; the SSE stream reports progress.
    result: dict[str, Any] = {}
    ready = threading.Event()

    def _run():
        try:
            instance_id = _run_start(coord, req.workflow_type, req.domain, req.case_input)
            result["instance_id"] = instance_id
        except Exception as e:
            result["error"] = str(e)
        finally:
            ready.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait briefly for the instance to be created (fast — just DB write)
    # then return. The workflow continues running in the background.
    ready.wait(timeout=30)

    if "error" in result:
        raise HTTPException(500, f"Failed to start workflow: {result['error']}")
    if "instance_id" not in result:
        raise HTTPException(504, "Workflow start timed out")

    instance_id = result["instance_id"]
    return {
        "instance_id": instance_id,
        "workflow_type": req.workflow_type,
        "domain": req.domain,
        "status": "running",
        "trace_url": f"/instances/{instance_id}/trace",
        "stream_url": f"/api/instances/{instance_id}/stream",
    }


def _run_workflow_background(coord, instance_id: str, workflow_type: str,
                              domain: str, case_input: dict) -> None:
    """Run workflow to completion in a background thread."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coord.start(
            workflow_type=workflow_type,
            domain=domain,
            case_input=case_input,
            _existing_instance_id=None,  # coord.start creates a new instance
        )
    except Exception as e:
        import logging
        logging.getLogger("permit_server").error(
            f"Background workflow {instance_id} failed: {e}"
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _run_start(coord, workflow_type: str, domain: str, case_input: dict) -> str:
    """
    Create the instance record and launch the workflow in a background thread.
    Returns the instance_id immediately — the workflow runs asynchronously.
    The trace page SSE stream shows progress as steps complete.
    """
    import asyncio
    import threading

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        instance_id = coord.start(
            workflow_type=workflow_type,
            domain=domain,
            case_input=case_input,
        )
        # Patch case_meta with domain-agnostic fields
        inst = coord.store.get_instance(instance_id)
        if inst and not any((inst.case_meta or {}).values()):
            inst.case_meta = {k: v for k, v in {
                "applicant_name": case_input.get("applicant_name", ""),
                "loan_amount":    str(case_input.get("loan_amount", "")),
                "loan_purpose":   case_input.get("loan_purpose", ""),
                "case_id":        case_input.get("case_id", ""),
                "permit_number":  case_input.get("get_application", {}).get("permit_number", ""),
                "proposed_use":   case_input.get("get_application", {}).get("proposed_use", ""),
            }.items() if v}
            coord.store.save_instance(inst)
        return instance_id
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _enriched_instance(inst, case_input: dict | None = None) -> dict:
    """Enrich an InstanceState with computed fields for API responses."""
    import time as _time

    # elapsed_seconds: computed from timestamps if not set by runtime
    elapsed = inst.elapsed_seconds or 0.0
    if not elapsed and inst.created_at and inst.updated_at:
        elapsed = round(inst.updated_at - inst.created_at, 2)

    # case_meta: extract domain-agnostic fields from case_input if meta is empty
    meta = inst.case_meta or {}
    if case_input and not any(meta.values()):
        meta = {
            "applicant_name": case_input.get("applicant_name", ""),
            "loan_amount":    case_input.get("loan_amount"),
            "loan_purpose":   case_input.get("loan_purpose", ""),
            "case_id":        case_input.get("case_id", ""),
        }
        # Remove None/empty for cleanliness
        meta = {k: v for k, v in meta.items() if v}

    return {
        "instance_id":     inst.instance_id,
        "workflow_type":   inst.workflow_type,
        "domain":          inst.domain,
        "status":          inst.status.value,
        "governance_tier": inst.governance_tier,
        "created_at":      inst.created_at,
        "updated_at":      inst.updated_at,
        "elapsed_seconds": elapsed,
        "step_count":      inst.step_count,
        "current_step":    inst.current_step,
        "case_meta":       meta,
        "error":           inst.error,
    }



@app.get("/api/instances")
async def list_instances(limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
    """List workflow instances."""
    coord = deps.get_coordinator()
    instances = coord.store.list_instances(limit=limit)
    result = []
    for inst in instances:
        if status and inst.status.value != status:
            continue
        result.append(_enriched_instance(inst))
    return result


@app.get("/api/instances/{instance_id}")
async def get_instance(instance_id: str) -> dict[str, Any]:
    """Get full instance state including result and ledger."""
    coord = deps.get_coordinator()
    inst = coord.store.get_instance(instance_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    ledger = coord.store.get_ledger(instance_id=instance_id)

    enriched = _enriched_instance(inst)
    enriched["correlation_id"] = inst.correlation_id
    enriched["result"] = inst.result
    enriched["ledger"] = ledger
    return enriched



# ── Ledger verification (Sprint 4.1) ──────────────────────────────
@app.get("/api/instances/{instance_id}/verify")
async def verify_instance_ledger(instance_id: str) -> dict[str, Any]:
    """
    Verify the hash chain integrity of an instance's action ledger.
    Returns {valid, first_invalid_entry, entries_checked}.
    """
    coord = deps.get_coordinator()
    inst = coord.store.get_instance(instance_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    try:
        result = coord.store.verify_ledger(instance_id)
        # Normalise field names to match spec: valid, first_invalid_entry, entries_checked
        return {
            "valid": result.get("valid", False),
            "first_invalid_entry": result.get("first_invalid_entry"),
            "entries_checked": result.get("entry_count", 0),
            "instance_id": instance_id,
        }
    except Exception as exc:
        raise HTTPException(500, f"Verification failed: {exc}")



@app.get("/", response_class=HTMLResponse)
async def landing():
    """Landing page — lists all instances with links to their trace pages."""
    coord = deps.get_coordinator()
    try:
        instances = coord.store.list_instances(limit=100)
    except Exception:
        instances = []

    STATUS_COLOR = {
        "completed":  "#3fb950",
        "suspended":  "#d29922",
        "running":    "#58a6ff",
        "failed":     "#f85149",
        "terminated": "#7d8590",
        "created":    "#7d8590",
    }
    STATUS_ICON = {
        "completed":  "✓",
        "suspended":  "⏸",
        "running":    "▶",
        "failed":     "✗",
        "terminated": "—",
        "created":    "·",
    }

    rows = []
    for inst in instances:
        status = inst.status.value
        color = STATUS_COLOR.get(status, "#7d8590")
        icon  = STATUS_ICON.get(status, "?")
        meta  = inst.case_meta or {}
        label = meta.get("case_id") or meta.get("member_name") or meta.get("applicant_name") or inst.instance_id[:16]
        elapsed = f"{inst.elapsed_seconds:.1f}s" if inst.elapsed_seconds else "—"
        tier = inst.governance_tier or "—"
        rows.append(f"""
        <tr onclick="window.location='/instances/{inst.instance_id}/trace'" style="cursor:pointer">
          <td style="font-family:monospace;font-size:12px;color:#7d8590">{inst.instance_id[:16]}</td>
          <td>{label}</td>
          <td style="font-family:monospace;font-size:12px">{inst.workflow_type}</td>
          <td style="font-family:monospace;font-size:12px;color:{color}">{icon} {status}</td>
          <td style="font-family:monospace;font-size:12px;color:#7d8590">{tier}</td>
          <td style="font-family:monospace;font-size:12px;color:#7d8590">{elapsed}</td>
          <td style="font-family:monospace;font-size:12px;color:#7d8590">{inst.step_count or 0} steps</td>
        </tr>""")

    rows_html = "".join(rows) if rows else """
        <tr><td colspan="7" style="text-align:center;color:#7d8590;padding:40px">
          No instances yet. Submit a case to get started.
        </td></tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cognitive Core</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;padding:32px}}
  h1{{font-size:18px;font-weight:700;margin-bottom:4px;display:flex;align-items:center;gap:10px}}
  .sub{{color:#7d8590;font-size:13px;margin-bottom:28px}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#7d8590;padding:8px 12px;border-bottom:1px solid #30363d}}
  td{{padding:10px 12px;border-bottom:1px solid #21262d}}
  tr:hover td{{background:#161b22}}
  .badge{{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;text-transform:uppercase}}
  a{{color:#58a6ff;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  .empty{{color:#7d8590;text-align:center;padding:60px;font-size:13px}}
</style>
<script>setTimeout(()=>location.reload(), 5000)</script>
</head>
<body>
<h1>
  <svg width="20" height="20" viewBox="0 0 22 22" fill="none">
    <rect width="22" height="22" rx="5" fill="#58a6ff" opacity=".15"/>
    <path d="M4 11h4l2-6 3 12 2-6h3" stroke="#58a6ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  Cognitive Core
</h1>
<p class="sub">Governed institutional AI — <a href="/docs">API docs</a></p>
<table>
  <thead>
    <tr>
      <th>Instance</th><th>Case</th><th>Workflow</th>
      <th>Status</th><th>Tier</th><th>Elapsed</th><th>Steps</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── HTML Trace page ────────────────────────────────────────────────
@app.get("/instances/{instance_id}/trace", response_class=HTMLResponse)
async def trace_page(instance_id: str):
    """Serve the self-contained HTML trace page for a workflow instance."""
    coord = deps.get_coordinator()
    inst = coord.store.get_instance(instance_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    # Read the trace page template
    trace_html_path = Path(__file__).parent / "trace.html"
    if not trace_html_path.exists():
        raise HTTPException(500, "Trace page template not found. Run the server from the repo root.")

    html = trace_html_path.read_text()
    # Inject the instance_id so the page bootstraps immediately
    html = html.replace("__INSTANCE_ID__", instance_id)
    return HTMLResponse(content=html)


# ── CLI entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Cognitive Core API Server")
    parser.add_argument("--config", default=COORD_CONFIG, help="Path to coordinator_config.yaml")
    parser.add_argument("--base", default=COORD_BASE, help="Coordinator working directory")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    os.environ["CC_COORD_CONFIG"] = args.config
    os.environ["CC_COORD_BASE"] = args.base

    uvicorn.run(
        "cognitive_core.api.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )