"""
Cognitive Core — HTML Trace Page (Sprint 1.3)

Serves a self-contained HTML trace page at GET /instances/{id}/trace

Three modes:
  Watch mode  — live SSE rendering as the workflow executes
  Input mode  — renders when workflow suspends at a HITL gate
  Result mode — complete audit trace after workflow completes

Single self-contained HTML file, no build step, no framework.
Vanilla JS, served directly from the FastAPI server.
"""

from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response

import cognitive_core.api.deps as deps

router = APIRouter()


_TRACE_HTML_PATH = Path(__file__).parent.parent / "trace.html"

def _load_trace_html() -> str:
    """Load trace HTML from file — single source of truth, never stale."""
    return _TRACE_HTML_PATH.read_text()



@router.get("/instances/{instance_id}/trace", response_class=Response)
async def get_trace_page(instance_id: str):
    """
    Serve the self-contained HTML trace page for a workflow instance.

    Three modes (handled client-side via SSE + REST):
      Watch mode  — live rendering as the workflow executes
      Input mode  — HITL review form when workflow is suspended
      Result mode — complete audit trace after workflow completes
    """
    coord = deps.get_coordinator()
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    return Response(content=_load_trace_html(), media_type="text/html")
