"""
Cognitive Core — API Server (entry point)

This module re-exports the app from server.py, which is the canonical
framework-level server implementation.

Use server.py directly for full configuration options:
    uvicorn cognitive_core.api.server:app --port 8000

Or use this module for the default configuration:
    uvicorn cognitive_core.api.main:app --port 8000
"""

from cognitive_core.api.server import app  # noqa: F401

__all__ = ["app"]
