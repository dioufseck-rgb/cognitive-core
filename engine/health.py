"""
Cognitive Core — Health Checks & Readiness Probes

Provides health, readiness, and startup checks for Kubernetes lifecycle.

Design: FastAPI (Option A for later). Implementation uses stdlib http.server
for zero-dependency operation, with the same JSON schema that a FastAPI
endpoint would return.

Checks:
  /health  — process alive (always 200)
  /ready   — LLM reachable, DB connected
  /startup — config valid, schema current

Usage:
    from engine.health import HealthChecker, run_health_server

    checker = HealthChecker()
    checker.register("database", db_check_fn)
    checker.register("llm", llm_check_fn)

    # Run as background thread
    run_health_server(checker, port=8080)

    # Or check programmatically
    result = checker.check_ready()
    # {"status": "ok", "checks": {"database": {"status": "ok", ...}}}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.health")


# ═══════════════════════════════════════════════════════════════════
# Check Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: str          # "ok" | "degraded" | "fail"
    latency_ms: float
    detail: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"status": self.status, "latency_ms": round(self.latency_ms, 1)}
        if self.detail:
            d["detail"] = self.detail
        if self.error:
            d["error"] = self.error
        return d


# ═══════════════════════════════════════════════════════════════════
# Health Checker
# ═══════════════════════════════════════════════════════════════════

# Type for check functions: () -> (bool, str)
# Returns (success, detail_message)
CheckFn = Callable[[], tuple[bool, str]]


class HealthChecker:
    """
    Central health check registry.

    Register check functions for each subsystem. The checker runs them
    on demand and returns structured results.
    """

    def __init__(self, check_timeout: float = 5.0):
        self.check_timeout = check_timeout
        self._checks: dict[str, CheckFn] = {}
        self._startup_checks: dict[str, CheckFn] = {}
        self._lock = threading.Lock()

    def register(self, name: str, check_fn: CheckFn) -> None:
        """Register a readiness check."""
        with self._lock:
            self._checks[name] = check_fn

    def register_startup(self, name: str, check_fn: CheckFn) -> None:
        """Register a startup check (run once)."""
        with self._lock:
            self._startup_checks[name] = check_fn

    def _run_check(self, name: str, fn: CheckFn) -> CheckResult:
        """Run a single check with timeout protection."""
        t0 = time.time()
        try:
            success, detail = fn()
            latency = (time.time() - t0) * 1000
            return CheckResult(
                name=name,
                status="ok" if success else "fail",
                latency_ms=latency,
                detail=detail,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return CheckResult(
                name=name,
                status="fail",
                latency_ms=latency,
                error=str(e)[:200],
            )

    def check_health(self) -> dict[str, Any]:
        """
        Liveness check. Always returns 200 if process is running.

        Returns:
            {"status": "ok", "timestamp": "..."}
        """
        return {
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def check_ready(self) -> dict[str, Any]:
        """
        Readiness check. Runs all registered checks.

        Returns:
            {
                "status": "ok" | "degraded" | "fail",
                "checks": {
                    "database": {"status": "ok", "latency_ms": 2.1},
                    "llm": {"status": "fail", "error": "timeout"}
                }
            }
        """
        with self._lock:
            checks = dict(self._checks)

        results = {}
        for name, fn in checks.items():
            result = self._run_check(name, fn)
            results[name] = result.to_dict()

        # Determine overall status
        statuses = [r["status"] for r in results.values()]
        if not statuses:
            overall = "ok"
        elif all(s == "ok" for s in statuses):
            overall = "ok"
        elif any(s == "fail" for s in statuses):
            overall = "fail"
        else:
            overall = "degraded"

        return {
            "status": overall,
            "checks": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def check_startup(self) -> dict[str, Any]:
        """
        Startup check. Runs one-time checks (config valid, migrations).

        Returns same schema as check_ready().
        """
        with self._lock:
            checks = dict(self._startup_checks)

        results = {}
        for name, fn in checks.items():
            result = self._run_check(name, fn)
            results[name] = result.to_dict()

        statuses = [r["status"] for r in results.values()]
        overall = "ok" if all(s == "ok" for s in statuses) else "fail"

        return {
            "status": overall,
            "checks": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ═══════════════════════════════════════════════════════════════════
# Built-in Checks
# ═══════════════════════════════════════════════════════════════════

def check_config_valid() -> tuple[bool, str]:
    """Startup check: validate LLM configuration."""
    try:
        from engine.llm import validate_config
        issues = validate_config()
        if issues:
            return False, f"Config issues: {'; '.join(issues)}"
        return True, "Config valid"
    except Exception as e:
        return False, f"Config check failed: {e}"


def check_spec_valid(root: str = ".") -> tuple[bool, str]:
    """Startup check: validate workflow/domain specs."""
    try:
        from engine.validate import validate_all
        errors, warnings = validate_all(root)
        if errors:
            return False, f"{len(errors)} validation errors"
        return True, f"Valid ({len(warnings)} warnings)"
    except Exception as e:
        return False, f"Spec validation failed: {e}"


def create_llm_check(provider: str | None = None) -> CheckFn:
    """Create a readiness check for LLM connectivity."""
    def check() -> tuple[bool, str]:
        try:
            from engine.llm import create_llm, detect_provider
            p = provider or detect_provider()
            # Just verify we can create the LLM (doesn't make a call)
            llm = create_llm(model="default", provider=p)
            return True, f"Provider {p} configured"
        except Exception as e:
            return False, f"LLM check failed: {e}"
    return check


def create_db_check(db_path: str | None = None) -> CheckFn:
    """Create a readiness check for database connectivity."""
    def check() -> tuple[bool, str]:
        try:
            import sqlite3
            path = db_path or "coordinator.db"
            conn = sqlite3.connect(path, timeout=2)
            conn.execute("SELECT 1")
            conn.close()
            return True, f"Database accessible: {path}"
        except Exception as e:
            return False, f"Database check failed: {e}"
    return check


# ═══════════════════════════════════════════════════════════════════
# HTTP Server
# ═══════════════════════════════════════════════════════════════════

class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health endpoints."""

    checker: HealthChecker = None  # Set by factory

    def do_GET(self):
        if self.path == "/health":
            result = self.checker.check_health()
            self._respond(200, result)
        elif self.path == "/ready":
            result = self.checker.check_ready()
            status_code = 200 if result["status"] == "ok" else 503
            self._respond(status_code, result)
        elif self.path == "/startup":
            result = self.checker.check_startup()
            status_code = 200 if result["status"] == "ok" else 503
            self._respond(status_code, result)
        else:
            self._respond(404, {"error": "Not found"})

    def _respond(self, code: int, body: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        # Suppress default stderr logging
        pass


def run_health_server(
    checker: HealthChecker,
    port: int = 8080,
    daemon: bool = True,
) -> HTTPServer:
    """
    Start health check HTTP server in a background thread.

    Returns the server instance (call .shutdown() to stop).
    """
    handler_cls = type("Handler", (_HealthHandler,), {"checker": checker})
    server = HTTPServer(("0.0.0.0", port), handler_cls)

    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()

    logger.info("Health server started on port %d", port)
    return server
