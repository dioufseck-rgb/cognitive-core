"""
Cognitive Core — Structured Logging with Correlation IDs

Implements TraceCallback protocol from engine.nodes to emit structured
JSON log lines for every workflow event. Compatible with OpenTelemetry
semantic conventions for future migration to OTel SDK.

Design decisions:
  - Transport: Python logging with JSON formatter (P-002 decision: Option A)
  - Schema: OTel-compatible (trace_id, span_id, service.name)
  - Configurable log level: DEBUG (full prompts), INFO (actions), WARN (errors only)

Usage:
    from engine.logging import StructuredLogger, configure_logging

    configure_logging(level="INFO")
    logger = StructuredLogger(workflow="product_return", domain="electronics_return")
    set_trace(logger)  # Wire into node execution

    # Logger auto-generates trace_id, propagates through delegation
    child_logger = logger.child(workflow="sar_investigation")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# JSON Formatter (OTel-compatible)
# ═══════════════════════════════════════════════════════════════════

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON lines.

    OTel semantic conventions used:
      - trace_id: maps to OTel trace ID
      - span_id: maps to OTel span ID
      - service.name: "cognitive_core"
      - service.version: from env
    """

    def __init__(self, service_name: str = "cognitive_core"):
        super().__init__()
        self.service_name = service_name
        self.service_version = os.environ.get("CC_VERSION", "0.1.0")

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            # OTel resource attributes
            "service.name": self.service_name,
            "service.version": self.service_version,
        }

        # Merge structured fields from extra
        if hasattr(record, "structured"):
            entry.update(record.structured)

        # Exception info
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception.type"] = record.exc_info[0].__name__
            entry["exception.message"] = str(record.exc_info[1])

        return json.dumps(entry, default=str)


# ═══════════════════════════════════════════════════════════════════
# Log Configuration
# ═══════════════════════════════════════════════════════════════════

_configured = False


def configure_logging(
    level: str = "INFO",
    stream: Any = None,
    service_name: str = "cognitive_core",
) -> logging.Logger:
    """
    Configure the cognitive_core logger with JSON output.

    Args:
        level: DEBUG, INFO, WARNING, ERROR
        stream: Output stream (default: sys.stderr)
        service_name: Service name in log entries

    Returns:
        The configured root logger for cognitive_core
    """
    global _configured

    logger = logging.getLogger("cognitive_core")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on reconfigure
    logger.handlers.clear()

    # Also clear any child loggers that may have been created
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("cognitive_core."):
            child = logging.getLogger(name)
            child.handlers.clear()
            child.setLevel(logging.NOTSET)  # Inherit from parent

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter(service_name=service_name))
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    _configured = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Get a child logger under cognitive_core namespace."""
    if name:
        return logging.getLogger(f"cognitive_core.{name}")
    return logging.getLogger("cognitive_core")


# ═══════════════════════════════════════════════════════════════════
# Trace ID Generation
# ═══════════════════════════════════════════════════════════════════

def generate_trace_id() -> str:
    """Generate an OTel-compatible trace ID (32 hex chars)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate an OTel-compatible span ID (16 hex chars)."""
    return uuid.uuid4().hex[:16]


# ═══════════════════════════════════════════════════════════════════
# Structured Logger (implements TraceCallback)
# ═══════════════════════════════════════════════════════════════════

class StructuredLogger:
    """
    Structured logger that implements the TraceCallback protocol.

    Wire into node execution via set_trace(logger).
    Every log entry includes trace_id for end-to-end correlation.
    """

    def __init__(
        self,
        workflow: str = "",
        domain: str = "",
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
    ):
        self.workflow = workflow
        self.domain = domain
        self.trace_id = trace_id or generate_trace_id()
        self.parent_trace_id = parent_trace_id
        self._logger = get_logger("trace")
        self._step_spans: dict[str, str] = {}  # step_name → span_id

    def child(
        self,
        workflow: str = "",
        domain: str = "",
    ) -> StructuredLogger:
        """Create a child logger for delegated workflows."""
        return StructuredLogger(
            workflow=workflow or self.workflow,
            domain=domain or self.domain,
            trace_id=generate_trace_id(),
            parent_trace_id=self.trace_id,
        )

    def _base_fields(self) -> dict[str, Any]:
        """Common fields for every log entry."""
        fields = {
            "trace_id": self.trace_id,
            "workflow": self.workflow,
            "domain": self.domain,
        }
        if self.parent_trace_id:
            fields["parent_trace_id"] = self.parent_trace_id
        return fields

    def _emit(self, level: int, action: str, **fields):
        """Emit a structured log entry."""
        if not self._logger.isEnabledFor(level):
            return
        structured = {**self._base_fields(), "action": action, **fields}
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="", lno=0, msg=action,
            args=(), exc_info=None,
        )
        record.structured = structured
        self._logger.handle(record)

    # ── TraceCallback Protocol ──────────────────────────────────

    def on_step_start(self, step_name: str, primitive: str, loop_iteration: int) -> None:
        span_id = generate_span_id()
        self._step_spans[step_name] = span_id
        self._emit(
            logging.INFO, "step_start",
            step_name=step_name,
            primitive=primitive,
            loop_iteration=loop_iteration,
            span_id=span_id,
        )

    def on_llm_start(self, step_name: str, prompt_chars: int) -> None:
        fields = {
            "step_name": step_name,
            "prompt_chars": prompt_chars,
        }
        if step_name in self._step_spans:
            fields["span_id"] = self._step_spans[step_name]
        self._emit(logging.DEBUG, "llm_start", **fields)

    def on_llm_end(self, step_name: str, response_chars: int, elapsed: float) -> None:
        fields = {
            "step_name": step_name,
            "response_chars": response_chars,
            "latency_ms": round(elapsed * 1000, 1),
        }
        if step_name in self._step_spans:
            fields["span_id"] = self._step_spans[step_name]
        self._emit(logging.INFO, "llm_end", **fields)

    def on_parse_result(self, step_name: str, primitive: str, output: dict) -> None:
        # At INFO level, log summary. At DEBUG, log full output.
        fields = {
            "step_name": step_name,
            "primitive": primitive,
        }
        if step_name in self._step_spans:
            fields["span_id"] = self._step_spans[step_name]

        # Extract key metrics from output
        if "confidence" in output:
            fields["confidence"] = output["confidence"]
        if "category" in output:
            fields["category"] = output["category"]
        if "_parse_failed" in output:
            fields["parse_failed"] = output["_parse_failed"]

        self._emit(logging.INFO, "parse_result", **fields)

        # Full output at DEBUG
        if self._logger.isEnabledFor(logging.DEBUG):
            self._emit(logging.DEBUG, "parse_result_full",
                       step_name=step_name, output=output)

    def on_parse_error(self, step_name: str, error: str) -> None:
        self._emit(
            logging.WARNING, "parse_error",
            step_name=step_name,
            error=error[:500],
        )

    def on_route_decision(self, from_step: str, to_step: str,
                          decision_type: str, reason: str) -> None:
        self._emit(
            logging.INFO, "route_decision",
            from_step=from_step,
            to_step=to_step,
            decision_type=decision_type,
            reason=reason[:500],
        )

    def on_retrieve_start(self, step_name: str, source_name: str) -> None:
        self._emit(
            logging.DEBUG, "retrieve_start",
            step_name=step_name,
            source_name=source_name,
        )

    def on_retrieve_end(self, step_name: str, source_name: str,
                        status: str, latency_ms: float) -> None:
        self._emit(
            logging.INFO, "retrieve_end",
            step_name=step_name,
            source_name=source_name,
            status=status,
            latency_ms=round(latency_ms, 1),
        )

    # ── Extended Events (beyond TraceCallback) ──────────────────

    def on_governance_decision(
        self,
        domain: str,
        declared_tier: str,
        applied_tier: str,
        quality_gate_result: str | None = None,
        reason: str = "",
    ) -> None:
        """Log governance tier evaluation."""
        self._emit(
            logging.INFO, "governance_decision",
            domain=domain,
            declared_tier=declared_tier,
            applied_tier=applied_tier,
            quality_gate_result=quality_gate_result,
            reason=reason,
        )

    def on_delegation_start(
        self,
        parent_workflow: str,
        child_workflow: str,
        policy: str,
        mode: str,
        child_trace_id: str,
    ) -> None:
        """Log delegation dispatch."""
        self._emit(
            logging.INFO, "delegation_start",
            parent_workflow=parent_workflow,
            child_workflow=child_workflow,
            policy=policy,
            mode=mode,
            child_trace_id=child_trace_id,
        )

    def on_delegation_complete(
        self,
        child_trace_id: str,
        status: str,
        elapsed_s: float,
    ) -> None:
        """Log delegation completion."""
        self._emit(
            logging.INFO, "delegation_complete",
            child_trace_id=child_trace_id,
            status=status,
            elapsed_s=round(elapsed_s, 2),
        )

    def on_workflow_start(self) -> None:
        """Log workflow execution start."""
        self._emit(logging.INFO, "workflow_start")

    def on_workflow_end(self, status: str, elapsed_s: float, steps_completed: int = 0) -> None:
        """Log workflow execution end."""
        self._emit(
            logging.INFO, "workflow_end",
            status=status,
            elapsed_s=round(elapsed_s, 2),
            steps_completed=steps_completed,
        )
