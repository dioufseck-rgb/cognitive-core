"""
Cognitive Core — Shadow Mode (H-004)

Dark-launch capability: run full workflow through primitives 1-7,
skip Act (primitive 8), log what it would have done.

Usage:
    # Per-instance
    shadow = ShadowMode(enabled=True, audit_trail=audit)
    if shadow.should_skip_act(instance_id, step_name):
        shadow.record_shadow_act(instance_id, step_name, proposed_action)
        return shadow_result

    # Per-domain config in domain YAML:
    #   shadow_mode: true

    # Or via environment:
    #   CC_SHADOW_MODE=true
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.shadow")


@dataclass
class ShadowActRecord:
    """Record of what Act would have done in shadow mode."""
    instance_id: str
    step_name: str
    proposed_action: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    primitive: str = "act"
    shadow: bool = True


class ShadowMode:
    """
    Shadow mode controller.

    When enabled, Act primitives are intercepted: the full proposed
    action is logged but not executed. All other primitives (1-7)
    run normally with real LLM calls and real data.
    """

    def __init__(
        self,
        enabled: bool = False,
        audit_trail: Any = None,
    ):
        # Check env override
        env_shadow = os.environ.get("CC_SHADOW_MODE", "").lower()
        self._enabled = enabled or env_shadow in ("true", "1", "yes")
        self._audit = audit_trail
        self._shadow_records: list[ShadowActRecord] = []
        if self._enabled:
            logger.info("Shadow mode ENABLED — Act primitives will be logged but not executed")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_skip_act(self, primitive: str) -> bool:
        """Check if this primitive should be skipped (only Act in shadow mode)."""
        return self._enabled and primitive == "act"

    def record_shadow_act(
        self,
        instance_id: str,
        step_name: str,
        proposed_action: dict[str, Any],
        trace_id: str = "",
    ) -> ShadowActRecord:
        """
        Record what Act would have done without executing it.

        The record goes to:
        1. In-memory shadow records (always)
        2. Audit trail as event_type='shadow_act' (if audit available)
        3. Structured log (always)
        """
        record = ShadowActRecord(
            instance_id=instance_id,
            step_name=step_name,
            proposed_action=proposed_action,
        )
        self._shadow_records.append(record)

        # Structured log
        logger.info(
            "SHADOW_ACT: instance=%s step=%s action=%s",
            instance_id,
            step_name,
            json.dumps(proposed_action, default=str)[:500],
            extra={
                "event_type": "shadow_act",
                "instance_id": instance_id,
                "step_name": step_name,
                "shadow": True,
            },
        )

        # Audit trail
        if self._audit and trace_id:
            try:
                self._audit.record(
                    trace_id=trace_id,
                    event_type="shadow_act",
                    payload={
                        "instance_id": instance_id,
                        "step_name": step_name,
                        "proposed_action": proposed_action,
                        "shadow_mode": True,
                        "message": "Act primitive skipped in shadow mode",
                    },
                )
            except Exception as e:
                logger.warning("Failed to record shadow act to audit: %s", e)

        return record

    def get_shadow_result(self, step_name: str) -> dict[str, Any]:
        """Return a synthetic Act result for shadow mode."""
        return {
            "action_taken": "SHADOW_MODE_NO_ACTION",
            "step": step_name,
            "shadow": True,
            "message": "Act was not executed — shadow mode active",
            "timestamp": time.time(),
        }

    @property
    def shadow_records(self) -> list[ShadowActRecord]:
        """All shadow act records for this session."""
        return list(self._shadow_records)

    @property
    def shadow_count(self) -> int:
        return len(self._shadow_records)

    def get_stats(self) -> dict[str, Any]:
        """Shadow mode statistics."""
        return {
            "shadow_mode_enabled": self._enabled,
            "shadow_acts_recorded": len(self._shadow_records),
            "instances": list(set(r.instance_id for r in self._shadow_records)),
        }
