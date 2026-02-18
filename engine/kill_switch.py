"""
Cognitive Core — Kill Switches (S-007)

Runtime-toggleable controls for production safety:
  - disable_act: prevents all write-side operations
  - disable_delegation: prevents cross-workflow dispatch
  - disabled_domains: set of domain names to reject
  - disabled_workflows: set of workflow names to reject
  - disabled_policies: set of delegation policy names to skip

Kill switches are stored in a thread-safe in-memory store with
optional persistence to SQLite (coordinator store).

Usage:
    from engine.kill_switch import KillSwitchManager, get_kill_switches

    ks = get_kill_switches()
    ks.disable_act("Runaway Act calls in prod — incident #1234")
    ks.disable_domain("card_dispute", "Domain under SME review")

    # Check before execution
    if ks.is_act_disabled():
        raise KillSwitchTripped("Act is disabled: " + ks.act_reason())

    # List all active switches
    ks.status()  # → dict of all switches and reasons
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.kill_switch")


@dataclass
class SwitchState:
    """Single kill switch state."""
    enabled: bool = False
    reason: str = ""
    toggled_by: str = ""
    toggled_at: float = 0.0

    def activate(self, reason: str = "", by: str = "system"):
        self.enabled = True
        self.reason = reason
        self.toggled_by = by
        self.toggled_at = time.time()

    def deactivate(self, by: str = "system"):
        self.enabled = False
        self.reason = ""
        self.toggled_by = by
        self.toggled_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "toggled_by": self.toggled_by,
            "toggled_at": self.toggled_at,
        }


class KillSwitchTripped(Exception):
    """Raised when a kill switch prevents execution."""
    def __init__(self, switch_name: str, reason: str = ""):
        self.switch_name = switch_name
        self.reason = reason
        super().__init__(f"Kill switch tripped: {switch_name}" +
                        (f" — {reason}" if reason else ""))


class KillSwitchManager:
    """
    Thread-safe kill switch manager.

    Global switches:
      - act_disabled: blocks all Act primitive execution
      - delegation_disabled: blocks all cross-workflow delegation

    Granular switches:
      - disabled_domains: set of domain names
      - disabled_workflows: set of workflow names
      - disabled_policies: set of delegation policy names
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._act = SwitchState()
        self._delegation = SwitchState()
        self._domains: dict[str, SwitchState] = {}
        self._workflows: dict[str, SwitchState] = {}
        self._policies: dict[str, SwitchState] = {}

    # ── Global: Act ──────────────────────────────────────────

    def disable_act(self, reason: str = "", by: str = "system"):
        with self._lock:
            self._act.activate(reason, by)
        logger.warning("KILL SWITCH: Act DISABLED — %s (by %s)", reason, by)

    def enable_act(self, by: str = "system"):
        with self._lock:
            self._act.deactivate(by)
        logger.info("Kill switch: Act re-enabled (by %s)", by)

    def is_act_disabled(self) -> bool:
        with self._lock:
            return self._act.enabled

    def check_act(self):
        """Raise if Act is disabled."""
        with self._lock:
            if self._act.enabled:
                raise KillSwitchTripped("act_disabled", self._act.reason)

    # ── Global: Delegation ───────────────────────────────────

    def disable_delegation(self, reason: str = "", by: str = "system"):
        with self._lock:
            self._delegation.activate(reason, by)
        logger.warning("KILL SWITCH: Delegation DISABLED — %s (by %s)", reason, by)

    def enable_delegation(self, by: str = "system"):
        with self._lock:
            self._delegation.deactivate(by)
        logger.info("Kill switch: Delegation re-enabled (by %s)", by)

    def is_delegation_disabled(self) -> bool:
        with self._lock:
            return self._delegation.enabled

    def check_delegation(self):
        with self._lock:
            if self._delegation.enabled:
                raise KillSwitchTripped("delegation_disabled", self._delegation.reason)

    # ── Granular: Domains ────────────────────────────────────

    def disable_domain(self, domain: str, reason: str = "", by: str = "system"):
        with self._lock:
            state = self._domains.setdefault(domain, SwitchState())
            state.activate(reason, by)
        logger.warning("KILL SWITCH: Domain '%s' DISABLED — %s (by %s)", domain, reason, by)

    def enable_domain(self, domain: str, by: str = "system"):
        with self._lock:
            if domain in self._domains:
                self._domains[domain].deactivate(by)
        logger.info("Kill switch: Domain '%s' re-enabled (by %s)", domain, by)

    def is_domain_disabled(self, domain: str) -> bool:
        with self._lock:
            return domain in self._domains and self._domains[domain].enabled

    def check_domain(self, domain: str):
        with self._lock:
            if domain in self._domains and self._domains[domain].enabled:
                raise KillSwitchTripped(f"domain_disabled:{domain}",
                                       self._domains[domain].reason)

    # ── Granular: Workflows ──────────────────────────────────

    def disable_workflow(self, workflow: str, reason: str = "", by: str = "system"):
        with self._lock:
            state = self._workflows.setdefault(workflow, SwitchState())
            state.activate(reason, by)
        logger.warning("KILL SWITCH: Workflow '%s' DISABLED — %s (by %s)", workflow, reason, by)

    def enable_workflow(self, workflow: str, by: str = "system"):
        with self._lock:
            if workflow in self._workflows:
                self._workflows[workflow].deactivate(by)
        logger.info("Kill switch: Workflow '%s' re-enabled (by %s)", workflow, by)

    def is_workflow_disabled(self, workflow: str) -> bool:
        with self._lock:
            return workflow in self._workflows and self._workflows[workflow].enabled

    def check_workflow(self, workflow: str):
        with self._lock:
            if workflow in self._workflows and self._workflows[workflow].enabled:
                raise KillSwitchTripped(f"workflow_disabled:{workflow}",
                                       self._workflows[workflow].reason)

    # ── Granular: Policies ───────────────────────────────────

    def disable_policy(self, policy: str, reason: str = "", by: str = "system"):
        with self._lock:
            state = self._policies.setdefault(policy, SwitchState())
            state.activate(reason, by)
        logger.warning("KILL SWITCH: Policy '%s' DISABLED — %s (by %s)", policy, reason, by)

    def enable_policy(self, policy: str, by: str = "system"):
        with self._lock:
            if policy in self._policies:
                self._policies[policy].deactivate(by)

    def is_policy_disabled(self, policy: str) -> bool:
        with self._lock:
            return policy in self._policies and self._policies[policy].enabled

    def check_policy(self, policy: str):
        with self._lock:
            if policy in self._policies and self._policies[policy].enabled:
                raise KillSwitchTripped(f"policy_disabled:{policy}",
                                       self._policies[policy].reason)

    # ── Pre-flight check (all relevant switches at once) ─────

    def preflight_check(
        self,
        workflow: str = "",
        domain: str = "",
        has_act: bool = False,
        has_delegation: bool = False,
        policies: list[str] | None = None,
    ):
        """
        Run all relevant kill switch checks for a workflow execution.
        Raises KillSwitchTripped on first match.
        """
        if has_act:
            self.check_act()
        if has_delegation:
            self.check_delegation()
        if workflow:
            self.check_workflow(workflow)
        if domain:
            self.check_domain(domain)
        for p in (policies or []):
            self.check_policy(p)

    # ── Status ───────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Full status of all kill switches."""
        with self._lock:
            result = {
                "act_disabled": self._act.to_dict(),
                "delegation_disabled": self._delegation.to_dict(),
                "disabled_domains": {
                    k: v.to_dict() for k, v in self._domains.items() if v.enabled
                },
                "disabled_workflows": {
                    k: v.to_dict() for k, v in self._workflows.items() if v.enabled
                },
                "disabled_policies": {
                    k: v.to_dict() for k, v in self._policies.items() if v.enabled
                },
            }
            result["any_active"] = (
                self._act.enabled or
                self._delegation.enabled or
                any(v.enabled for v in self._domains.values()) or
                any(v.enabled for v in self._workflows.values()) or
                any(v.enabled for v in self._policies.values())
            )
            return result

    def reset_all(self, by: str = "system"):
        """Clear all kill switches."""
        with self._lock:
            self._act.deactivate(by)
            self._delegation.deactivate(by)
            for s in self._domains.values():
                s.deactivate(by)
            for s in self._workflows.values():
                s.deactivate(by)
            for s in self._policies.values():
                s.deactivate(by)
        logger.info("All kill switches cleared (by %s)", by)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_instance: KillSwitchManager | None = None
_instance_lock = threading.Lock()


def get_kill_switches() -> KillSwitchManager:
    """Get the global KillSwitchManager instance."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = KillSwitchManager()
    return _instance


def reset_kill_switches():
    """Reset global instance (for testing)."""
    global _instance
    with _instance_lock:
        _instance = None
