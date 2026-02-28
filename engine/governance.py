"""
Cognitive Core — Governance Pipeline

Wires all passive governance modules into the execution path.
This is the single integration point. Every governance module has
a clean interface; this module initializes them and provides the
chokepoints that nodes.py and runtime.py call.

Architecture:
    GovernancePipeline (singleton)
      ├── InputGuardrail       — prompt injection detection
      ├── PiiRedactor           — PII masking on LLM I/O
      ├── SemanticCache         — LLM response dedup
      ├── ProviderRateLimiter   — per-provider concurrency control
      ├── CostTracker           — token + cost accounting
      ├── LogicCircuitBreaker   — quality monitoring → tier escalation
      ├── KillSwitchManager     — runtime emergency stop
      ├── ShadowMode            — dark launch (skip Act)
      ├── CompensationLedger    — Act reversal on failure
      ├── WebhookNotifier       — notify on governance suspension
      ├── ReplayManager         — checkpoint save per step
      ├── SpecManifest          — config hash at start
      ├── RoutingManager        — HITL capability-based routing
      ├── HITLStateMachine      — review lifecycle
      └── resolve_effective_tier — upward-only tier enforcement

Usage:
    from engine.governance import get_governance, GovernancePipeline

    gov = get_governance()
    gov.protected_llm_call(llm, prompt, step_name, domain, model, case_input)
    gov.check_start_gates(workflow_type, domain, case_input)
    gov.record_step_result(domain, primitive, confidence)
    gov.on_governance_suspension(instance_id, workflow, domain, tier, reason)
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.governance")

# ═══════════════════════════════════════════════════════════════════
# Governance Pipeline
# ═══════════════════════════════════════════════════════════════════

_instance: GovernancePipeline | None = None
_lock = threading.Lock()


@dataclass
class LLMCallResult:
    """Result from a protected LLM call."""
    raw_response: str
    cached: bool = False
    redacted: bool = False
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class GovernancePipeline:
    """
    Central governance pipeline. Initializes all modules lazily and
    provides the chokepoint methods called from nodes.py and runtime.py.

    All modules are optional — if a module fails to initialize (e.g.,
    missing config), it's disabled with a warning. The pipeline never
    blocks execution due to a governance module failure.
    """

    def __init__(self):
        self._guardrail = None
        self._pii_redactor = None
        self._cache = None
        self._rate_limiters: dict[str, Any] = {}
        self._cost_tracker = None
        self._circuit_breaker = None
        self._kill_switches = None
        self._shadow_mode = None
        self._compensation = None
        self._webhook_notifier = None
        self._replay_manager = None
        self._hitl_router = None
        self._hitl_state_machine = None
        self._initialized = False
        # ── Proof ledger: records every governance decision ──
        self._proof_ledger: list[dict[str, Any]] = []
        self._proof_lock = threading.Lock()

    def initialize(self):
        """Lazy-initialize all governance modules."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("Initializing governance pipeline...")

        # ── Input Guardrails ──
        try:
            from engine.guardrails import InputGuardrail
            self._guardrail = InputGuardrail()
            logger.info("  ✓ InputGuardrail")
        except Exception as e:
            logger.warning("  ✗ InputGuardrail: %s", e)

        # ── PII Redaction ──
        try:
            from engine.pii import PiiRedactor
            enabled = os.environ.get("CC_PII_ENABLED", "true").lower() != "false"
            self._pii_redactor = PiiRedactor(enabled=enabled)
            logger.info("  ✓ PiiRedactor (enabled=%s)", enabled)
        except Exception as e:
            logger.warning("  ✗ PiiRedactor: %s", e)

        # ── Semantic Cache ──
        try:
            from engine.semantic_cache import SemanticCache
            enabled = os.environ.get("CC_CACHE_ENABLED", "false").lower() == "true"
            ttl = int(os.environ.get("CC_CACHE_TTL", "3600"))
            self._cache = SemanticCache(enabled=enabled, ttl_seconds=ttl)
            logger.info("  ✓ SemanticCache (enabled=%s, ttl=%ds)", enabled, ttl)
        except Exception as e:
            logger.warning("  ✗ SemanticCache: %s", e)

        # ── Cost Tracker ──
        try:
            from engine.cost import CostTracker
            self._cost_tracker = CostTracker()
            logger.info("  ✓ CostTracker")
        except Exception as e:
            logger.warning("  ✗ CostTracker: %s", e)

        # ── Circuit Breaker ──
        try:
            from engine.logic_breaker import get_logic_breaker
            self._circuit_breaker = get_logic_breaker()
            logger.info("  ✓ LogicCircuitBreaker")
        except Exception as e:
            logger.warning("  ✗ LogicCircuitBreaker: %s", e)

        # ── Kill Switches ──
        try:
            from engine.kill_switch import KillSwitchManager
            self._kill_switches = KillSwitchManager()
            logger.info("  ✓ KillSwitchManager")
        except Exception as e:
            logger.warning("  ✗ KillSwitchManager: %s", e)

        # ── Shadow Mode ──
        try:
            from engine.shadow import ShadowMode
            enabled = os.environ.get("CC_SHADOW_MODE", "false").lower() == "true"
            self._shadow_mode = ShadowMode(enabled=enabled)
            logger.info("  ✓ ShadowMode (enabled=%s)", enabled)
        except Exception as e:
            logger.warning("  ✗ ShadowMode: %s", e)

        # ── Compensation Ledger ──
        try:
            from engine.compensation import CompensationLedger
            self._compensation = CompensationLedger()
            logger.info("  ✓ CompensationLedger")
        except Exception as e:
            logger.warning("  ✗ CompensationLedger: %s", e)

        # ── Webhook Notifier ──
        try:
            from engine.webhooks import WebhookNotifier, WebhookConfig
            url = os.environ.get("CC_WEBHOOK_URL", "")
            fmt = os.environ.get("CC_WEBHOOK_FORMAT", "generic")
            if url:
                config = WebhookConfig(url=url, format=fmt)
                self._webhook_notifier = WebhookNotifier(configs=[config])
                logger.info("  ✓ WebhookNotifier (%s)", fmt)
            else:
                logger.info("  – WebhookNotifier (no URL configured)")
        except Exception as e:
            logger.warning("  ✗ WebhookNotifier: %s", e)

        # ── Replay Manager ──
        try:
            from engine.replay import ReplayManager
            enabled = os.environ.get("CC_REPLAY_ENABLED", "true").lower() != "false"
            if enabled:
                self._replay_manager = ReplayManager()
                logger.info("  ✓ ReplayManager")
            else:
                logger.info("  – ReplayManager (disabled)")
        except Exception as e:
            logger.warning("  ✗ ReplayManager: %s", e)

        # ── HITL Routing ──
        try:
            from engine.hitl_routing import RoutingManager
            self._hitl_router = RoutingManager()
            logger.info("  ✓ RoutingManager")
        except Exception as e:
            logger.warning("  ✗ RoutingManager: %s", e)

        # ── HITL State Machine ──
        try:
            from engine.hitl_state import HITLStateMachine
            self._hitl_state_machine = HITLStateMachine()
            logger.info("  ✓ HITLStateMachine")
        except Exception as e:
            logger.warning("  ✗ HITLStateMachine: %s", e)

        logger.info("Governance pipeline initialized.")

    # ─── Rate Limiter (per-provider, lazy) ───────────────────────

    def _get_rate_limiter(self, provider: str):
        """Get or create rate limiter for a provider."""
        if provider in self._rate_limiters:
            return self._rate_limiters[provider]
        try:
            from engine.rate_limit import ProviderRateLimiter, RateLimitConfig
            # Load config from llm_config.yaml or env
            config = None
            config_path = os.environ.get("LLM_CONFIG_PATH", "llm_config.yaml")
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                rl_cfg = cfg.get("rate_limits", {}).get(provider, {})
                if rl_cfg:
                    config = RateLimitConfig(
                        max_concurrent=rl_cfg.get("max_concurrent", 10),
                        requests_per_minute=rl_cfg.get("requests_per_minute", 60),
                    )
            except (FileNotFoundError, Exception):
                pass
            limiter = ProviderRateLimiter(config=config)
            self._rate_limiters[provider] = limiter
            return limiter
        except Exception as e:
            logger.warning("Rate limiter init failed for %s: %s", provider, e)
            self._rate_limiters[provider] = None
            return None

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 1: Protected LLM Call
    # Called from engine/nodes.py at every llm.invoke() site
    # ═══════════════════════════════════════════════════════════════

    def protected_llm_call(
        self,
        llm: Any,
        prompt: str,
        step_name: str,
        domain: str,
        model: str,
        case_input: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        """
        Execute an LLM call with full governance pipeline:
        1. PII redaction
        2. Semantic cache check
        3. Rate limiting
        4. LLM invoke
        5. PII de-redaction
        6. Cache store
        7. Cost tracking

        Returns LLMCallResult with the raw_response and metadata.
        """
        self.initialize()
        result = LLMCallResult(raw_response="")
        working_prompt = prompt

        # ── 1. PII Redaction ──
        if self._pii_redactor:
            try:
                if case_input and not self._pii_redactor.redaction_count:
                    self._pii_redactor.register_entities_from_case(case_input)
                working_prompt = self._pii_redactor.redact(working_prompt)
                if working_prompt != prompt:
                    result.redacted = True
                    self._record_proof("pii.redact", step=step_name,
                                       entities_redacted=True)
            except Exception as e:
                logger.warning("PII redaction failed for %s: %s", step_name, e)

        # ── 2. Semantic Cache Check ──
        if self._cache:
            try:
                hit = self._cache.get(working_prompt, domain=domain)
                if hit:
                    raw = hit.response
                    if self._pii_redactor and result.redacted:
                        try:
                            raw = self._pii_redactor.deredact(raw)
                        except Exception:
                            pass
                    result.raw_response = raw
                    result.cached = True
                    self._record_proof("cache.hit", step=step_name, domain=domain)
                    return result
                else:
                    self._record_proof("cache.miss", step=step_name, domain=domain)
            except Exception as e:
                logger.warning("Cache lookup failed for %s: %s", step_name, e)

        # ── 3. Rate Limiting ──
        # Detect provider from model/llm
        provider = _detect_provider_name(model)
        limiter = self._get_rate_limiter(provider)

        t0 = time.time()

        if limiter:
            try:
                from langchain_core.messages import HumanMessage
                with limiter.acquire(timeout=30):
                    response = llm.invoke([HumanMessage(content=working_prompt)])
            except ImportError:
                response = llm.invoke(working_prompt)
            except Exception as e:
                logger.warning("Rate limit acquire failed for %s/%s: %s",
                               provider, step_name, e)
                from langchain_core.messages import HumanMessage
                response = llm.invoke([HumanMessage(content=working_prompt)])
        else:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=working_prompt)])

        elapsed_ms = (time.time() - t0) * 1000
        result.latency_ms = elapsed_ms
        raw = response.content

        # ── 4. PII De-redaction ──
        if self._pii_redactor and result.redacted:
            try:
                raw = self._pii_redactor.deredact(raw)
            except Exception as e:
                logger.warning("PII de-redaction failed for %s: %s", step_name, e)

        result.raw_response = raw

        self._record_proof("llm.call", step=step_name, model=model,
                           provider=provider, latency_ms=elapsed_ms,
                           redacted=result.redacted)

        # ── 5. Cache Store ──
        if self._cache and not result.cached:
            try:
                self._cache.put(
                    working_prompt, response.content,
                    domain=domain, model=model,
                )
            except Exception as e:
                logger.warning("Cache store failed for %s: %s", step_name, e)

        # ── 6. Cost Tracking ──
        if self._cost_tracker:
            try:
                # Extract token counts from response metadata if available
                usage = getattr(response, "response_metadata", {})
                if isinstance(usage, dict):
                    usage = usage.get("usage", usage)
                input_tokens = 0
                output_tokens = 0
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens",
                                   usage.get("prompt_tokens", 0))
                    output_tokens = usage.get("output_tokens",
                                    usage.get("completion_tokens", 0))

                result.input_tokens = input_tokens
                result.output_tokens = output_tokens

                self._cost_tracker.record_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    step_name=step_name,
                )
                result.cost_usd = self._cost_tracker.total_cost
                self._record_proof("cost.record", step=step_name, model=model,
                                   input_tokens=input_tokens,
                                   output_tokens=output_tokens)
            except Exception as e:
                logger.warning("Cost tracking failed for %s: %s", step_name, e)

        return result

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 2: Start Gates
    # Called from coordinator/runtime.py → start()
    # ═══════════════════════════════════════════════════════════════

    def check_start_gates(
        self,
        workflow_type: str,
        domain: str,
        case_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run all pre-execution checks. Returns a dict with:
          - blocked: bool (True → reject the start)
          - reason: str (why blocked)
          - guardrail_result: dict (injection scan details)
          - tier_escalation: str | None (tier override from guardrails)
          - spec_manifest: dict | None (config hash for audit)

        Checks:
          1. Kill switch — domain or workflow disabled?
          2. Input guardrails — prompt injection in case_input?
          3. Spec lock — capture config manifest
        """
        self.initialize()
        gate_result = {
            "blocked": False,
            "reason": "",
            "guardrail_result": None,
            "tier_escalation": None,
            "spec_manifest": None,
        }

        # ── 1. Kill Switch ──
        if self._kill_switches:
            try:
                self._kill_switches.check_domain(domain)
                self._record_proof("kill_switch.domain_check",
                                   domain=domain, result="allowed")
            except Exception as e:
                gate_result["blocked"] = True
                gate_result["reason"] = f"Kill switch: {e}"
                self._record_proof("kill_switch.domain_check",
                                   domain=domain, result="blocked",
                                   reason=str(e))
                return gate_result

        # ── 2. Input Guardrails ──
        if self._guardrail:
            try:
                scan = self._guardrail.scan(case_input)
                gate_result["guardrail_result"] = scan.to_dict()
                self._record_proof("guardrail.scan",
                                   workflow=workflow_type, domain=domain,
                                   risk=scan.risk, score=scan.score,
                                   patterns=scan.patterns_matched)
                if scan.risk == "HIGH":
                    gate_result["tier_escalation"] = "hold"
                    logger.warning(
                        "Guardrail HIGH risk for %s/%s: %s",
                        workflow_type, domain, scan.patterns_matched,
                    )
                elif scan.risk == "AMBIGUOUS":
                    gate_result["tier_escalation"] = "gate"
            except Exception as e:
                logger.warning("Guardrail scan failed: %s", e)

        # ── 3. PII — register entities for this case ──
        if self._pii_redactor:
            try:
                self._pii_redactor.register_entities_from_case(case_input)
            except Exception as e:
                logger.warning("PII entity registration failed: %s", e)

        return gate_result

    def capture_spec_manifest(
        self,
        workflow_path: str,
        domain_path: str,
        coordinator_path: str = "",
    ) -> dict[str, Any] | None:
        """Capture config hashes at workflow start for audit trail."""
        try:
            from engine.spec_lock import create_manifest, is_spec_lock_enabled
            if not is_spec_lock_enabled():
                return None
            manifest = create_manifest(
                workflow_path=workflow_path,
                domain_path=domain_path,
                coordinator_config_path=coordinator_path,
            )
            return manifest.to_dict()
        except Exception as e:
            logger.warning("Spec manifest capture failed: %s", e)
            return None

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 3: Tier Resolution
    # Called from coordinator/runtime.py → _resolve_governance_tier()
    # ═══════════════════════════════════════════════════════════════

    def resolve_tier(self, declared_tier: str, domain: str) -> tuple[str, str]:
        """
        Resolve effective tier with circuit breaker override.
        Returns (effective_tier, override_source).
        """
        self.initialize()
        try:
            from engine.tier import resolve_effective_tier
            breaker_override = None
            if self._circuit_breaker:
                breaker_override = self._circuit_breaker.get_tier_override(domain)
            effective, source = resolve_effective_tier(
                declared_tier,
                breaker_override,
                source_labels=["circuit_breaker"],
            )
            self._record_proof("tier.resolve",
                               domain=domain,
                               declared=declared_tier,
                               effective=effective,
                               source=source,
                               breaker_override=breaker_override)
            return effective, source
        except Exception as e:
            logger.warning("Tier resolution failed: %s", e)
            return declared_tier, "declared"

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 4: Step Result Recording
    # Called from nodes.py after parse or from stepper callback
    # ═══════════════════════════════════════════════════════════════

    def record_step_result(
        self,
        domain: str,
        primitive: str,
        confidence: float,
        step_name: str,
        instance_id: str = "",
        state_snapshot: dict[str, Any] | None = None,
    ):
        """
        Record a step completion for circuit breaker + replay.

        Args:
            domain: Domain name
            primitive: Primitive name
            confidence: Output confidence (0-1)
            step_name: Step name
            instance_id: For replay checkpointing
            state_snapshot: For replay checkpointing
        """
        # ── Circuit Breaker ──
        if self._circuit_breaker:
            try:
                floor = 0.5  # default confidence floor
                self._circuit_breaker.record(
                    domain, primitive,
                    confidence=confidence, floor=floor,
                )
                override = self._circuit_breaker.get_tier_override(domain)
                self._record_proof("circuit_breaker.record",
                                   domain=domain, primitive=primitive,
                                   confidence=confidence, floor=floor,
                                   tier_override=override)
            except Exception as e:
                logger.warning("Circuit breaker record failed: %s", e)

        # ── Replay Checkpoint ──
        if self._replay_manager and instance_id and state_snapshot:
            try:
                self._replay_manager.save_checkpoint(
                    trace_id=instance_id,
                    step_name=step_name,
                    step_index=len(state_snapshot.get("steps", [])),
                    state_snapshot=state_snapshot,
                )
                self._record_proof("step.checkpoint",
                                   step=step_name, instance_id=instance_id)
            except Exception as e:
                logger.warning("Replay checkpoint failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 5: Governance Suspension Notification
    # Called from coordinator/runtime.py → _suspend_for_governance()
    # ═══════════════════════════════════════════════════════════════

    def on_governance_suspension(
        self,
        instance_id: str,
        workflow: str,
        domain: str,
        tier: str,
        reason: str,
        approve_url: str = "",
    ):
        """Notify external systems of governance suspension."""
        self._record_proof("suspension.notify",
                           instance_id=instance_id, workflow=workflow,
                           domain=domain, tier=tier, reason=reason)

        # ── Webhook ──
        if self._webhook_notifier:
            try:
                self._webhook_notifier.notify_suspension(
                    instance_id=instance_id,
                    workflow=workflow,
                    domain=domain,
                    tier=tier,
                    step="governance_gate",
                    reason=reason,
                    approve_url=approve_url,
                )
            except Exception as e:
                logger.warning("Webhook notification failed: %s", e)

        # ── HITL State Machine ──
        if self._hitl_state_machine:
            try:
                self._hitl_state_machine.initialize(instance_id)
                self._hitl_state_machine.suspend(instance_id, reason=reason)
            except Exception as e:
                logger.warning("HITL state init failed: %s", e)

    def route_review_task(
        self,
        instance_id: str,
        domain: str,
        tier: str,
    ) -> list[str]:
        """Route a review task to qualified reviewers."""
        if self._hitl_router:
            try:
                result = self._hitl_router.route_task(
                    instance_id=instance_id,
                    domain=domain,
                    tier=tier,
                )
                if result:
                    return [r.reviewer_id for r in [result]
                            if hasattr(r, "reviewer_id")]
                return result.reviewer_ids if hasattr(result, "reviewer_ids") else []
            except Exception as e:
                logger.warning("HITL routing failed: %s", e)
        return []

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 6: Act Protection
    # Called from engine/nodes.py → create_act_node()
    # ═══════════════════════════════════════════════════════════════

    def should_skip_act(self, step_name: str) -> bool:
        """Check if Act should be skipped (shadow mode)."""
        if self._shadow_mode:
            try:
                skip = self._shadow_mode.should_skip_act("act")
                self._record_proof("act.shadow_check",
                                   step=step_name, skip=skip)
                return skip
            except Exception:
                pass
        return False

    def record_shadow_act(
        self,
        instance_id: str,
        step_name: str,
        proposed_actions: list[dict],
    ) -> dict[str, Any]:
        """Record what Act would have done in shadow mode."""
        if self._shadow_mode:
            try:
                self._shadow_mode.record_shadow_act(
                    instance_id=instance_id,
                    step_name=step_name,
                    proposed_actions=proposed_actions,
                )
                return self._shadow_mode.get_shadow_result(step_name)
            except Exception as e:
                logger.warning("Shadow act recording failed: %s", e)
        return {"shadow": True, "actions_taken": []}

    def register_compensation(
        self,
        instance_id: str,
        step_name: str,
        action_description: str,
        compensation_data: dict[str, Any],
        idempotency_key: str = "",
    ):
        """Register a reversible action before execution."""
        if self._compensation:
            try:
                self._compensation.register(
                    instance_id=instance_id,
                    step_name=step_name,
                    idempotency_key=idempotency_key or f"{instance_id}:{step_name}",
                    action_description=action_description,
                    compensation_data=compensation_data,
                )
            except Exception as e:
                logger.warning("Compensation registration failed: %s", e)

    def confirm_compensation(self, idempotency_key: str):
        """Confirm an action succeeded (mark compensation as confirmed)."""
        if self._compensation:
            try:
                self._compensation.confirm(idempotency_key)
            except Exception as e:
                logger.warning("Compensation confirm failed: %s", e)

    def fire_compensations(self, instance_id: str):
        """Fire all pending compensations for an instance."""
        if self._compensation:
            try:
                self._compensation.compensate(instance_id)
            except Exception as e:
                logger.warning("Compensation fire failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    # CHOKEPOINT 7: Kill Switch Checks
    # Called from coordinator before delegation dispatch
    # ═══════════════════════════════════════════════════════════════

    def check_delegation_allowed(self):
        """Check if delegation is enabled."""
        if self._kill_switches:
            try:
                self._kill_switches.check_delegation()
                self._record_proof("delegation.kill_switch_check", result="allowed")
            except Exception as e:
                self._record_proof("delegation.kill_switch_check",
                                   result="blocked", reason=str(e))
                raise e

    def check_act_allowed(self):
        """Check if Act execution is enabled."""
        if self._kill_switches:
            try:
                self._kill_switches.check_act()
                self._record_proof("act.kill_switch_check", result="allowed")
            except Exception as e:
                self._record_proof("act.kill_switch_check",
                                   result="blocked", reason=str(e))
                raise e

    # ═══════════════════════════════════════════════════════════════
    # Diagnostics
    # ═══════════════════════════════════════════════════════════════

    def stats(self) -> dict[str, Any]:
        """Return operational stats from all governance modules."""
        s: dict[str, Any] = {}
        if self._cost_tracker:
            try:
                s["cost"] = self._cost_tracker.summary()
            except Exception:
                pass
        if self._cache:
            try:
                s["cache"] = self._cache.stats().to_dict()
            except Exception:
                pass
        if self._circuit_breaker:
            try:
                s["circuit_breakers"] = self._circuit_breaker.get_all_states()
            except Exception:
                pass
        if self._kill_switches:
            try:
                s["kill_switches"] = self._kill_switches.status()
            except Exception:
                pass
        if self._pii_redactor:
            try:
                s["pii"] = self._pii_redactor.audit_summary()
            except Exception:
                pass
        if self._shadow_mode:
            try:
                s["shadow"] = self._shadow_mode.get_stats()
            except Exception:
                pass
        for provider, limiter in self._rate_limiters.items():
            if limiter:
                try:
                    s.setdefault("rate_limits", {})[provider] = limiter.metrics()
                except Exception:
                    pass
        return s

    # ═══════════════════════════════════════════════════════════════
    # Proof Ledger — records every governance decision for audit
    # ═══════════════════════════════════════════════════════════════

    def _record_proof(self, event_type: str, **details):
        """Record a governance event to the proof ledger."""
        entry = {
            "timestamp": time.time(),
            "event": event_type,
            **details,
        }
        with self._proof_lock:
            self._proof_ledger.append(entry)

    def proof(self) -> dict[str, Any]:
        """
        Return the consolidated governance proof for the current run.

        This is the audit artifact that demonstrates every governance
        check that was performed, every decision made, and every
        protection that was active during execution.
        """
        with self._proof_lock:
            ledger = list(self._proof_ledger)

        # Categorize events
        categories: dict[str, list] = {}
        for entry in ledger:
            cat = entry["event"].split(".")[0]
            categories.setdefault(cat, []).append(entry)

        # Build summary
        return {
            "governance_proof": {
                "generated_at": time.time(),
                "total_events": len(ledger),
                "modules_active": self._active_modules(),
                "summary": {
                    "guardrail_scans": len(categories.get("guardrail", [])),
                    "pii_redactions": len(categories.get("pii", [])),
                    "cache_checks": len(categories.get("cache", [])),
                    "cache_hits": sum(
                        1 for e in categories.get("cache", [])
                        if e.get("hit")
                    ),
                    "rate_limit_acquires": len(categories.get("rate_limit", [])),
                    "llm_calls": len(categories.get("llm", [])),
                    "cost_records": len(categories.get("cost", [])),
                    "tier_resolutions": len(categories.get("tier", [])),
                    "circuit_breaker_records": len(categories.get("circuit_breaker", [])),
                    "kill_switch_checks": len(categories.get("kill_switch", [])),
                    "step_recordings": len(categories.get("step", [])),
                    "act_checks": len(categories.get("act", [])),
                    "delegation_checks": len(categories.get("delegation", [])),
                    "suspension_notifications": len(categories.get("suspension", [])),
                },
                "module_stats": self.stats(),
                "events": ledger,
            }
        }

    def reset_proof(self):
        """Clear the proof ledger (between test runs)."""
        with self._proof_lock:
            self._proof_ledger.clear()

    def _active_modules(self) -> list[str]:
        """Return list of active (successfully initialized) modules."""
        modules = []
        if self._guardrail: modules.append("guardrails")
        if self._pii_redactor: modules.append("pii")
        if self._cache: modules.append("semantic_cache")
        if self._cost_tracker: modules.append("cost_tracker")
        if self._circuit_breaker: modules.append("circuit_breaker")
        if self._kill_switches: modules.append("kill_switch")
        if self._shadow_mode: modules.append("shadow_mode")
        if self._compensation: modules.append("compensation")
        if self._webhook_notifier: modules.append("webhook")
        if self._replay_manager: modules.append("replay")
        if self._hitl_router: modules.append("hitl_routing")
        if self._hitl_state_machine: modules.append("hitl_state")
        if self._rate_limiters: modules.append("rate_limit")
        return modules


# ═══════════════════════════════════════════════════════════════════
# Singleton Access
# ═══════════════════════════════════════════════════════════════════

def get_governance() -> GovernancePipeline:
    """Get the singleton governance pipeline."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GovernancePipeline()
    return _instance


def reset_governance():
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _detect_provider_name(model: str) -> str:
    """Detect provider name from model string for rate limiting."""
    model_lower = model.lower()
    if "gemini" in model_lower or "google" in model_lower:
        return "google"
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    # Try env var
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider:
        return provider
    return "default"
