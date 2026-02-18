"""
Cognitive Core — LLM Retry with Backoff & Same-Provider Fallback

Wraps LLM invocations with:
  - Configurable retry on transient failures (timeout, rate limit, 5xx, parse failure)
  - Exponential backoff between retries
  - Same-provider fallback model (e.g., gemini-2.0-flash → gemini-2.0-pro)
  - Circuit breaker: N consecutive failures → stop accepting calls
  - Structured logging of every attempt

Design decisions:
  - Config lives in llm_config.yaml per-provider (P-001 decision: per-provider)
  - Fallback is same-provider (P-001 decision: Option B)
  - Retry wraps both LLM call AND JSON parse (a parse failure is retried)

Usage:
    from engine.retry import invoke_with_retry, get_retry_policy

    policy = get_retry_policy("google")
    result = invoke_with_retry(llm, messages, policy, step_name="classify")
    # result.content is the raw LLM response string
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.retry")


# ═══════════════════════════════════════════════════════════════════
# Retry Policy
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetryPolicy:
    """Configuration for LLM retry behavior."""
    max_attempts: int = 3
    backoff_base: float = 1.0       # seconds; actual delay = base * 2^attempt + jitter
    backoff_max: float = 30.0       # cap on delay between retries
    jitter: float = 0.2             # ±20% randomization on backoff
    fallback_model: str | None = None  # same-provider fallback model name

    # Circuit breaker
    circuit_breaker_threshold: int = 5   # consecutive failures to open circuit
    circuit_breaker_reset_seconds: float = 60.0  # time before half-open

    # What counts as retryable
    retryable_exceptions: tuple = (
        TimeoutError,
        ConnectionError,
        OSError,
    )
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)

    # Whether to retry on JSON parse failure
    retry_on_parse_failure: bool = True


DEFAULT_POLICY = RetryPolicy()


# ═══════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open (too many consecutive failures)."""
    pass


class CircuitBreaker:
    """
    Per-provider circuit breaker.

    States:
      CLOSED  — normal operation, failures increment counter
      OPEN    — all calls rejected, waiting for reset timeout
      HALF_OPEN — one probe call allowed; success → CLOSED, failure → OPEN
    """

    def __init__(self, threshold: int = 5, reset_seconds: float = 60.0):
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._last_failure_time = 0.0
        self._state = "closed"  # closed | open | half_open
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time >= self.reset_seconds:
                    self._state = "half_open"
            return self._state

    def check(self):
        """Raise CircuitBreakerOpen if circuit is open."""
        s = self.state
        if s == "open":
            raise CircuitBreakerOpen(
                f"Circuit breaker open: {self._failures} consecutive failures. "
                f"Resets in {self.reset_seconds - (time.time() - self._last_failure_time):.0f}s"
            )
        # half_open allows one probe call through

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def record_failure(self):
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.threshold:
                self._state = "open"

    def reset(self):
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._last_failure_time = 0.0


# Global circuit breakers per provider
_circuit_breakers: dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(provider: str, policy: RetryPolicy) -> CircuitBreaker:
    """Get or create a circuit breaker for a provider."""
    with _cb_lock:
        if provider not in _circuit_breakers:
            _circuit_breakers[provider] = CircuitBreaker(
                threshold=policy.circuit_breaker_threshold,
                reset_seconds=policy.circuit_breaker_reset_seconds,
            )
        return _circuit_breakers[provider]


def reset_all_circuit_breakers():
    """Reset all circuit breakers. For testing."""
    with _cb_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
        _circuit_breakers.clear()


# ═══════════════════════════════════════════════════════════════════
# Retry Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetryResult:
    """Result of an LLM invocation with retry."""
    content: str                      # raw LLM response text
    attempts: int                     # total attempts made (1 = first try succeeded)
    model_used: str                   # which model produced the response
    used_fallback: bool               # whether the fallback model was used
    total_latency: float              # total wall time including retries
    attempt_log: list[dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Config Loader
# ═══════════════════════════════════════════════════════════════════

def _find_config_path() -> str | None:
    """Find llm_config.yaml."""
    candidates = [
        os.environ.get("LLM_CONFIG_PATH", ""),
        "llm_config.yaml",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_config.yaml"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def get_retry_policy(provider: str | None = None) -> RetryPolicy:
    """
    Load retry policy from llm_config.yaml for the given provider.

    Config format:
        retry:
          google:
            max_attempts: 3
            backoff_base: 1.0
            fallback_model: gemini-2.5-pro
          azure:
            max_attempts: 3
            fallback_model: gpt-4o
    """
    config_path = _find_config_path()
    if not config_path:
        return DEFAULT_POLICY

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    retry_cfg = cfg.get("retry", {})

    # Provider-specific overrides
    provider_cfg = {}
    if provider and provider in retry_cfg:
        provider_cfg = retry_cfg[provider]
    elif "default" in retry_cfg:
        provider_cfg = retry_cfg["default"]

    if not provider_cfg:
        return DEFAULT_POLICY

    return RetryPolicy(
        max_attempts=provider_cfg.get("max_attempts", DEFAULT_POLICY.max_attempts),
        backoff_base=provider_cfg.get("backoff_base", DEFAULT_POLICY.backoff_base),
        backoff_max=provider_cfg.get("backoff_max", DEFAULT_POLICY.backoff_max),
        jitter=provider_cfg.get("jitter", DEFAULT_POLICY.jitter),
        fallback_model=provider_cfg.get("fallback_model", DEFAULT_POLICY.fallback_model),
        circuit_breaker_threshold=provider_cfg.get(
            "circuit_breaker_threshold", DEFAULT_POLICY.circuit_breaker_threshold),
        circuit_breaker_reset_seconds=provider_cfg.get(
            "circuit_breaker_reset_seconds", DEFAULT_POLICY.circuit_breaker_reset_seconds),
        retry_on_parse_failure=provider_cfg.get(
            "retry_on_parse_failure", DEFAULT_POLICY.retry_on_parse_failure),
    )


# ═══════════════════════════════════════════════════════════════════
# Retry Logic
# ═══════════════════════════════════════════════════════════════════

def _is_retryable(error: Exception, policy: RetryPolicy) -> bool:
    """Determine if an exception is retryable."""
    # Direct type match
    if isinstance(error, policy.retryable_exceptions):
        return True

    # Check for HTTP status codes in the error message or attributes
    err_str = str(error).lower()

    # Auth errors are NOT retryable
    if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "forbidden" in err_str:
        return False

    # Rate limit
    if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
        return True

    # Server errors
    for code in policy.retryable_status_codes:
        if str(code) in err_str:
            return True

    # Connection/timeout patterns
    if any(term in err_str for term in ["timeout", "timed out", "connection", "unavailable"]):
        return True

    return False


def _calculate_backoff(attempt: int, policy: RetryPolicy) -> float:
    """Calculate backoff delay with jitter."""
    base_delay = policy.backoff_base * (2 ** attempt)
    capped = min(base_delay, policy.backoff_max)
    jitter_range = capped * policy.jitter
    actual = capped + random.uniform(-jitter_range, jitter_range)
    return max(0.0, actual)


class ParseFailure(Exception):
    """Raised when LLM response fails to parse as valid JSON."""
    def __init__(self, raw_response: str, parse_error: Exception):
        self.raw_response = raw_response
        self.parse_error = parse_error
        super().__init__(f"JSON parse failed: {parse_error}")


def invoke_with_retry(
    llm: Any,
    messages: list,
    policy: RetryPolicy | None = None,
    step_name: str = "",
    provider: str = "unknown",
    model_name: str = "unknown",
    parse_fn: Callable[[str], Any] | None = None,
    fallback_llm: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RetryResult:
    """
    Invoke an LLM with retry, backoff, fallback, and circuit breaker.

    Args:
        llm:          Primary LLM instance (.invoke(messages) -> response)
        messages:     List of messages to send
        policy:       RetryPolicy (or default)
        step_name:    For logging
        provider:     Provider name for circuit breaker scoping
        model_name:   Model name for logging
        parse_fn:     Optional JSON parse function. If provided, parse failures
                      are retried (when policy.retry_on_parse_failure=True).
                      Signature: parse_fn(raw_response_text) -> parsed_dict
                      Should raise Exception on parse failure.
        fallback_llm: Fallback LLM instance (same provider, different model).
                      Used after primary exhausts all retries.
        sleep_fn:     Sleep function (injectable for testing)

    Returns:
        RetryResult with the successful response

    Raises:
        CircuitBreakerOpen: If circuit breaker is open
        Exception: The last exception if all retries and fallback exhausted
    """
    if policy is None:
        policy = DEFAULT_POLICY

    cb = get_circuit_breaker(provider, policy)
    cb.check()

    attempt_log = []
    last_error = None
    total_t0 = time.time()

    # ── Try primary model ──
    for attempt in range(policy.max_attempts):
        entry = {
            "attempt": attempt + 1,
            "model": model_name,
            "is_fallback": False,
            "step": step_name,
        }
        try:
            t0 = time.time()
            response = llm.invoke(messages)
            raw = response.content
            elapsed = time.time() - t0
            entry["latency_s"] = round(elapsed, 2)
            entry["status"] = "success"

            # Optional parse check
            if parse_fn is not None:
                try:
                    parse_fn(raw)
                except Exception as pe:
                    if policy.retry_on_parse_failure and attempt < policy.max_attempts - 1:
                        entry["status"] = "parse_failure"
                        entry["error"] = str(pe)[:200]
                        attempt_log.append(entry)
                        logger.warning(
                            "LLM parse failure (attempt %d/%d, step=%s): %s",
                            attempt + 1, policy.max_attempts, step_name, str(pe)[:100]
                        )
                        delay = _calculate_backoff(attempt, policy)
                        entry["backoff_s"] = round(delay, 2)
                        sleep_fn(delay)
                        continue
                    else:
                        # Last attempt or parse retry disabled — return raw, let caller handle
                        entry["status"] = "parse_failure_final"
                        entry["error"] = str(pe)[:200]
                        attempt_log.append(entry)
                        cb.record_success()  # LLM call itself succeeded
                        return RetryResult(
                            content=raw,
                            attempts=attempt + 1,
                            model_used=model_name,
                            used_fallback=False,
                            total_latency=time.time() - total_t0,
                            attempt_log=attempt_log,
                        )

            # Full success
            attempt_log.append(entry)
            cb.record_success()
            logger.debug(
                "LLM call succeeded (attempt %d, step=%s, model=%s, %.1fs)",
                attempt + 1, step_name, model_name, elapsed,
            )
            return RetryResult(
                content=raw,
                attempts=attempt + 1,
                model_used=model_name,
                used_fallback=False,
                total_latency=time.time() - total_t0,
                attempt_log=attempt_log,
            )

        except Exception as e:
            elapsed = time.time() - t0
            entry["latency_s"] = round(elapsed, 2)
            entry["error"] = str(e)[:200]
            last_error = e

            if not _is_retryable(e, policy):
                entry["status"] = "non_retryable"
                attempt_log.append(entry)
                cb.record_failure()
                logger.error(
                    "LLM non-retryable error (step=%s, model=%s): %s",
                    step_name, model_name, str(e)[:100],
                )
                raise

            entry["status"] = "retryable_error"
            attempt_log.append(entry)
            logger.warning(
                "LLM retryable error (attempt %d/%d, step=%s): %s",
                attempt + 1, policy.max_attempts, step_name, str(e)[:100],
            )

            if attempt < policy.max_attempts - 1:
                delay = _calculate_backoff(attempt, policy)
                entry["backoff_s"] = round(delay, 2)
                sleep_fn(delay)

    # ── Primary exhausted → try fallback ──
    if fallback_llm is not None:
        fallback_model = policy.fallback_model or "fallback"
        logger.info(
            "Primary model exhausted %d attempts. Trying fallback: %s (step=%s)",
            policy.max_attempts, fallback_model, step_name,
        )
        entry = {
            "attempt": policy.max_attempts + 1,
            "model": fallback_model,
            "is_fallback": True,
            "step": step_name,
        }
        try:
            t0 = time.time()
            response = fallback_llm.invoke(messages)
            raw = response.content
            elapsed = time.time() - t0
            entry["latency_s"] = round(elapsed, 2)
            entry["status"] = "success"
            attempt_log.append(entry)
            cb.record_success()
            logger.info(
                "Fallback model succeeded (step=%s, model=%s, %.1fs)",
                step_name, fallback_model, elapsed,
            )
            return RetryResult(
                content=raw,
                attempts=policy.max_attempts + 1,
                model_used=fallback_model,
                used_fallback=True,
                total_latency=time.time() - total_t0,
                attempt_log=attempt_log,
            )
        except Exception as e:
            entry["latency_s"] = round(time.time() - t0, 2)
            entry["status"] = "fallback_failed"
            entry["error"] = str(e)[:200]
            attempt_log.append(entry)
            cb.record_failure()
            last_error = e
            logger.error(
                "Fallback model also failed (step=%s, model=%s): %s",
                step_name, fallback_model, str(e)[:100],
            )

    # ── All attempts exhausted ──
    cb.record_failure()
    logger.error(
        "All retry attempts exhausted (step=%s, attempts=%d, fallback=%s)",
        step_name, len(attempt_log), "yes" if fallback_llm else "no",
    )
    raise last_error
