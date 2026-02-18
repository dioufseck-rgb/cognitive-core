"""
Cognitive Core — LLM Rate Limiting & Concurrency Control

Prevents cascade failures from LLM API throttling by controlling:
  - Concurrent call count per provider (semaphore)
  - Request rate per minute per provider (token bucket)
  - Backpressure when queue is full

Design: threading.Semaphore (Option A) — sync-compatible with current
LangGraph execution. Abstract interface allows swap to Redis-based
limiter for horizontal scaling later.

Config in llm_config.yaml:
    rate_limits:
      google:
        max_concurrent: 10
        requests_per_minute: 60
      azure:
        max_concurrent: 20
        requests_per_minute: 100

Usage:
    limiter = get_rate_limiter("google")
    with limiter.acquire(timeout=30):
        response = llm.invoke(messages)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("cognitive_core.rate_limit")


# ═══════════════════════════════════════════════════════════════════
# Rate Limit Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RateLimitConfig:
    """Configuration for a provider's rate limits."""
    max_concurrent: int = 10
    requests_per_minute: int = 60
    queue_timeout: float = 30.0  # seconds to wait for a slot


DEFAULT_CONFIG = RateLimitConfig()


# ═══════════════════════════════════════════════════════════════════
# Token Bucket Rate Limiter
# ═══════════════════════════════════════════════════════════════════

class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Tokens replenish at a fixed rate. Each request consumes one token.
    If no tokens available, caller waits or gets rejected.
    """

    def __init__(self, rate_per_minute: int):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.max_tokens = float(rate_per_minute)
        self._tokens = float(rate_per_minute)  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def try_acquire(self) -> bool:
        """Try to consume one token. Returns True if available."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def wait_for_token(self, timeout: float = 30.0) -> bool:
        """
        Wait until a token is available or timeout.

        Returns True if token acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.try_acquire():
                return True
            # Sleep for the time needed for one token
            wait_time = min(1.0 / self.rate if self.rate > 0 else 1.0, 0.1)
            time.sleep(wait_time)
        return False

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


# ═══════════════════════════════════════════════════════════════════
# Provider Rate Limiter
# ═══════════════════════════════════════════════════════════════════

class BackpressureError(Exception):
    """Raised when rate limiter queue is full and timeout expires."""
    pass


class ProviderRateLimiter:
    """
    Combined concurrency + rate limiter for an LLM provider.

    Uses:
      - threading.Semaphore for concurrent call limiting
      - TokenBucket for request rate limiting
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._semaphore = threading.Semaphore(self.config.max_concurrent)
        self._bucket = TokenBucket(self.config.requests_per_minute)
        self._metrics = _LimiterMetrics()

    def acquire(self, timeout: float | None = None):
        """
        Context manager that acquires both concurrency slot and rate token.

        Usage:
            with limiter.acquire(timeout=30):
                response = llm.invoke(messages)

        Raises BackpressureError if timeout expires.
        """
        return _AcquireContext(self, timeout or self.config.queue_timeout)

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics.snapshot()


class _LimiterMetrics:
    """Thread-safe metrics for the rate limiter."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_waits = 0
        self._total_rejections = 0
        self._total_wait_ms = 0.0
        self._current_active = 0
        self._peak_active = 0

    def record_acquire(self, wait_ms: float):
        with self._lock:
            self._total_requests += 1
            if wait_ms > 0:
                self._total_waits += 1
                self._total_wait_ms += wait_ms
            self._current_active += 1
            self._peak_active = max(self._peak_active, self._current_active)

    def record_release(self):
        with self._lock:
            self._current_active = max(0, self._current_active - 1)

    def record_rejection(self):
        with self._lock:
            self._total_rejections += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_waits": self._total_waits,
                "total_rejections": self._total_rejections,
                "avg_wait_ms": round(
                    self._total_wait_ms / self._total_waits, 1
                ) if self._total_waits > 0 else 0.0,
                "current_active": self._current_active,
                "peak_active": self._peak_active,
            }


class _AcquireContext:
    """Context manager for acquiring rate limiter slots."""

    def __init__(self, limiter: ProviderRateLimiter, timeout: float):
        self._limiter = limiter
        self._timeout = timeout
        self._acquired = False

    def __enter__(self):
        t0 = time.monotonic()

        # Step 1: Acquire semaphore (concurrency limit)
        if not self._limiter._semaphore.acquire(timeout=self._timeout):
            self._limiter._metrics.record_rejection()
            raise BackpressureError(
                f"Concurrency limit reached ({self._limiter.config.max_concurrent}). "
                f"Timeout after {self._timeout}s."
            )

        # Step 2: Acquire rate token
        remaining = self._timeout - (time.monotonic() - t0)
        if remaining <= 0 or not self._limiter._bucket.wait_for_token(remaining):
            self._limiter._semaphore.release()
            self._limiter._metrics.record_rejection()
            raise BackpressureError(
                f"Rate limit reached ({self._limiter.config.requests_per_minute}/min). "
                f"Timeout after {self._timeout}s."
            )

        wait_ms = (time.monotonic() - t0) * 1000
        self._limiter._metrics.record_acquire(wait_ms)
        self._acquired = True
        return self

    def __exit__(self, *args):
        if self._acquired:
            self._limiter._semaphore.release()
            self._limiter._metrics.record_release()


# ═══════════════════════════════════════════════════════════════════
# Global Limiter Registry
# ═══════════════════════════════════════════════════════════════════

_limiters: dict[str, ProviderRateLimiter] = {}
_limiter_lock = threading.Lock()


def get_rate_limiter(provider: str) -> ProviderRateLimiter:
    """Get or create a rate limiter for a provider."""
    with _limiter_lock:
        if provider not in _limiters:
            config = _load_config_for_provider(provider)
            _limiters[provider] = ProviderRateLimiter(config)
        return _limiters[provider]


def reset_all_limiters():
    """Reset all limiters. For testing."""
    with _limiter_lock:
        _limiters.clear()


def _load_config_for_provider(provider: str) -> RateLimitConfig:
    """Load rate limit config from llm_config.yaml."""
    config_path = _find_config_path()
    if not config_path:
        return DEFAULT_CONFIG

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    rate_cfg = cfg.get("rate_limits", {})
    provider_cfg = rate_cfg.get(provider, rate_cfg.get("default", {}))

    if not provider_cfg:
        return DEFAULT_CONFIG

    return RateLimitConfig(
        max_concurrent=provider_cfg.get("max_concurrent", DEFAULT_CONFIG.max_concurrent),
        requests_per_minute=provider_cfg.get("requests_per_minute", DEFAULT_CONFIG.requests_per_minute),
        queue_timeout=provider_cfg.get("queue_timeout", DEFAULT_CONFIG.queue_timeout),
    )


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
