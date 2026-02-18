"""
Cognitive Core — Semantic Cache (S-019)

Caches LLM responses to avoid redundant calls for similar inputs.
Two-layer strategy:
  Layer 1: Exact-match hash cache (always on when caching is enabled)
  Layer 2: Vector similarity cache (requires embedding provider)

On/off toggle via config: CC_CACHE_ENABLED=true/false

Features:
  - TTL-based expiry
  - Cache hit/miss metrics
  - Per-domain enable/disable
  - Audit trail annotation when cached result is returned
  - Thread-safe

Usage:
    from engine.semantic_cache import SemanticCache

    cache = SemanticCache(enabled=True, ttl_seconds=3600)

    # Check cache
    hit = cache.get(prompt_text, domain="card_dispute")
    if hit:
        return hit.response  # Cached result

    # After LLM call, store result
    cache.put(prompt_text, response_text, domain="card_dispute", model="gemini-2.0-flash")
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.cache")


@dataclass
class CacheEntry:
    """A cached LLM response."""
    key: str
    prompt_hash: str
    response: str
    domain: str
    model: str
    created_at: float
    ttl_seconds: int
    hit_count: int = 0
    last_hit_at: float = 0.0
    similarity_score: float = 1.0  # 1.0 = exact match

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class CacheStats:
    """Cache performance metrics."""
    total_gets: int = 0
    exact_hits: int = 0
    similarity_hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    expired: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_gets == 0:
            return 0.0
        return (self.exact_hits + self.similarity_hits) / self.total_gets

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gets": self.total_gets,
            "exact_hits": self.exact_hits,
            "similarity_hits": self.similarity_hits,
            "misses": self.misses,
            "puts": self.puts,
            "evictions": self.evictions,
            "expired": self.expired,
            "hit_rate": round(self.hit_rate, 3),
        }


class SemanticCache:
    """
    Two-layer LLM response cache.

    Layer 1: Exact prompt hash match (OrderedDict LRU)
    Layer 2: Vector similarity (optional, requires embed_fn)
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
        similarity_threshold: float = 0.98,
        embed_fn: Callable[[str], list[float]] | None = None,
        disabled_domains: list[str] | None = None,
    ):
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.embed_fn = embed_fn
        self.disabled_domains = set(disabled_domains or [])

        # Layer 1: Exact match (prompt_hash → CacheEntry)
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()

        # Layer 2: Vector index (list of (embedding, CacheEntry))
        self._vectors: list[tuple[list[float], CacheEntry]] = []

        self._stats = CacheStats()
        self._lock = threading.Lock()

    def get(
        self,
        prompt_text: str,
        domain: str = "",
    ) -> CacheEntry | None:
        """
        Look up a cached response.
        Checks exact match first, then vector similarity if available.
        """
        if not self.enabled:
            return None
        if domain and domain in self.disabled_domains:
            return None

        with self._lock:
            self._stats.total_gets += 1

            # Layer 1: Exact match
            prompt_hash = _hash_prompt(prompt_text)
            entry = self._exact.get(prompt_hash)
            if entry is not None:
                if entry.is_expired:
                    del self._exact[prompt_hash]
                    self._stats.expired += 1
                elif domain and entry.domain != domain:
                    pass  # Domain mismatch
                else:
                    entry.hit_count += 1
                    entry.last_hit_at = time.time()
                    self._exact.move_to_end(prompt_hash)
                    self._stats.exact_hits += 1
                    return entry

            # Layer 2: Vector similarity
            if self.embed_fn and self._vectors:
                try:
                    query_vec = self.embed_fn(prompt_text)
                    best_score = 0.0
                    best_entry = None

                    for vec, ent in self._vectors:
                        if ent.is_expired:
                            continue
                        if domain and ent.domain != domain:
                            continue
                        score = _cosine_similarity(query_vec, vec)
                        if score > best_score:
                            best_score = score
                            best_entry = ent

                    if best_entry and best_score >= self.similarity_threshold:
                        best_entry.hit_count += 1
                        best_entry.last_hit_at = time.time()
                        best_entry.similarity_score = best_score
                        self._stats.similarity_hits += 1
                        return best_entry
                except Exception as e:
                    logger.warning("Vector similarity search failed: %s", e)

            self._stats.misses += 1
            return None

    def put(
        self,
        prompt_text: str,
        response: str,
        domain: str = "",
        model: str = "",
        ttl_seconds: int | None = None,
    ):
        """Store a response in the cache."""
        if not self.enabled:
            return
        if domain and domain in self.disabled_domains:
            return

        with self._lock:
            prompt_hash = _hash_prompt(prompt_text)
            ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

            entry = CacheEntry(
                key=prompt_hash,
                prompt_hash=prompt_hash,
                response=response,
                domain=domain,
                model=model,
                created_at=time.time(),
                ttl_seconds=ttl,
            )

            # Layer 1: Exact store
            self._exact[prompt_hash] = entry
            self._exact.move_to_end(prompt_hash)
            self._stats.puts += 1

            # Evict if over capacity
            while len(self._exact) > self.max_entries:
                evicted_key, _ = self._exact.popitem(last=False)
                self._stats.evictions += 1

            # Layer 2: Vector store
            if self.embed_fn:
                try:
                    vec = self.embed_fn(prompt_text)
                    self._vectors.append((vec, entry))
                    # Cap vector store
                    if len(self._vectors) > self.max_entries:
                        self._vectors = self._vectors[-self.max_entries:]
                except Exception as e:
                    logger.warning("Vector embedding failed: %s", e)

    def invalidate(self, domain: str | None = None):
        """Clear cache entries. If domain specified, only that domain."""
        with self._lock:
            if domain is None:
                self._exact.clear()
                self._vectors.clear()
            else:
                keys_to_remove = [
                    k for k, v in self._exact.items() if v.domain == domain
                ]
                for k in keys_to_remove:
                    del self._exact[k]
                self._vectors = [
                    (v, e) for v, e in self._vectors if e.domain != domain
                ]

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            expired_keys = [
                k for k, v in self._exact.items() if v.is_expired
            ]
            for k in expired_keys:
                del self._exact[k]
                self._stats.expired += 1

            before = len(self._vectors)
            self._vectors = [
                (v, e) for v, e in self._vectors if not e.is_expired
            ]
            vector_expired = before - len(self._vectors)
            return len(expired_keys) + vector_expired

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._exact)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _hash_prompt(text: str) -> str:
    """SHA-256 hash of prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
