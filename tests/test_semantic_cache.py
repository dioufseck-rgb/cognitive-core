"""Tests for S-019: Semantic Cache."""

import importlib.util
import math
import os
import sys
import threading
import time
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_mod_path = os.path.join(_base, "engine", "semantic_cache.py")
_spec = importlib.util.spec_from_file_location("engine.semantic_cache", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.semantic_cache"] = _mod
_spec.loader.exec_module(_mod)

SemanticCache = _mod.SemanticCache
CacheEntry = _mod.CacheEntry
_cosine_similarity = _mod._cosine_similarity
_hash_prompt = _mod._hash_prompt


class TestExactMatchCache(unittest.TestCase):
    """Test Layer 1: exact prompt hash matching."""

    def test_put_and_get(self):
        cache = SemanticCache()
        cache.put("What is the weather?", "It's sunny", domain="test")
        hit = cache.get("What is the weather?", domain="test")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response, "It's sunny")

    def test_miss(self):
        cache = SemanticCache()
        self.assertIsNone(cache.get("unknown prompt"))

    def test_different_prompts_different_entries(self):
        cache = SemanticCache()
        cache.put("prompt A", "response A")
        cache.put("prompt B", "response B")
        self.assertEqual(cache.get("prompt A").response, "response A")
        self.assertEqual(cache.get("prompt B").response, "response B")

    def test_overwrite_same_prompt(self):
        cache = SemanticCache()
        cache.put("prompt", "old response")
        cache.put("prompt", "new response")
        self.assertEqual(cache.get("prompt").response, "new response")

    def test_hit_count_increments(self):
        cache = SemanticCache()
        cache.put("prompt", "response")
        cache.get("prompt")
        cache.get("prompt")
        hit = cache.get("prompt")
        self.assertEqual(hit.hit_count, 3)


class TestCacheDisabled(unittest.TestCase):
    """Test on/off toggle."""

    def test_disabled_returns_none(self):
        cache = SemanticCache(enabled=False)
        cache.put("prompt", "response")
        self.assertIsNone(cache.get("prompt"))

    def test_disabled_domain(self):
        cache = SemanticCache(disabled_domains=["sensitive_domain"])
        cache.put("prompt", "response", domain="sensitive_domain")
        self.assertIsNone(cache.get("prompt", domain="sensitive_domain"))

    def test_enabled_domain_works(self):
        cache = SemanticCache(disabled_domains=["sensitive"])
        cache.put("prompt", "response", domain="normal")
        self.assertIsNotNone(cache.get("prompt", domain="normal"))


class TestTTL(unittest.TestCase):
    """Test time-to-live expiry."""

    def test_entry_expires(self):
        cache = SemanticCache(ttl_seconds=1)
        cache.put("prompt", "response")
        time.sleep(1.5)
        self.assertIsNone(cache.get("prompt"))

    def test_entry_not_expired(self):
        cache = SemanticCache(ttl_seconds=60)
        cache.put("prompt", "response")
        self.assertIsNotNone(cache.get("prompt"))

    def test_no_ttl_never_expires(self):
        cache = SemanticCache(ttl_seconds=0)
        cache.put("prompt", "response", ttl_seconds=0)
        # Entry should not expire
        entry = CacheEntry(
            key="k", prompt_hash="ph", response="r", domain="d",
            model="m", created_at=time.time() - 999999, ttl_seconds=0,
        )
        self.assertFalse(entry.is_expired)

    def test_custom_ttl_per_entry(self):
        cache = SemanticCache(ttl_seconds=3600)
        cache.put("prompt", "response", ttl_seconds=1)
        time.sleep(1.5)
        self.assertIsNone(cache.get("prompt"))

    def test_cleanup_expired(self):
        cache = SemanticCache(ttl_seconds=1)
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        time.sleep(1.5)
        removed = cache.cleanup_expired()
        self.assertEqual(removed, 2)
        self.assertEqual(cache.size, 0)


class TestLRUEviction(unittest.TestCase):
    """Test LRU eviction when cache is full."""

    def test_evicts_oldest(self):
        cache = SemanticCache(max_entries=3)
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        cache.put("p3", "r3")
        cache.put("p4", "r4")  # Should evict p1
        self.assertIsNone(cache.get("p1"))
        self.assertIsNotNone(cache.get("p2"))
        self.assertIsNotNone(cache.get("p4"))

    def test_eviction_stats(self):
        cache = SemanticCache(max_entries=2)
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        cache.put("p3", "r3")
        self.assertEqual(cache.stats.evictions, 1)

    def test_access_refreshes_lru(self):
        cache = SemanticCache(max_entries=3)
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        cache.put("p3", "r3")
        cache.get("p1")  # Access p1 — moves it to end
        cache.put("p4", "r4")  # Should evict p2 (oldest untouched)
        self.assertIsNotNone(cache.get("p1"))  # p1 was refreshed
        self.assertIsNone(cache.get("p2"))  # p2 was evicted


class TestVectorSimilarity(unittest.TestCase):
    """Test Layer 2: vector similarity matching."""

    @staticmethod
    def _simple_embed(text: str) -> list[float]:
        """Deterministic mock embedding based on character frequencies."""
        vec = [0.0] * 26
        for c in text.lower():
            if 'a' <= c <= 'z':
                vec[ord(c) - ord('a')] += 1.0
        # Normalize
        mag = sum(v * v for v in vec) ** 0.5
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec

    def test_similar_prompts_hit(self):
        cache = SemanticCache(
            embed_fn=self._simple_embed,
            similarity_threshold=0.90,
        )
        cache.put("What is the weather today?", "It's sunny")
        # Very similar prompt
        hit = cache.get("What is the weather today")
        # Character similarity should be very high
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response, "It's sunny")

    def test_dissimilar_prompts_miss(self):
        cache = SemanticCache(
            embed_fn=self._simple_embed,
            similarity_threshold=0.98,
        )
        cache.put("apple banana cherry", "fruits")
        hit = cache.get("xyz quantum physics")
        self.assertIsNone(hit)

    def test_exact_match_preferred_over_vector(self):
        """Layer 1 should be checked before Layer 2."""
        cache = SemanticCache(
            embed_fn=self._simple_embed,
            similarity_threshold=0.90,
        )
        cache.put("exact prompt", "exact response")
        hit = cache.get("exact prompt")
        self.assertEqual(hit.response, "exact response")
        self.assertEqual(cache.stats.exact_hits, 1)
        self.assertEqual(cache.stats.similarity_hits, 0)

    def test_no_embed_fn_skips_vector(self):
        cache = SemanticCache(embed_fn=None)
        cache.put("prompt", "response")
        # Different prompt, no vector search
        hit = cache.get("slightly different prompt")
        self.assertIsNone(hit)


class TestCosineHelper(unittest.TestCase):
    """Test cosine similarity calculation."""

    def test_identical_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1, 0], [1, 0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_empty_vectors(self):
        self.assertEqual(_cosine_similarity([], []), 0.0)

    def test_zero_vector(self):
        self.assertEqual(_cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_different_lengths(self):
        self.assertEqual(_cosine_similarity([1, 2], [1, 2, 3]), 0.0)


class TestDomainFiltering(unittest.TestCase):
    """Test domain-specific caching."""

    def test_domain_mismatch_miss(self):
        cache = SemanticCache()
        cache.put("prompt", "response", domain="domain_a")
        hit = cache.get("prompt", domain="domain_b")
        self.assertIsNone(hit)

    def test_domain_match_hit(self):
        cache = SemanticCache()
        cache.put("prompt", "response", domain="domain_a")
        hit = cache.get("prompt", domain="domain_a")
        self.assertIsNotNone(hit)

    def test_no_domain_matches_any(self):
        cache = SemanticCache()
        cache.put("prompt", "response", domain="")
        hit = cache.get("prompt", domain="")
        self.assertIsNotNone(hit)

    def test_invalidate_domain(self):
        cache = SemanticCache()
        cache.put("p1", "r1", domain="a")
        cache.put("p2", "r2", domain="b")
        cache.invalidate(domain="a")
        self.assertIsNone(cache.get("p1", domain="a"))
        self.assertIsNotNone(cache.get("p2", domain="b"))

    def test_invalidate_all(self):
        cache = SemanticCache()
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        cache.invalidate()
        self.assertEqual(cache.size, 0)


class TestCacheStats(unittest.TestCase):
    """Test metrics tracking."""

    def test_stats_after_operations(self):
        cache = SemanticCache()
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        cache.get("p1")  # hit
        cache.get("p3")  # miss

        s = cache.stats
        self.assertEqual(s.puts, 2)
        self.assertEqual(s.exact_hits, 1)
        self.assertEqual(s.misses, 1)
        self.assertEqual(s.total_gets, 2)

    def test_hit_rate(self):
        cache = SemanticCache()
        cache.put("p1", "r1")
        cache.get("p1")  # hit
        cache.get("p2")  # miss
        self.assertAlmostEqual(cache.stats.hit_rate, 0.5)

    def test_stats_to_dict(self):
        cache = SemanticCache()
        d = cache.stats.to_dict()
        self.assertIn("hit_rate", d)
        self.assertIn("total_gets", d)


class TestThreadSafety(unittest.TestCase):
    """Test concurrent access."""

    def test_concurrent_put_get(self):
        cache = SemanticCache(max_entries=100)
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache.put(f"prompt_{i}", f"response_{i}")
            except Exception as e:
                errors.append(str(e))

        def reader():
            try:
                for i in range(50):
                    cache.get(f"prompt_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=f) for f in [writer, reader, writer, reader]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
