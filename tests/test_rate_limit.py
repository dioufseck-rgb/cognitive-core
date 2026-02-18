"""
Cognitive Core — P-005: Rate Limiting Tests
"""

import os
import sys
import time
import threading
import unittest

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_rl_path = os.path.join(_base, "engine", "rate_limit.py")
_spec = importlib.util.spec_from_file_location("engine.rate_limit", _rl_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.rate_limit"] = _mod
_spec.loader.exec_module(_mod)

ProviderRateLimiter = _mod.ProviderRateLimiter
RateLimitConfig = _mod.RateLimitConfig
BackpressureError = _mod.BackpressureError
TokenBucket = _mod.TokenBucket
get_rate_limiter = _mod.get_rate_limiter
reset_all_limiters = _mod.reset_all_limiters


class TestConcurrentLimitEnforced(unittest.TestCase):
    def test_concurrent_count_matches_limit(self):
        config = RateLimitConfig(max_concurrent=3, requests_per_minute=10000)
        limiter = ProviderRateLimiter(config)
        max_concurrent = [0]
        current = [0]
        lock = threading.Lock()

        def worker():
            with limiter.acquire(timeout=10):
                with lock:
                    current[0] += 1
                    max_concurrent[0] = max(max_concurrent[0], current[0])
                time.sleep(0.05)
                with lock:
                    current[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        self.assertLessEqual(max_concurrent[0], 3)


class TestRateLimitThrottling(unittest.TestCase):
    def test_token_bucket_limits_burst(self):
        bucket = TokenBucket(rate_per_minute=600)
        acquired = sum(1 for _ in range(700) if bucket.try_acquire())
        self.assertEqual(acquired, 600)
        self.assertFalse(bucket.try_acquire())

    def test_token_bucket_wait_succeeds(self):
        bucket = TokenBucket(rate_per_minute=6000)
        for _ in range(6000):
            bucket.try_acquire()
        self.assertTrue(bucket.wait_for_token(timeout=1.0))


class TestQueueBackpressure(unittest.TestCase):
    def test_timeout_raises_backpressure(self):
        config = RateLimitConfig(max_concurrent=1, requests_per_minute=10000)
        limiter = ProviderRateLimiter(config)
        slot_held = threading.Event()
        release = threading.Event()

        def holder():
            with limiter.acquire(timeout=10):
                slot_held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        slot_held.wait(timeout=2)

        with self.assertRaises(BackpressureError):
            with limiter.acquire(timeout=0.1):
                pass

        release.set()
        t.join(timeout=5)

    def test_rejection_counted(self):
        config = RateLimitConfig(max_concurrent=1, requests_per_minute=10000)
        limiter = ProviderRateLimiter(config)
        slot_held = threading.Event()
        release = threading.Event()

        def holder():
            with limiter.acquire(timeout=10):
                slot_held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        slot_held.wait(timeout=2)

        try:
            with limiter.acquire(timeout=0.1):
                pass
        except BackpressureError:
            pass

        self.assertEqual(limiter.metrics["total_rejections"], 1)
        release.set()
        t.join(timeout=5)


class TestPerProviderConfig(unittest.TestCase):
    def setUp(self):
        reset_all_limiters()

    def test_different_providers(self):
        g = ProviderRateLimiter(RateLimitConfig(max_concurrent=10, requests_per_minute=60))
        a = ProviderRateLimiter(RateLimitConfig(max_concurrent=20, requests_per_minute=100))
        self.assertEqual(g.config.max_concurrent, 10)
        self.assertEqual(a.config.max_concurrent, 20)

    def test_config_from_yaml(self):
        google = get_rate_limiter("google")
        self.assertEqual(google.config.max_concurrent, 10)
        self.assertEqual(google.config.requests_per_minute, 60)
        azure = get_rate_limiter("azure")
        self.assertEqual(azure.config.max_concurrent, 20)
        reset_all_limiters()


class TestMetricsExposed(unittest.TestCase):
    def test_metrics_after_requests(self):
        config = RateLimitConfig(max_concurrent=10, requests_per_minute=10000)
        limiter = ProviderRateLimiter(config)
        for _ in range(5):
            with limiter.acquire(timeout=5):
                pass
        m = limiter.metrics
        self.assertEqual(m["total_requests"], 5)
        self.assertEqual(m["current_active"], 0)
        self.assertIn("peak_active", m)
        self.assertIn("avg_wait_ms", m)

    def test_peak_active_tracked(self):
        config = RateLimitConfig(max_concurrent=10, requests_per_minute=10000)
        limiter = ProviderRateLimiter(config)
        barrier = threading.Barrier(3, timeout=5)

        def worker():
            with limiter.acquire(timeout=5):
                barrier.wait()
                time.sleep(0.05)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertGreaterEqual(limiter.metrics["peak_active"], 2)


class TestNormalOperation(unittest.TestCase):
    def test_single_call_immediate(self):
        limiter = ProviderRateLimiter(RateLimitConfig(max_concurrent=10, requests_per_minute=10000))
        t0 = time.time()
        with limiter.acquire(timeout=5):
            pass
        self.assertLess(time.time() - t0, 0.1)


class TestSemaphoreReleaseOnException(unittest.TestCase):
    def test_release_on_error(self):
        limiter = ProviderRateLimiter(RateLimitConfig(max_concurrent=1, requests_per_minute=10000))
        try:
            with limiter.acquire(timeout=5):
                raise ValueError("fail")
        except ValueError:
            pass
        t0 = time.time()
        with limiter.acquire(timeout=1):
            pass
        self.assertLess(time.time() - t0, 0.1)


class TestTokenBucketRefill(unittest.TestCase):
    def test_tokens_refill(self):
        bucket = TokenBucket(rate_per_minute=6000)
        for _ in range(100):
            bucket.try_acquire()
        time.sleep(0.05)
        self.assertGreater(bucket.available_tokens, 0)

    def test_tokens_capped(self):
        bucket = TokenBucket(rate_per_minute=60)
        time.sleep(0.1)
        self.assertLessEqual(bucket.available_tokens, 60.1)


if __name__ == "__main__":
    unittest.main()
