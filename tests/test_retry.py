"""
Cognitive Core — P-001: LLM Retry with Backoff & Fallback Tests

Tests:
  - Retry on parse failure
  - Retry on timeout
  - Fallback on primary exhausted
  - Circuit breaker stops cascade
  - No retry on auth error
  - Retry count logged
  - Backoff timing
  - Non-retryable errors propagate immediately
  - Parse failure on last attempt returns raw
  - Fallback also fails → original error raised
  - Circuit breaker resets after timeout
  - Circuit breaker half-open probe
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Import retry module directly to avoid engine/__init__.py langgraph dependency
import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_retry_path = os.path.join(_base, "engine", "retry.py")
_spec = importlib.util.spec_from_file_location("engine.retry", _retry_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.retry"] = _mod
_spec.loader.exec_module(_mod)

RetryPolicy = _mod.RetryPolicy
RetryResult = _mod.RetryResult
CircuitBreaker = _mod.CircuitBreaker
CircuitBreakerOpen = _mod.CircuitBreakerOpen
invoke_with_retry = _mod.invoke_with_retry
get_retry_policy = _mod.get_retry_policy
get_circuit_breaker = _mod.get_circuit_breaker
reset_all_circuit_breakers = _mod.reset_all_circuit_breakers
_is_retryable = _mod._is_retryable
_calculate_backoff = _mod._calculate_backoff
DEFAULT_POLICY = _mod.DEFAULT_POLICY


class MockResponse:
    """Mock LLM response."""
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """Mock LLM with configurable behavior per call."""
    def __init__(self, responses=None, errors=None):
        """
        responses: list of strings to return in order
        errors: list of exceptions to raise (None = use response)
        """
        self.responses = responses or []
        self.errors = errors or []
        self.call_count = 0
        self.calls = []

    def invoke(self, messages):
        idx = self.call_count
        self.call_count += 1
        self.calls.append(messages)

        if idx < len(self.errors) and self.errors[idx] is not None:
            raise self.errors[idx]

        if idx < len(self.responses):
            return MockResponse(self.responses[idx])

        return MockResponse('{"result": "default"}')


class TestRetryOnParseFailure(unittest.TestCase):
    """test_retry_on_parse_failure — mock LLM returns garbage once, valid JSON on retry → step succeeds"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_retry_on_parse_failure(self):
        llm = MockLLM(responses=["not valid json", '{"category": "defective"}'])

        parse_calls = []
        def parse_fn(raw):
            parse_calls.append(raw)
            import json
            return json.loads(raw)  # will fail on "not valid json"

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="classify",
            parse_fn=parse_fn,
            sleep_fn=lambda x: None,  # skip actual sleep
        )

        self.assertEqual(result.content, '{"category": "defective"}')
        self.assertEqual(result.attempts, 2)
        self.assertFalse(result.used_fallback)
        self.assertEqual(llm.call_count, 2)
        # First parse failed, second succeeded
        self.assertEqual(len(parse_calls), 2)


class TestRetryOnTimeout(unittest.TestCase):
    """test_retry_on_timeout — mock raises TimeoutError twice, succeeds on third → step succeeds"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_retry_on_timeout(self):
        llm = MockLLM(
            responses=[None, None, '{"result": "ok"}'],
            errors=[TimeoutError("timed out"), TimeoutError("timed out"), None],
        )

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="investigate",
            sleep_fn=lambda x: None,
        )

        self.assertEqual(result.content, '{"result": "ok"}')
        self.assertEqual(result.attempts, 3)
        self.assertFalse(result.used_fallback)
        self.assertEqual(llm.call_count, 3)


class TestFallbackOnPrimaryExhausted(unittest.TestCase):
    """test_fallback_on_primary_exhausted — primary fails 3x, fallback succeeds"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_fallback_on_primary_exhausted(self):
        primary = MockLLM(errors=[
            TimeoutError("timeout1"),
            TimeoutError("timeout2"),
            TimeoutError("timeout3"),
        ])
        fallback = MockLLM(responses=['{"result": "from_fallback"}'])

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0, fallback_model="gemini-2.5-pro")
        result = invoke_with_retry(
            primary, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="classify",
            model_name="gemini-2.0-flash",
            fallback_llm=fallback,
            sleep_fn=lambda x: None,
        )

        self.assertEqual(result.content, '{"result": "from_fallback"}')
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.model_used, "gemini-2.5-pro")
        self.assertEqual(primary.call_count, 3)
        self.assertEqual(fallback.call_count, 1)
        # Verify fallback logged in attempt_log
        fallback_entries = [a for a in result.attempt_log if a["is_fallback"]]
        self.assertEqual(len(fallback_entries), 1)
        self.assertEqual(fallback_entries[0]["status"], "success")


class TestCircuitBreakerStopsCascade(unittest.TestCase):
    """test_circuit_breaker_stops_cascade — 5 consecutive failures → raises CircuitBreakerOpen"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_circuit_breaker_opens(self):
        failing_llm = MockLLM(errors=[
            TimeoutError("fail") for _ in range(20)
        ])

        policy = RetryPolicy(
            max_attempts=1,  # fail fast per invocation
            backoff_base=0.0,
            circuit_breaker_threshold=5,
        )

        # First 5 calls should fail normally (opening the circuit)
        for i in range(5):
            with self.assertRaises(TimeoutError):
                invoke_with_retry(
                    failing_llm, [{"role": "user", "content": "test"}],
                    policy=policy,
                    provider="test_cb",
                    step_name=f"step_{i}",
                    sleep_fn=lambda x: None,
                )

        # 6th call should hit circuit breaker
        with self.assertRaises(CircuitBreakerOpen):
            invoke_with_retry(
                failing_llm, [{"role": "user", "content": "test"}],
                policy=policy,
                provider="test_cb",
                step_name="step_blocked",
                sleep_fn=lambda x: None,
            )

        # LLM should NOT have been called for the 6th invocation
        # 5 calls × 1 attempt each = 5 LLM calls
        self.assertEqual(failing_llm.call_count, 5)


class TestNoRetryOnAuthError(unittest.TestCase):
    """test_no_retry_on_auth_error — 401/403 fails immediately without retry"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_no_retry_on_401(self):
        llm = MockLLM(errors=[
            Exception("HTTP 401 Unauthorized: Invalid API key"),
        ])

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0)
        with self.assertRaises(Exception) as ctx:
            invoke_with_retry(
                llm, [{"role": "user", "content": "test"}],
                policy=policy,
                step_name="classify",
                sleep_fn=lambda x: None,
            )

        self.assertIn("401", str(ctx.exception))
        self.assertEqual(llm.call_count, 1)  # No retry

    def test_no_retry_on_403(self):
        llm = MockLLM(errors=[
            Exception("HTTP 403 Forbidden"),
        ])

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0)
        with self.assertRaises(Exception):
            invoke_with_retry(
                llm, [{"role": "user", "content": "test"}],
                policy=policy,
                step_name="classify",
                sleep_fn=lambda x: None,
            )

        self.assertEqual(llm.call_count, 1)


class TestRetryCountLogged(unittest.TestCase):
    """test_retry_count_logged — each attempt produces structured log entry"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_attempt_log_entries(self):
        llm = MockLLM(
            responses=[None, None, '{"ok": true}'],
            errors=[TimeoutError("t1"), TimeoutError("t2"), None],
        )

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="investigate",
            model_name="gemini-2.0-flash",
            sleep_fn=lambda x: None,
        )

        # 3 attempts logged
        self.assertEqual(len(result.attempt_log), 3)

        # First two are retryable errors
        self.assertEqual(result.attempt_log[0]["status"], "retryable_error")
        self.assertEqual(result.attempt_log[0]["attempt"], 1)
        self.assertEqual(result.attempt_log[0]["model"], "gemini-2.0-flash")
        self.assertIn("error", result.attempt_log[0])
        self.assertIn("latency_s", result.attempt_log[0])

        self.assertEqual(result.attempt_log[1]["status"], "retryable_error")
        self.assertEqual(result.attempt_log[1]["attempt"], 2)

        # Third is success
        self.assertEqual(result.attempt_log[2]["status"], "success")
        self.assertEqual(result.attempt_log[2]["attempt"], 3)


class TestBackoffTiming(unittest.TestCase):
    """test_backoff_timing — verify delay between retries follows exponential curve"""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_backoff_increases_exponentially(self):
        policy = RetryPolicy(backoff_base=1.0, backoff_max=30.0, jitter=0.0)

        d0 = _calculate_backoff(0, policy)  # 1 * 2^0 = 1.0
        d1 = _calculate_backoff(1, policy)  # 1 * 2^1 = 2.0
        d2 = _calculate_backoff(2, policy)  # 1 * 2^2 = 4.0
        d3 = _calculate_backoff(3, policy)  # 1 * 2^3 = 8.0

        self.assertAlmostEqual(d0, 1.0, places=1)
        self.assertAlmostEqual(d1, 2.0, places=1)
        self.assertAlmostEqual(d2, 4.0, places=1)
        self.assertAlmostEqual(d3, 8.0, places=1)

    def test_backoff_capped_at_max(self):
        policy = RetryPolicy(backoff_base=1.0, backoff_max=5.0, jitter=0.0)

        d5 = _calculate_backoff(5, policy)  # 1 * 2^5 = 32.0 → capped at 5.0
        self.assertAlmostEqual(d5, 5.0, places=1)

    def test_backoff_with_jitter(self):
        policy = RetryPolicy(backoff_base=1.0, backoff_max=30.0, jitter=0.2)

        # With 20% jitter on base 1.0 (attempt 0), range is [0.8, 1.2]
        results = [_calculate_backoff(0, policy) for _ in range(100)]
        self.assertTrue(all(0.7 <= r <= 1.3 for r in results))  # ±30% tolerance for randomness
        # Should not all be the same (jitter is working)
        self.assertTrue(len(set(round(r, 4) for r in results)) > 1)

    def test_actual_sleep_called_with_backoff(self):
        llm = MockLLM(
            responses=[None, '{"ok": true}'],
            errors=[TimeoutError("timeout"), None],
        )

        sleep_delays = []
        def mock_sleep(delay):
            sleep_delays.append(delay)

        policy = RetryPolicy(max_attempts=2, backoff_base=1.0, jitter=0.0)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="test",
            sleep_fn=mock_sleep,
        )

        # One retry → one sleep call
        self.assertEqual(len(sleep_delays), 1)
        self.assertAlmostEqual(sleep_delays[0], 1.0, places=1)  # 1.0 * 2^0 = 1.0


class TestParseFailureLastAttempt(unittest.TestCase):
    """Parse failure on final attempt returns raw response (doesn't raise)."""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_parse_failure_on_last_attempt_returns_raw(self):
        llm = MockLLM(responses=["garbage1", "garbage2", "garbage3"])

        def bad_parse(raw):
            raise ValueError("cannot parse")

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0, retry_on_parse_failure=True)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="test",
            parse_fn=bad_parse,
            sleep_fn=lambda x: None,
        )

        # Returns the raw response from last attempt
        self.assertEqual(result.content, "garbage3")
        self.assertEqual(result.attempts, 3)
        # Last entry should be parse_failure_final
        self.assertEqual(result.attempt_log[-1]["status"], "parse_failure_final")


class TestFallbackAlsoFails(unittest.TestCase):
    """When both primary and fallback fail, raise the last error."""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_both_fail_raises_last_error(self):
        primary = MockLLM(errors=[TimeoutError("primary_fail")])
        fallback = MockLLM(errors=[TimeoutError("fallback_fail")])

        policy = RetryPolicy(max_attempts=1, backoff_base=0.0)
        with self.assertRaises(TimeoutError) as ctx:
            invoke_with_retry(
                primary, [{"role": "user", "content": "test"}],
                policy=policy,
                fallback_llm=fallback,
                sleep_fn=lambda x: None,
            )

        self.assertIn("fallback_fail", str(ctx.exception))
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.call_count, 1)


class TestCircuitBreakerReset(unittest.TestCase):
    """Circuit breaker resets after timeout period."""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_circuit_breaker_resets(self):
        cb = CircuitBreaker(threshold=2, reset_seconds=0.1)

        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, "open")

        # Wait for reset
        time.sleep(0.15)
        self.assertEqual(cb.state, "half_open")

        # Successful probe closes the circuit
        cb.record_success()
        self.assertEqual(cb.state, "closed")

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(threshold=2, reset_seconds=0.1)

        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, "open")

        time.sleep(0.15)
        self.assertEqual(cb.state, "half_open")

        # Probe fails → back to open
        cb.record_failure()
        self.assertEqual(cb.state, "open")


class TestIsRetryable(unittest.TestCase):
    """Verify retryable vs non-retryable error classification."""

    def test_timeout_is_retryable(self):
        self.assertTrue(_is_retryable(TimeoutError("timed out"), DEFAULT_POLICY))

    def test_connection_error_is_retryable(self):
        self.assertTrue(_is_retryable(ConnectionError("refused"), DEFAULT_POLICY))

    def test_rate_limit_is_retryable(self):
        self.assertTrue(_is_retryable(Exception("HTTP 429 Too Many Requests"), DEFAULT_POLICY))

    def test_server_error_is_retryable(self):
        self.assertTrue(_is_retryable(Exception("HTTP 500 Internal Server Error"), DEFAULT_POLICY))
        self.assertTrue(_is_retryable(Exception("HTTP 503 Service Unavailable"), DEFAULT_POLICY))

    def test_auth_error_not_retryable(self):
        self.assertFalse(_is_retryable(Exception("HTTP 401 Unauthorized"), DEFAULT_POLICY))
        self.assertFalse(_is_retryable(Exception("HTTP 403 Forbidden"), DEFAULT_POLICY))

    def test_generic_error_not_retryable(self):
        self.assertFalse(_is_retryable(ValueError("bad value"), DEFAULT_POLICY))

    def test_unavailable_is_retryable(self):
        self.assertTrue(_is_retryable(Exception("service unavailable"), DEFAULT_POLICY))


class TestFirstTrySuccess(unittest.TestCase):
    """When LLM succeeds on first try, no retries happen."""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_first_try_success(self):
        llm = MockLLM(responses=['{"result": "ok"}'])

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            step_name="test",
            sleep_fn=lambda x: None,
        )

        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.content, '{"result": "ok"}')
        self.assertFalse(result.used_fallback)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(len(result.attempt_log), 1)
        self.assertEqual(result.attempt_log[0]["status"], "success")


class TestRetryPolicyConfig(unittest.TestCase):
    """Test retry policy loading from config."""

    def test_default_policy_values(self):
        p = DEFAULT_POLICY
        self.assertEqual(p.max_attempts, 3)
        self.assertEqual(p.backoff_base, 1.0)
        self.assertEqual(p.backoff_max, 30.0)
        self.assertAlmostEqual(p.jitter, 0.2)
        self.assertIsNone(p.fallback_model)
        self.assertEqual(p.circuit_breaker_threshold, 5)
        self.assertTrue(p.retry_on_parse_failure)

    def test_custom_policy(self):
        p = RetryPolicy(max_attempts=5, backoff_base=0.5, fallback_model="gpt-4o")
        self.assertEqual(p.max_attempts, 5)
        self.assertEqual(p.backoff_base, 0.5)
        self.assertEqual(p.fallback_model, "gpt-4o")

    def test_load_google_policy_from_config(self):
        """Verify actual llm_config.yaml loads correct google retry config."""
        policy = get_retry_policy("google")
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.backoff_base, 1.0)
        self.assertEqual(policy.fallback_model, "gemini-2.5-pro")

    def test_load_azure_policy_from_config(self):
        """Verify actual llm_config.yaml loads correct azure retry config."""
        policy = get_retry_policy("azure")
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.backoff_base, 1.5)
        self.assertEqual(policy.fallback_model, "gpt-4o")

    def test_unknown_provider_gets_default(self):
        """Unknown provider should get the default retry config."""
        policy = get_retry_policy("unknown_provider_xyz")
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.backoff_base, 1.0)
        self.assertTrue(policy.retry_on_parse_failure)


class TestRetryDisabled(unittest.TestCase):
    """With max_attempts=1, failures go straight through."""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_single_attempt_failure(self):
        llm = MockLLM(errors=[TimeoutError("fail")])

        policy = RetryPolicy(max_attempts=1, backoff_base=0.0)
        with self.assertRaises(TimeoutError):
            invoke_with_retry(
                llm, [{"role": "user", "content": "test"}],
                policy=policy,
                sleep_fn=lambda x: None,
            )

        self.assertEqual(llm.call_count, 1)


class TestParseRetryDisabled(unittest.TestCase):
    """With retry_on_parse_failure=False, parse failures return raw immediately."""

    def setUp(self):
        reset_all_circuit_breakers()

    def test_no_parse_retry(self):
        llm = MockLLM(responses=["garbage"])

        def bad_parse(raw):
            raise ValueError("cannot parse")

        policy = RetryPolicy(max_attempts=3, backoff_base=0.0, retry_on_parse_failure=False)
        result = invoke_with_retry(
            llm, [{"role": "user", "content": "test"}],
            policy=policy,
            parse_fn=bad_parse,
            sleep_fn=lambda x: None,
        )

        self.assertEqual(result.content, "garbage")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(llm.call_count, 1)


if __name__ == "__main__":
    unittest.main()
