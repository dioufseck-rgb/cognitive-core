"""
Cognitive Core — S-006: Secrets Management Tests

Tests the hybrid SecretStore: Key Vault (mocked) + env var fallback.
Verifies cache behavior, name mapping, thread safety, and that
secrets never appear in logs.
"""

import importlib.util
import logging
import os
import sys
import threading
import time
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_secrets_path = os.path.join(_base, "engine", "secrets.py")
_spec = importlib.util.spec_from_file_location("engine.secrets", _secrets_path)
_secrets_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.secrets"] = _secrets_mod
_spec.loader.exec_module(_secrets_mod)

SecretStore = _secrets_mod.SecretStore
get_secret = _secrets_mod.get_secret
reset_store = _secrets_mod.reset_store


class TestSecretStoreEnvVarFallback(unittest.TestCase):
    """Test env var fallback (no Key Vault)."""

    def setUp(self):
        self.store = SecretStore()  # No vault_url → env vars only

    def test_get_from_env(self):
        os.environ["TEST_SECRET_ABC"] = "secret_value_123"
        try:
            value = self.store.get("TEST_SECRET_ABC")
            self.assertEqual(value, "secret_value_123")
        finally:
            del os.environ["TEST_SECRET_ABC"]

    def test_missing_returns_default(self):
        value = self.store.get("NONEXISTENT_SECRET_XYZ", default="fallback")
        self.assertEqual(value, "fallback")

    def test_missing_returns_empty_string(self):
        value = self.store.get("NONEXISTENT_SECRET_XYZ")
        self.assertEqual(value, "")

    def test_vault_not_available_without_url(self):
        self.assertFalse(self.store.vault_available)

    def test_vault_check_returns_none_without_url(self):
        result = self.store._get_from_vault("ANY_SECRET")
        self.assertIsNone(result)


class TestSecretStoreCache(unittest.TestCase):
    """Test in-memory caching."""

    def test_value_cached_after_first_get(self):
        store = SecretStore(cache_ttl=60)
        os.environ["TEST_CACHE_SECRET"] = "cached_value"
        try:
            store.get("TEST_CACHE_SECRET")
            self.assertEqual(store.cache_size, 1)
        finally:
            del os.environ["TEST_CACHE_SECRET"]

    def test_cached_value_returned_without_env(self):
        store = SecretStore(cache_ttl=60)
        os.environ["TEST_CACHE_2"] = "original"
        try:
            store.get("TEST_CACHE_2")
        finally:
            del os.environ["TEST_CACHE_2"]

        # Env var gone, but cache should still return the value
        value = store.get("TEST_CACHE_2")
        self.assertEqual(value, "original")

    def test_cache_expires(self):
        store = SecretStore(cache_ttl=0)  # Instant expiry
        os.environ["TEST_TTL_SECRET"] = "first"
        try:
            store.get("TEST_TTL_SECRET")
            time.sleep(0.01)
            # Cache expired, should re-read from env
            os.environ["TEST_TTL_SECRET"] = "second"
            value = store.get("TEST_TTL_SECRET")
            self.assertEqual(value, "second")
        finally:
            del os.environ["TEST_TTL_SECRET"]

    def test_clear_cache(self):
        store = SecretStore(cache_ttl=3600)
        os.environ["TEST_CLEAR"] = "val"
        try:
            store.get("TEST_CLEAR")
            self.assertEqual(store.cache_size, 1)
            store.clear_cache()
            self.assertEqual(store.cache_size, 0)
        finally:
            del os.environ["TEST_CLEAR"]

    def test_cache_ttl_from_env(self):
        os.environ["CC_SECRETS_CACHE_TTL"] = "120"
        try:
            store = SecretStore()
            self.assertEqual(store._cache_ttl, 120)
        finally:
            del os.environ["CC_SECRETS_CACHE_TTL"]


class TestKVNameMapping(unittest.TestCase):
    """Test env var ↔ Key Vault name conversion."""

    def test_to_kv_name(self):
        self.assertEqual(SecretStore._to_kv_name("GOOGLE_API_KEY"), "google-api-key")
        self.assertEqual(SecretStore._to_kv_name("AZURE_OPENAI_API_KEY"), "azure-openai-api-key")
        self.assertEqual(SecretStore._to_kv_name("simple"), "simple")

    def test_from_kv_name(self):
        self.assertEqual(SecretStore._from_kv_name("google-api-key"), "GOOGLE_API_KEY")
        self.assertEqual(SecretStore._from_kv_name("azure-openai-api-key"), "AZURE_OPENAI_API_KEY")

    def test_roundtrip(self):
        original = "GOOGLE_API_KEY"
        kv = SecretStore._to_kv_name(original)
        back = SecretStore._from_kv_name(kv)
        self.assertEqual(back, original)


class TestMockKeyVault(unittest.TestCase):
    """Test Key Vault path with mock client."""

    def test_vault_value_wins_over_env(self):
        store = SecretStore(vault_url="https://myvault.vault.azure.net")

        # Mock the KV client
        class MockSecret:
            value = "from_vault"
        class MockClient:
            def get_secret(self, name):
                if name == "google-api-key":
                    return MockSecret()
                raise Exception("not found")

        store._kv_client = MockClient()
        store._kv_available = True

        os.environ["GOOGLE_API_KEY"] = "from_env"
        try:
            value = store.get("GOOGLE_API_KEY")
            self.assertEqual(value, "from_vault")
        finally:
            del os.environ["GOOGLE_API_KEY"]

    def test_vault_miss_falls_to_env(self):
        store = SecretStore(vault_url="https://myvault.vault.azure.net")

        class MockClient:
            def get_secret(self, name):
                raise Exception("not found")

        store._kv_client = MockClient()
        store._kv_available = True

        os.environ["OPENAI_API_KEY"] = "from_env_fallback"
        try:
            value = store.get("OPENAI_API_KEY")
            self.assertEqual(value, "from_env_fallback")
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_vault_failure_cached_as_unavailable(self):
        store = SecretStore(vault_url="https://broken.vault.azure.net")
        # _init_kv_client will fail (no azure SDK)
        store._get_from_vault("ANY_KEY")
        # Should mark vault as unavailable
        self.assertFalse(store.vault_available)
        # Subsequent calls should skip vault entirely
        result = store._get_from_vault("ANOTHER_KEY")
        self.assertIsNone(result)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_gets(self):
        store = SecretStore(cache_ttl=60)
        os.environ["THREAD_SAFE_SECRET"] = "safe_value"
        results = []
        errors = []

        def get_secret_thread():
            try:
                val = store.get("THREAD_SAFE_SECRET")
                results.append(val)
            except Exception as e:
                errors.append(str(e))

        try:
            threads = [threading.Thread(target=get_secret_thread) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(results), 20)
            self.assertEqual(len(errors), 0)
            self.assertTrue(all(v == "safe_value" for v in results))
        finally:
            del os.environ["THREAD_SAFE_SECRET"]


class TestNeverLogSecrets(unittest.TestCase):
    """Verify secrets don't appear in log output."""

    def test_log_output_clean(self):
        handler = logging.Handler()
        logged_messages = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(record.getMessage())

        logger = logging.getLogger("cognitive_core.secrets")
        capture = CaptureHandler()
        logger.addHandler(capture)
        logger.setLevel(logging.DEBUG)

        try:
            store = SecretStore()
            os.environ["SENSITIVE_KEY"] = "super_secret_password_123"
            try:
                store.get("SENSITIVE_KEY")
            finally:
                del os.environ["SENSITIVE_KEY"]

            # Check no log message contains the secret value
            for msg in logged_messages:
                self.assertNotIn("super_secret_password_123", msg,
                                "Secret value leaked into log output!")
        finally:
            logger.removeHandler(capture)


class TestModuleLevelConvenience(unittest.TestCase):
    def setUp(self):
        reset_store()

    def test_get_secret_function(self):
        os.environ["MODULE_TEST_SECRET"] = "mod_value"
        try:
            val = get_secret("MODULE_TEST_SECRET")
            self.assertEqual(val, "mod_value")
        finally:
            del os.environ["MODULE_TEST_SECRET"]

    def test_get_secret_default(self):
        val = get_secret("NONEXISTENT_MODULE_SECRET", default="safe")
        self.assertEqual(val, "safe")

    def test_reset_store(self):
        get_secret("ANYTHING")
        reset_store()
        # Should create a new store on next call
        val = get_secret("STILL_NOTHING", default="ok")
        self.assertEqual(val, "ok")


if __name__ == "__main__":
    unittest.main()
