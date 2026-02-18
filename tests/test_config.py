"""
Cognitive Core — S-004: Environment Config Loader Tests

Tests three-tier config loading: overlay files → env vars → deep merge.
Azure App Config tested structurally (SDK not available in test env).
"""

import os
import sys
import tempfile
import shutil
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

import importlib.util

_config_path = os.path.join(_base, "engine", "config.py")
_spec = importlib.util.spec_from_file_location("engine.config", _config_path)
_config_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.config"] = _config_mod
_spec.loader.exec_module(_config_mod)

deep_merge = _config_mod.deep_merge
load_config = _config_mod.load_config
get_config_value = _config_mod.get_config_value
_load_overlay_file = _config_mod._load_overlay_file
_load_env_overrides = _config_mod._load_env_overrides
_set_nested = _config_mod._set_nested


class TestDeepMerge(unittest.TestCase):
    """Core merge logic."""

    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 99, "c": 3}
        result = deep_merge(base, overlay)
        self.assertEqual(result, {"a": 1, "b": 99, "c": 3})

    def test_nested_merge(self):
        base = {"outer": {"a": 1, "b": 2, "inner": {"x": 10}}}
        overlay = {"outer": {"b": 99, "inner": {"y": 20}}}
        result = deep_merge(base, overlay)
        self.assertEqual(result["outer"]["a"], 1)
        self.assertEqual(result["outer"]["b"], 99)
        self.assertEqual(result["outer"]["inner"]["x"], 10)
        self.assertEqual(result["outer"]["inner"]["y"], 20)

    def test_overlay_replaces_list(self):
        base = {"items": [1, 2, 3]}
        overlay = {"items": [99]}
        result = deep_merge(base, overlay)
        self.assertEqual(result["items"], [99])

    def test_overlay_replaces_scalar_with_dict(self):
        base = {"a": "string_value"}
        overlay = {"a": {"nested": True}}
        result = deep_merge(base, overlay)
        self.assertEqual(result["a"], {"nested": True})

    def test_base_unmodified(self):
        base = {"a": {"b": 1}}
        overlay = {"a": {"b": 99}}
        deep_merge(base, overlay)
        self.assertEqual(base["a"]["b"], 1)  # Original unchanged

    def test_empty_overlay(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        self.assertEqual(result, {"a": 1})

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        self.assertEqual(result, {"a": 1})

    def test_deeply_nested(self):
        base = {"l1": {"l2": {"l3": {"l4": "base"}}}}
        overlay = {"l1": {"l2": {"l3": {"l4": "overlay", "new": True}}}}
        result = deep_merge(base, overlay)
        self.assertEqual(result["l1"]["l2"]["l3"]["l4"], "overlay")
        self.assertTrue(result["l1"]["l2"]["l3"]["new"])


class TestSetNested(unittest.TestCase):
    def test_single_key(self):
        d = {}
        _set_nested(d, ["key"], "value")
        self.assertEqual(d["key"], "value")

    def test_nested_keys(self):
        d = {}
        _set_nested(d, ["a", "b", "c"], "deep")
        self.assertEqual(d["a"]["b"]["c"], "deep")

    def test_numeric_value_parsed(self):
        d = {}
        _set_nested(d, ["count"], "42")
        self.assertEqual(d["count"], 42)

    def test_boolean_value_parsed(self):
        d = {}
        _set_nested(d, ["enabled"], "true")
        self.assertTrue(d["enabled"])


class TestOverlayFiles(unittest.TestCase):
    """Test per-environment overlay file loading."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.tmpdir, "config")
        os.makedirs(self.config_dir)

        # Base config
        self.base_path = os.path.join(self.tmpdir, "llm_config.yaml")
        with open(self.base_path, "w") as f:
            f.write("providers:\n  google:\n    default_model: gemini-2.0-flash\n"
                    "rate_limits:\n  google:\n    requests_per_minute: 60\n")

        # Prod overlay
        with open(os.path.join(self.config_dir, "prod.yaml"), "w") as f:
            f.write("rate_limits:\n  google:\n    requests_per_minute: 200\n"
                    "  azure:\n    requests_per_minute: 300\n")

        # Staging overlay
        with open(os.path.join(self.config_dir, "staging.yaml"), "w") as f:
            f.write("rate_limits:\n  google:\n    requests_per_minute: 100\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_prod_overlay(self):
        overlay = _load_overlay_file(self.base_path, env="prod", config_dir=self.config_dir)
        self.assertEqual(overlay["rate_limits"]["google"]["requests_per_minute"], 200)
        self.assertEqual(overlay["rate_limits"]["azure"]["requests_per_minute"], 300)

    def test_load_staging_overlay(self):
        overlay = _load_overlay_file(self.base_path, env="staging", config_dir=self.config_dir)
        self.assertEqual(overlay["rate_limits"]["google"]["requests_per_minute"], 100)

    def test_nonexistent_env_returns_empty(self):
        overlay = _load_overlay_file(self.base_path, env="nonexistent", config_dir=self.config_dir)
        self.assertEqual(overlay, {})

    def test_empty_env_returns_empty(self):
        overlay = _load_overlay_file(self.base_path, env="", config_dir=self.config_dir)
        self.assertEqual(overlay, {})

    def test_merged_config_prod(self):
        """Prod overlay overrides base RPM but preserves base model."""
        cfg = load_config(
            base_path=self.base_path,
            env="prod",
            config_dir=self.config_dir,
            include_azure=False,
            include_env_vars=False,
        )
        # Overlay value wins
        self.assertEqual(cfg["rate_limits"]["google"]["requests_per_minute"], 200)
        # Base value preserved
        self.assertEqual(cfg["providers"]["google"]["default_model"], "gemini-2.0-flash")
        # New key from overlay
        self.assertEqual(cfg["rate_limits"]["azure"]["requests_per_minute"], 300)

    def test_active_env_stamped(self):
        cfg = load_config(
            base_path=self.base_path, env="prod",
            config_dir=self.config_dir,
            include_azure=False, include_env_vars=False,
        )
        self.assertEqual(cfg["_active_env"], "prod")


class TestEnvVarOverrides(unittest.TestCase):
    """Test CC_* environment variable overrides."""

    def setUp(self):
        # Clean any existing CC_ vars
        self._saved = {}
        for k in list(os.environ.keys()):
            if k.startswith("CC_") and k not in ("CC_ENV", "CC_CONFIG_DIR", "CC_WORKER_MODE", "CC_PROJECT_ROOT", "CC_VERSION"):
                self._saved[k] = os.environ.pop(k)

    def tearDown(self):
        # Restore
        for k in list(os.environ.keys()):
            if k.startswith("CC_") and k not in ("CC_ENV", "CC_CONFIG_DIR", "CC_WORKER_MODE", "CC_PROJECT_ROOT", "CC_VERSION"):
                del os.environ[k]
        os.environ.update(self._saved)

    def test_simple_override(self):
        os.environ["CC_TIMEOUT"] = "30"
        overrides = _load_env_overrides()
        self.assertEqual(overrides["timeout"], 30)

    def test_nested_override(self):
        os.environ["CC_RATE_LIMITS_GOOGLE_RPM"] = "120"
        overrides = _load_env_overrides()
        self.assertEqual(overrides["rate"]["limits"]["google"]["rpm"], 120)

    def test_excluded_vars(self):
        os.environ["CC_ENV"] = "prod"
        os.environ["CC_WORKER_MODE"] = "inline"
        overrides = _load_env_overrides()
        self.assertNotIn("env", overrides)
        self.assertNotIn("worker", overrides)
        # Cleanup
        del os.environ["CC_ENV"]
        del os.environ["CC_WORKER_MODE"]

    def test_no_cc_vars_returns_empty(self):
        overrides = _load_env_overrides()
        self.assertEqual(overrides, {})


class TestGetConfigValue(unittest.TestCase):
    """Test dotted-path config value retrieval."""

    def test_simple_path(self):
        cfg = {"a": 1}
        self.assertEqual(get_config_value("a", cfg), 1)

    def test_nested_path(self):
        cfg = {"a": {"b": {"c": 42}}}
        self.assertEqual(get_config_value("a.b.c", cfg), 42)

    def test_missing_path_returns_default(self):
        cfg = {"a": 1}
        self.assertEqual(get_config_value("x.y.z", cfg, default="fallback"), "fallback")

    def test_partial_path_returns_default(self):
        cfg = {"a": {"b": 1}}
        self.assertEqual(get_config_value("a.c", cfg, default=None), None)

    def test_default_is_none(self):
        cfg = {}
        self.assertIsNone(get_config_value("missing", cfg))

    def test_returns_dict_for_intermediate_path(self):
        cfg = {"a": {"b": {"c": 1, "d": 2}}}
        result = get_config_value("a.b", cfg)
        self.assertEqual(result, {"c": 1, "d": 2})


class TestFullLoadConfig(unittest.TestCase):
    """Integration test: base + overlay + env vars merged correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.tmpdir, "config")
        os.makedirs(self.config_dir)

        self.base_path = os.path.join(self.tmpdir, "base.yaml")
        with open(self.base_path, "w") as f:
            f.write(
                "providers:\n"
                "  google:\n"
                "    default_model: gemini-2.0-flash\n"
                "    timeout: 30\n"
                "rate_limits:\n"
                "  google:\n"
                "    requests_per_minute: 60\n"
                "    max_concurrent: 10\n"
            )

        with open(os.path.join(self.config_dir, "prod.yaml"), "w") as f:
            f.write(
                "rate_limits:\n"
                "  google:\n"
                "    requests_per_minute: 200\n"
            )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        for k in list(os.environ.keys()):
            if k.startswith("CC_") and k not in ("CC_ENV", "CC_CONFIG_DIR", "CC_WORKER_MODE", "CC_PROJECT_ROOT", "CC_VERSION"):
                del os.environ[k]

    def test_base_only(self):
        cfg = load_config(self.base_path, include_azure=False, include_env_vars=False)
        self.assertEqual(cfg["rate_limits"]["google"]["requests_per_minute"], 60)

    def test_base_plus_overlay(self):
        cfg = load_config(
            self.base_path, env="prod", config_dir=self.config_dir,
            include_azure=False, include_env_vars=False,
        )
        # Overlay overrides RPM
        self.assertEqual(cfg["rate_limits"]["google"]["requests_per_minute"], 200)
        # Base preserves max_concurrent (not in overlay)
        self.assertEqual(cfg["rate_limits"]["google"]["max_concurrent"], 10)
        # Base preserves provider config
        self.assertEqual(cfg["providers"]["google"]["default_model"], "gemini-2.0-flash")

    def test_missing_base_file(self):
        cfg = load_config(
            "/nonexistent/path.yaml",
            include_azure=False, include_env_vars=False,
        )
        # Should return minimal config (just metadata)
        self.assertIn("_active_env", cfg)

    def test_config_source_stamped(self):
        cfg = load_config(self.base_path, include_azure=False, include_env_vars=False)
        self.assertEqual(cfg["_config_source"], self.base_path)

    def test_azure_skipped_gracefully(self):
        """Azure App Config tier returns empty when not configured."""
        cfg = load_config(
            self.base_path, include_azure=True, include_env_vars=False,
        )
        # Should still work — Azure returns empty, base loads
        self.assertEqual(cfg["providers"]["google"]["default_model"], "gemini-2.0-flash")


class TestLoadConfigWithActualLlmConfig(unittest.TestCase):
    """Test against the real llm_config.yaml in the project."""

    def test_loads_real_config(self):
        real_path = os.path.join(_base, "llm_config.yaml")
        if not os.path.exists(real_path):
            self.skipTest("llm_config.yaml not found")

        cfg = load_config(real_path, include_azure=False, include_env_vars=False)
        # Should have aliases section (provider model aliases)
        self.assertIn("aliases", cfg)
        # Should have rate_limits section
        self.assertIn("rate_limits", cfg)
        # Should have pricing section
        self.assertIn("pricing", cfg)

    def test_get_value_from_real_config(self):
        real_path = os.path.join(_base, "llm_config.yaml")
        if not os.path.exists(real_path):
            self.skipTest("llm_config.yaml not found")

        cfg = load_config(real_path, include_azure=False, include_env_vars=False)
        rpm = get_config_value("rate_limits.google.requests_per_minute", cfg)
        self.assertIsNotNone(rpm)
        self.assertIsInstance(rpm, int)
        self.assertGreater(rpm, 0)


if __name__ == "__main__":
    unittest.main()
