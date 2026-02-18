"""Tests for S-004: Config Loader with three-tier hierarchy."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Direct import to bypass engine/__init__.py (needs langgraph/pydantic)
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_mod_path = os.path.join(_base, "engine", "config_loader.py")
_spec = importlib.util.spec_from_file_location("engine.config_loader", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.config_loader"] = _mod
_spec.loader.exec_module(_mod)

ConfigLoader = _mod.ConfigLoader
load_config = _mod.load_config
get_config = _mod.get_config
reset_config = _mod.reset_config
_deep_merge = _mod._deep_merge
_auto_convert = _mod._auto_convert
_load_env_overrides = _mod._load_env_overrides


class TestDeepMerge(unittest.TestCase):
    """Test recursive dict merging."""

    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = _deep_merge(base, overlay)
        self.assertEqual(result, {"a": 1, "b": 3, "c": 4})

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        overlay = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, overlay)
        self.assertEqual(result["a"], {"x": 1, "y": 99, "z": 100})
        self.assertEqual(result["b"], 3)

    def test_deep_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        overlay = {"a": {"b": {"d": 99, "e": 100}}}
        result = _deep_merge(base, overlay)
        self.assertEqual(result["a"]["b"], {"c": 1, "d": 99, "e": 100})

    def test_overlay_replaces_list(self):
        base = {"a": [1, 2, 3]}
        overlay = {"a": [4, 5]}
        result = _deep_merge(base, overlay)
        self.assertEqual(result["a"], [4, 5])

    def test_overlay_replaces_scalar_with_dict(self):
        base = {"a": "string_value"}
        overlay = {"a": {"nested": True}}
        result = _deep_merge(base, overlay)
        self.assertEqual(result["a"], {"nested": True})

    def test_empty_overlay(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        self.assertEqual(result, {"a": 1})

    def test_empty_base(self):
        overlay = {"a": 1}
        result = _deep_merge({}, overlay)
        self.assertEqual(result, {"a": 1})

    def test_immutability(self):
        """Merge should not modify originals."""
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        _deep_merge(base, overlay)
        self.assertEqual(base, {"a": {"x": 1}})
        self.assertEqual(overlay, {"a": {"y": 2}})


class TestAutoConvert(unittest.TestCase):
    """Test string-to-type conversion."""

    def test_true_values(self):
        for v in ["true", "True", "TRUE", "yes", "1"]:
            self.assertTrue(_auto_convert(v))

    def test_false_values(self):
        for v in ["false", "False", "FALSE", "no", "0"]:
            self.assertFalse(_auto_convert(v))

    def test_integer(self):
        self.assertEqual(_auto_convert("42"), 42)
        self.assertEqual(_auto_convert("-7"), -7)

    def test_float(self):
        self.assertAlmostEqual(_auto_convert("3.14"), 3.14)

    def test_string_passthrough(self):
        self.assertEqual(_auto_convert("hello"), "hello")
        self.assertEqual(_auto_convert("google"), "google")


class TestConfigLoaderBase(unittest.TestCase):
    """Test loading from base YAML files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_yaml(self, path, data):
        import yaml
        full = Path(self.tmpdir) / path
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w") as f:
            yaml.dump(data, f)

    def test_load_base_file(self):
        self._write_yaml("base.yaml", {
            "providers": {"google": {"model": "gemini-2.0-flash"}},
            "retry": {"max_attempts": 3},
        })
        loader = ConfigLoader(
            env="dev",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        self.assertEqual(loader.get("providers.google.model"), "gemini-2.0-flash")
        self.assertEqual(loader.get("retry.max_attempts"), 3)

    def test_missing_base_file_is_ok(self):
        loader = ConfigLoader(
            env="dev",
            project_root=self.tmpdir,
            base_files=["nonexistent.yaml"],
        )
        loader.load()  # Should not raise
        self.assertEqual(loader.get("anything", "default"), "default")

    def test_multiple_base_files_merge(self):
        self._write_yaml("a.yaml", {"shared": {"x": 1}, "only_a": True})
        self._write_yaml("b.yaml", {"shared": {"x": 2, "y": 3}, "only_b": True})
        loader = ConfigLoader(
            env="dev",
            project_root=self.tmpdir,
            base_files=["a.yaml", "b.yaml"],
        )
        loader.load()
        # b.yaml overrides a.yaml for shared.x
        self.assertEqual(loader.get("shared.x"), 2)
        self.assertEqual(loader.get("shared.y"), 3)
        self.assertTrue(loader.get("only_a"))
        self.assertTrue(loader.get("only_b"))


class TestConfigLoaderOverlay(unittest.TestCase):
    """Test environment overlay files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_yaml(self, path, data):
        import yaml
        full = Path(self.tmpdir) / path
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w") as f:
            yaml.dump(data, f)

    def test_overlay_merges_over_base(self):
        self._write_yaml("base.yaml", {
            "rate_limits": {"google": {"rpm": 60, "concurrent": 10}},
            "debug": False,
        })
        self._write_yaml("config/prod.yaml", {
            "rate_limits": {"google": {"rpm": 120}},
            "debug": False,
        })
        loader = ConfigLoader(
            env="prod",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        # prod overlay bumps rpm
        self.assertEqual(loader.get("rate_limits.google.rpm"), 120)
        # concurrent preserved from base
        self.assertEqual(loader.get("rate_limits.google.concurrent"), 10)

    def test_dev_overlay(self):
        self._write_yaml("base.yaml", {"mode": "default"})
        self._write_yaml("config/dev.yaml", {"mode": "development", "verbose": True})
        loader = ConfigLoader(
            env="dev",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        self.assertEqual(loader.get("mode"), "development")
        self.assertTrue(loader.get("verbose"))

    def test_no_overlay_file_is_ok(self):
        self._write_yaml("base.yaml", {"x": 1})
        loader = ConfigLoader(
            env="staging",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        self.assertEqual(loader.get("x"), 1)

    def test_sources_tracked(self):
        self._write_yaml("base.yaml", {"x": 1})
        self._write_yaml("config/prod.yaml", {"y": 2})
        loader = ConfigLoader(
            env="prod",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        self.assertIn("base:base.yaml", loader.sources)
        self.assertIn("overlay:config/prod.yaml", loader.sources)


class TestConfigLoaderEnvVars(unittest.TestCase):
    """Test environment variable overrides."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_yaml(self, path, data):
        import yaml
        full = Path(self.tmpdir) / path
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w") as f:
            yaml.dump(data, f)

    def test_known_mapping_override(self):
        self._write_yaml("base.yaml", {
            "rate_limits": {"google": {"requests_per_minute": 60}},
        })
        with patch.dict(os.environ, {"CC_RATE_LIMIT_GOOGLE_RPM": "200"}):
            loader = ConfigLoader(
                env="dev",
                project_root=self.tmpdir,
                base_files=["base.yaml"],
            )
            loader.load()
            self.assertEqual(
                loader.get("rate_limits.google.requests_per_minute"), 200
            )

    def test_arbitrary_config_override(self):
        self._write_yaml("base.yaml", {"custom": {"setting": "old"}})
        with patch.dict(os.environ, {"CC_CONFIG__custom__setting": "new"}):
            loader = ConfigLoader(
                env="dev",
                project_root=self.tmpdir,
                base_files=["base.yaml"],
            )
            loader.load()
            self.assertEqual(loader.get("custom.setting"), "new")

    def test_env_var_auto_converts_types(self):
        with patch.dict(os.environ, {
            "CC_PII_ENABLED": "true",
            "CC_RETRY_MAX_ATTEMPTS": "5",
        }):
            loader = ConfigLoader(
                env="dev",
                project_root=self.tmpdir,
                base_files=[],
            )
            loader.load()
            self.assertTrue(loader.get("pii.enabled"))
            self.assertEqual(loader.get("retry.max_attempts"), 5)

    def test_env_vars_override_overlay(self):
        """Env vars should beat overlay files."""
        self._write_yaml("base.yaml", {"retry": {"max_attempts": 3}})
        self._write_yaml("config/prod.yaml", {"retry": {"max_attempts": 5}})
        with patch.dict(os.environ, {"CC_RETRY_MAX_ATTEMPTS": "10"}):
            loader = ConfigLoader(
                env="prod",
                project_root=self.tmpdir,
                base_files=["base.yaml"],
            )
            loader.load()
            self.assertEqual(loader.get("retry.max_attempts"), 10)


class TestConfigLoaderAzure(unittest.TestCase):
    """Test Azure App Configuration fallback behavior."""

    def test_no_azure_endpoint_returns_empty(self):
        """Without AZURE_APP_CONFIG_ENDPOINT, Azure tier is skipped."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_APP_CONFIG_ENDPOINT", None)
            from engine.config_loader import _load_azure_app_config
            result = _load_azure_app_config("dev")
            self.assertEqual(result, {})

    def test_azure_sdk_missing_returns_empty(self):
        """If SDK not installed, gracefully returns empty dict."""
        with patch.dict(os.environ, {"AZURE_APP_CONFIG_ENDPOINT": "https://fake.azconfig.io"}):
            with patch.dict('sys.modules', {'azure.appconfiguration': None, 'azure.identity': None}):
                from engine.config_loader import _load_azure_app_config
                result = _load_azure_app_config("prod")
                self.assertEqual(result, {})


class TestConfigLoaderMeta(unittest.TestCase):
    """Test config metadata injection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_yaml(self, path, data):
        import yaml
        full = Path(self.tmpdir) / path
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w") as f:
            yaml.dump(data, f)

    def test_meta_injected(self):
        self._write_yaml("base.yaml", {"x": 1})
        loader = ConfigLoader(
            env="staging",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        meta = loader.get("_config_meta")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["env"], "staging")
        self.assertIn("base:base.yaml", meta["sources"])

    def test_get_default(self):
        loader = ConfigLoader(env="dev", project_root=self.tmpdir, base_files=[])
        loader.load()
        self.assertEqual(loader.get("nonexistent.path", "fallback"), "fallback")

    def test_reload(self):
        self._write_yaml("base.yaml", {"x": 1})
        loader = ConfigLoader(
            env="dev",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        self.assertEqual(loader.get("x"), 1)

        # Modify the file
        self._write_yaml("base.yaml", {"x": 99})
        loader.reload()
        self.assertEqual(loader.get("x"), 99)

    def test_get_all_returns_copy(self):
        self._write_yaml("base.yaml", {"a": {"b": 1}})
        loader = ConfigLoader(
            env="dev",
            project_root=self.tmpdir,
            base_files=["base.yaml"],
        )
        loader.load()
        data = loader.get_all()
        data["a"]["b"] = 999
        self.assertEqual(loader.get("a.b"), 1)  # Original unchanged


class TestSingleton(unittest.TestCase):
    """Test module-level singleton access."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_get_config_creates_singleton(self):
        with patch.dict(os.environ, {"CC_ENV": "test", "CC_PROJECT_ROOT": "/home/claude"}):
            config = get_config()
            self.assertIsNotNone(config)
            self.assertEqual(config.env, "test")

    def test_get_config_returns_same_instance(self):
        with patch.dict(os.environ, {"CC_ENV": "test", "CC_PROJECT_ROOT": "/home/claude"}):
            c1 = get_config()
            c2 = get_config()
            self.assertIs(c1, c2)

    def test_reset_clears_singleton(self):
        with patch.dict(os.environ, {"CC_ENV": "test", "CC_PROJECT_ROOT": "/home/claude"}):
            c1 = get_config()
            reset_config()
            c2 = get_config()
            self.assertIsNot(c1, c2)


class TestLoadConfigFactory(unittest.TestCase):
    """Test non-singleton factory function."""

    def test_creates_independent_instances(self):
        tmpdir = tempfile.mkdtemp()
        import yaml
        with open(Path(tmpdir) / "base.yaml", "w") as f:
            yaml.dump({"x": 1}, f)

        c1 = load_config(env="dev", project_root=tmpdir, base_files=["base.yaml"])
        c2 = load_config(env="dev", project_root=tmpdir, base_files=["base.yaml"])
        self.assertIsNot(c1, c2)


class TestRealProjectConfig(unittest.TestCase):
    """Test loading the actual project's llm_config.yaml."""

    def test_loads_llm_config(self):
        loader = ConfigLoader(
            env="dev",
            project_root="/home/claude",
            base_files=["llm_config.yaml"],
        )
        loader.load()
        # Should have retry section from llm_config.yaml
        retry = loader.get("retry")
        self.assertIsNotNone(retry)


if __name__ == "__main__":
    unittest.main()
