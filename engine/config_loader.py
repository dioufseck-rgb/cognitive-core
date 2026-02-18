"""
Cognitive Core — Configuration Loader (S-004)

Three-tier config loading with environment profile support:

  Tier 1: Azure App Configuration (if AZURE_APP_CONFIG_ENDPOINT set + SDK available)
  Tier 2: Per-environment overlay files (config/{env}.yaml merged over base)
  Tier 3: Environment variable overrides (CC_* prefix)

Active environment is set via CC_ENV (default: "dev").
Config is loaded once and cached for the process lifetime.

Usage:
    from engine.config_loader import load_config, get_config

    # Load explicitly
    config = load_config(env="prod", project_root=".")

    # Or use cached singleton
    config = get_config()

    # Access values
    rate_limit = config.get("rate_limits.google.rpm", 60)
    tier = config.get("governance.default_tier", "gate")
"""

from __future__ import annotations

import copy
import logging
import os
import re
import yaml
from pathlib import Path
from typing import Any

logger = logging.getLogger("cognitive_core.config")


class ConfigLoader:
    """
    Hierarchical config loader with deep merge.

    Merge order (later overrides earlier):
      1. Base YAML (llm_config.yaml, coordinator/config.yaml)
      2. Environment overlay (config/{env}.yaml)
      3. Azure App Configuration values (if available)
      4. Environment variable overrides (CC_* prefix)
    """

    def __init__(
        self,
        env: str = "dev",
        project_root: str = ".",
        base_files: list[str] | None = None,
    ):
        self.env = env
        self.project_root = Path(project_root)
        self.base_files = base_files or ["llm_config.yaml"]
        self._data: dict[str, Any] = {}
        self._source_log: list[str] = []
        self._loaded = False

    def load(self) -> dict[str, Any]:
        """Load and merge all config tiers. Returns merged dict."""
        self._data = {}
        self._source_log = []

        # Tier 0: Base YAML files
        for base_file in self.base_files:
            base_path = self.project_root / base_file
            if base_path.exists():
                with open(base_path) as f:
                    base_data = yaml.safe_load(f) or {}
                self._data = _deep_merge(self._data, base_data)
                self._source_log.append(f"base:{base_file}")

        # Tier 2: Environment overlay
        overlay_path = self.project_root / "config" / f"{self.env}.yaml"
        if overlay_path.exists():
            with open(overlay_path) as f:
                overlay_data = yaml.safe_load(f) or {}
            self._data = _deep_merge(self._data, overlay_data)
            self._source_log.append(f"overlay:config/{self.env}.yaml")

        # Tier 1: Azure App Configuration
        azure_overrides = _load_azure_app_config(self.env)
        if azure_overrides:
            self._data = _deep_merge(self._data, azure_overrides)
            self._source_log.append("azure_app_config")

        # Tier 3: Environment variable overrides
        env_overrides = _load_env_overrides()
        if env_overrides:
            self._data = _deep_merge(self._data, env_overrides)
            self._source_log.append(f"env_vars({len(env_overrides)} keys)")

        # Inject active profile metadata
        self._data["_config_meta"] = {
            "env": self.env,
            "sources": self._source_log,
            "project_root": str(self.project_root),
        }

        self._loaded = True
        logger.info(
            "Config loaded: env=%s sources=%s",
            self.env, self._source_log,
        )
        return self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Get a value by dotted key path.

        Example: config.get("rate_limits.google.rpm", 60)
        """
        if not self._loaded:
            self.load()

        keys = dotted_key.split(".")
        current = self._data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_all(self) -> dict[str, Any]:
        """Return the full merged config dict."""
        if not self._loaded:
            self.load()
        return copy.deepcopy(self._data)

    @property
    def sources(self) -> list[str]:
        """Which config sources were loaded."""
        return list(self._source_log)

    def reload(self) -> dict[str, Any]:
        """Force reload from all tiers."""
        self._loaded = False
        return self.load()


# ═══════════════════════════════════════════════════════════════════
# Deep Merge
# ═══════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, overlay: dict) -> dict:
    """
    Deep merge overlay into base. Overlay values win.
    Dicts are merged recursively. Lists and scalars are replaced.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ═══════════════════════════════════════════════════════════════════
# Tier 1: Azure App Configuration
# ═══════════════════════════════════════════════════════════════════

def _load_azure_app_config(env: str) -> dict[str, Any]:
    """
    Load config from Azure App Configuration if available.
    Requires: AZURE_APP_CONFIG_ENDPOINT env var + azure-appconfiguration SDK.
    Returns empty dict if not configured or SDK not available.
    """
    endpoint = os.environ.get("AZURE_APP_CONFIG_ENDPOINT")
    if not endpoint:
        return {}

    try:
        from azure.appconfiguration import AzureAppConfigurationClient
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        client = AzureAppConfigurationClient(endpoint, credential)

        # Load settings with label matching the environment
        result = {}
        settings = client.list_configuration_settings(label_filter=env)
        for setting in settings:
            # Convert dotted keys to nested dict
            keys = setting.key.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            # Auto-convert numeric strings
            current[keys[-1]] = _auto_convert(setting.value)

        logger.info("Loaded %d settings from Azure App Configuration (label=%s)", len(result), env)
        return result

    except ImportError:
        logger.debug("azure-appconfiguration not installed — skipping Azure App Config")
        return {}
    except Exception as e:
        logger.warning("Failed to load Azure App Configuration: %s", e)
        return {}


# ═══════════════════════════════════════════════════════════════════
# Tier 3: Environment Variable Overrides
# ═══════════════════════════════════════════════════════════════════

# Mapping from CC_* env vars to config dotted paths
_ENV_MAPPINGS: dict[str, str] = {
    "CC_RATE_LIMIT_GOOGLE_RPM": "rate_limits.google.requests_per_minute",
    "CC_RATE_LIMIT_GOOGLE_CONCURRENT": "rate_limits.google.max_concurrent",
    "CC_RATE_LIMIT_AZURE_RPM": "rate_limits.azure.requests_per_minute",
    "CC_RATE_LIMIT_AZURE_CONCURRENT": "rate_limits.azure.max_concurrent",
    "CC_RETRY_MAX_ATTEMPTS": "retry.max_attempts",
    "CC_RETRY_BACKOFF_BASE": "retry.backoff_base_seconds",
    "CC_GOVERNANCE_DEFAULT_TIER": "governance.default_tier",
    "CC_MAX_DELEGATION_DEPTH": "delegation.max_depth",
    "CC_COST_BUDGET_DEFAULT": "cost.default_budget_usd",
    "CC_COST_UNKNOWN_MODEL_ACTION": "cost.unknown_model_action",
    "CC_PII_ENABLED": "pii.enabled",
    "CC_GUARDRAILS_ENABLED": "guardrails.enabled",
    "CC_GUARDRAILS_LLM_ENABLED": "guardrails.llm_classifier_enabled",
    "CC_MANIFEST_ENABLED": "manifest.enabled",
    "CC_MANIFEST_STORE_CONTENT": "manifest.store_content",
    "CC_AUDIT_PAYLOAD_TTL_DAYS": "audit.payload_ttl_days",
    "CC_HEALTH_PORT": "health.port",
    "CC_LOG_LEVEL": "logging.level",
    "CC_LOG_FORMAT": "logging.format",
    "CC_WORKER_MAX_THREADS": "worker.max_threads",
}


def _load_env_overrides() -> dict[str, Any]:
    """
    Load CC_* environment variables and map to config paths.
    Also supports arbitrary CC_CONFIG_* for unmapped overrides.
    """
    result = {}

    # Known mappings
    for env_key, config_path in _ENV_MAPPINGS.items():
        value = os.environ.get(env_key)
        if value is not None:
            keys = config_path.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = _auto_convert(value)

    # Arbitrary overrides: CC_CONFIG__path__to__key=value
    # Double underscores map to dots in the config path
    for key, value in os.environ.items():
        if key.startswith("CC_CONFIG__"):
            config_path = key[len("CC_CONFIG__"):].lower().replace("__", ".")
            keys = config_path.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = _auto_convert(value)

    return result


def _auto_convert(value: str) -> Any:
    """Convert string values to appropriate types."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ═══════════════════════════════════════════════════════════════════
# Singleton / Module-level Access
# ═══════════════════════════════════════════════════════════════════

_instance: ConfigLoader | None = None


def get_config(
    env: str | None = None,
    project_root: str | None = None,
) -> ConfigLoader:
    """
    Get or create the singleton config loader.
    First call initializes; subsequent calls return cached instance.
    """
    global _instance
    if _instance is None:
        _env = env or os.environ.get("CC_ENV", "dev")
        _root = project_root or os.environ.get("CC_PROJECT_ROOT", ".")
        _instance = ConfigLoader(env=_env, project_root=_root)
        _instance.load()
    return _instance


def load_config(
    env: str = "dev",
    project_root: str = ".",
    base_files: list[str] | None = None,
) -> ConfigLoader:
    """Create a fresh (non-singleton) config loader."""
    loader = ConfigLoader(env=env, project_root=project_root, base_files=base_files)
    loader.load()
    return loader


def reset_config():
    """Reset singleton for testing."""
    global _instance
    _instance = None
