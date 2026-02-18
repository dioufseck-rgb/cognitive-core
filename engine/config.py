"""
Cognitive Core — Environment Config Loader (S-004)

Three-tier configuration loading:
  1. Azure App Configuration (if AZURE_APP_CONFIG_ENDPOINT is set and SDK available)
  2. Per-environment overlay files (config/{CC_ENV}.yaml merged over base)
  3. Environment variable overrides (CC_ prefixed)

Usage:
    from engine.config import load_config, get_config_value

    # Load full merged config
    cfg = load_config(base_path="llm_config.yaml", env="prod")

    # Get a specific value with fallback
    rpm = get_config_value("rate_limits.google.requests_per_minute", default=60)

Environment variables:
    CC_ENV                      — active profile (dev, staging, prod)
    CC_CONFIG_DIR               — directory for overlay files (default: config/)
    AZURE_APP_CONFIG_ENDPOINT   — Azure App Configuration endpoint (optional)
    CC_*                        — flat overrides (e.g., CC_RATE_LIMITS_GOOGLE_RPM=120)
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("cognitive_core.config")


# ═══════════════════════════════════════════════════════════════════
# Deep Merge
# ═══════════════════════════════════════════════════════════════════

def deep_merge(base: dict, overlay: dict) -> dict:
    """
    Deep-merge overlay into base. Overlay values win.
    Lists are replaced (not appended). Dicts are recursed.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ═══════════════════════════════════════════════════════════════════
# Tier 1: Azure App Configuration
# ═══════════════════════════════════════════════════════════════════

def _load_azure_app_config(label: str = "") -> dict[str, Any]:
    """
    Load config from Azure App Configuration.
    Returns empty dict if not configured or SDK unavailable.
    """
    endpoint = os.environ.get("AZURE_APP_CONFIG_ENDPOINT")
    if not endpoint:
        return {}

    try:
        from azure.appconfiguration import AzureAppConfigurationClient
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        client = AzureAppConfigurationClient(endpoint, credential)

        config = {}
        label_filter = label or os.environ.get("CC_ENV", "")
        settings = client.list_configuration_settings(label_filter=label_filter or None)

        for setting in settings:
            # Convert dotted keys to nested dict: "rate_limits.google.rpm" → nested
            _set_nested(config, setting.key.split("."), setting.value)

        logger.info("Loaded %d settings from Azure App Configuration (label=%s)",
                     len(config), label_filter)
        return config

    except ImportError:
        logger.debug("azure-appconfiguration SDK not installed — skipping Azure App Config")
        return {}
    except Exception as e:
        logger.warning("Azure App Configuration failed: %s — falling back to file config", e)
        return {}


def _set_nested(d: dict, keys: list[str], value: str):
    """Set a nested dict value from a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Try to parse value as YAML (handles numbers, booleans, lists)
    try:
        parsed = yaml.safe_load(value)
        d[keys[-1]] = parsed
    except Exception:
        d[keys[-1]] = value


# ═══════════════════════════════════════════════════════════════════
# Tier 2: Overlay Files
# ═══════════════════════════════════════════════════════════════════

def _load_overlay_file(
    base_path: str,
    env: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    """
    Load per-environment overlay file.
    Looks for config/{env}.yaml or {config_dir}/{env}.yaml.
    Returns empty dict if not found.
    """
    env = env or os.environ.get("CC_ENV", "")
    if not env:
        return {}

    config_dir = config_dir or os.environ.get("CC_CONFIG_DIR", "config")

    # Try multiple paths
    candidates = [
        Path(config_dir) / f"{env}.yaml",
        Path(config_dir) / f"{env}.yml",
        Path(os.path.dirname(base_path)) / "config" / f"{env}.yaml",
    ]

    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    overlay = yaml.safe_load(f) or {}
                logger.info("Loaded config overlay: %s (%d keys)", path, len(overlay))
                return overlay
            except Exception as e:
                logger.warning("Failed to load overlay %s: %s", path, e)

    logger.debug("No config overlay found for env=%s", env)
    return {}


# ═══════════════════════════════════════════════════════════════════
# Tier 3: Environment Variable Overrides
# ═══════════════════════════════════════════════════════════════════

def _load_env_overrides(prefix: str = "CC_") -> dict[str, Any]:
    """
    Load CC_ prefixed environment variables as config overrides.

    Naming convention:
      CC_SECTION_KEY=value → {"section": {"key": "value"}}
      CC_SECTION_SUBSECTION_KEY=value → {"section": {"subsection": {"key": "value"}}}

    Special handling:
      - Values are auto-parsed (numbers, booleans, lists)
      - CC_ENV, CC_CONFIG_DIR, CC_WORKER_MODE are excluded (meta config)
    """
    excluded = {"CC_ENV", "CC_CONFIG_DIR", "CC_WORKER_MODE", "CC_PROJECT_ROOT", "CC_VERSION"}
    overrides: dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(prefix) or key in excluded:
            continue

        # Strip prefix and split into path
        path = key[len(prefix):].lower().split("_")

        # Parse value
        try:
            parsed = yaml.safe_load(value)
        except Exception:
            parsed = value

        _set_nested(overrides, path, str(parsed) if not isinstance(parsed, (int, float, bool)) else str(parsed))

    if overrides:
        logger.debug("Loaded %d env var overrides", len(overrides))
    return overrides


# ═══════════════════════════════════════════════════════════════════
# Main Loader
# ═══════════════════════════════════════════════════════════════════

def load_config(
    base_path: str = "llm_config.yaml",
    env: str = "",
    config_dir: str = "",
    include_azure: bool = True,
    include_env_vars: bool = True,
) -> dict[str, Any]:
    """
    Load configuration with three-tier merging.

    Priority (highest wins):
      1. Environment variable overrides (CC_*)
      2. Azure App Configuration (if available)
      3. Per-environment overlay file (config/{env}.yaml)
      4. Base config file (llm_config.yaml)

    Args:
        base_path: Path to base YAML config
        env: Environment name (overrides CC_ENV)
        config_dir: Overlay directory (overrides CC_CONFIG_DIR)
        include_azure: Whether to check Azure App Configuration
        include_env_vars: Whether to check CC_* env vars

    Returns:
        Merged configuration dict
    """
    # Base
    config = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Loaded base config: %s", base_path)

    # Overlay file
    overlay = _load_overlay_file(base_path, env=env, config_dir=config_dir)
    if overlay:
        config = deep_merge(config, overlay)

    # Azure App Configuration
    if include_azure:
        azure_config = _load_azure_app_config(label=env)
        if azure_config:
            config = deep_merge(config, azure_config)

    # Environment variable overrides
    if include_env_vars:
        env_overrides = _load_env_overrides()
        if env_overrides:
            config = deep_merge(config, env_overrides)

    # Stamp active profile into config for observability
    config["_active_env"] = env or os.environ.get("CC_ENV", "default")
    config["_config_source"] = base_path

    return config


def get_config_value(
    path: str,
    config: dict[str, Any] | None = None,
    default: Any = None,
) -> Any:
    """
    Get a nested config value by dotted path.

    Example:
        get_config_value("rate_limits.google.requests_per_minute", cfg, 60)
    """
    if config is None:
        config = load_config()

    keys = path.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
