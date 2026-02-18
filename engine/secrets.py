"""
Cognitive Core — Secrets Management (S-006)

Hybrid secrets loader:
  1. Azure Key Vault (if AZURE_KEY_VAULT_URL set and azure-identity installed)
  2. Environment variables (universal fallback)

Secrets are loaded at startup and cached in-memory.
Secrets are NEVER logged, exposed in health endpoints, or written to audit trails.

Usage:
    from engine.secrets import get_secret, SecretStore

    # Auto-detects source (Key Vault or env var)
    api_key = get_secret("GOOGLE_API_KEY")

    # Explicit store
    store = SecretStore()
    store.get("AZURE_OPENAI_API_KEY")

Configuration:
    AZURE_KEY_VAULT_URL     — Key Vault URI (e.g., https://myvault.vault.azure.net)
    CC_SECRETS_CACHE_TTL    — Cache TTL in seconds (default: 3600 = 1 hour)

Key Vault secret name mapping:
    Environment variable names are converted to Key Vault secret names:
    GOOGLE_API_KEY → google-api-key (underscores → hyphens, lowercase)
"""

from __future__ import annotations

import logging
import os
import time
import threading
from typing import Any

logger = logging.getLogger("cognitive_core.secrets")


class SecretStore:
    """
    Thread-safe secrets store with Key Vault + env var fallback.

    Lazy-loads Key Vault client on first access. Caches values
    in memory with configurable TTL.
    """

    def __init__(
        self,
        vault_url: str = "",
        cache_ttl: int = 3600,
    ):
        self._vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL", "")
        self._cache_ttl = int(os.environ.get("CC_SECRETS_CACHE_TTL", str(cache_ttl)))
        self._cache: dict[str, tuple[str, float]] = {}  # name → (value, expires_at)
        self._lock = threading.Lock()
        self._kv_client = None
        self._kv_available: bool | None = None  # None = not checked yet

    def get(self, name: str, default: str = "") -> str:
        """
        Get a secret by name.

        Resolution order:
          1. In-memory cache (if not expired)
          2. Azure Key Vault (if available)
          3. Environment variable
          4. Default value

        Args:
            name: Secret name (e.g., "GOOGLE_API_KEY")
            default: Fallback if not found anywhere

        Returns:
            Secret value (never logged)
        """
        # Check cache
        with self._lock:
            if name in self._cache:
                value, expires = self._cache[name]
                if time.time() < expires:
                    return value

        # Try Key Vault
        value = self._get_from_vault(name)
        if value is not None:
            self._set_cache(name, value)
            return value

        # Fall back to env var
        value = os.environ.get(name, "")
        if value:
            self._set_cache(name, value)
            return value

        return default

    def _get_from_vault(self, name: str) -> str | None:
        """
        Fetch from Azure Key Vault.
        Returns None if vault not configured or secret not found.
        """
        if not self._vault_url:
            return None

        # Check if KV is available (lazy init)
        if self._kv_available is False:
            return None

        if self._kv_client is None:
            self._kv_client = self._init_kv_client()
            if self._kv_client is None:
                self._kv_available = False
                return None
            self._kv_available = True

        # Convert env var name to Key Vault name
        kv_name = self._to_kv_name(name)

        try:
            secret = self._kv_client.get_secret(kv_name)
            logger.debug("Secret loaded from Key Vault: %s", kv_name)
            return secret.value
        except Exception as e:
            # Don't log the error detail — it might contain hints about secret names
            logger.debug("Key Vault lookup failed for %s, falling back to env var", kv_name)
            return None

    def _init_kv_client(self):
        """Initialize Azure Key Vault client. Returns None if SDK unavailable."""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=self._vault_url, credential=credential)
            logger.info("Key Vault client initialized: %s", self._vault_url)
            return client
        except ImportError:
            logger.debug("azure-keyvault-secrets or azure-identity not installed — "
                        "using env vars only")
            return None
        except Exception as e:
            logger.warning("Key Vault initialization failed: %s — using env vars only", e)
            return None

    def _set_cache(self, name: str, value: str):
        """Cache a secret value with TTL."""
        with self._lock:
            self._cache[name] = (value, time.time() + self._cache_ttl)

    def clear_cache(self):
        """Clear the secret cache (e.g., for rotation)."""
        with self._lock:
            self._cache.clear()
        logger.info("Secret cache cleared")

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def vault_available(self) -> bool:
        """Whether Key Vault is configured and accessible."""
        return self._kv_available is True

    @staticmethod
    def _to_kv_name(env_name: str) -> str:
        """
        Convert environment variable name to Key Vault secret name.
        GOOGLE_API_KEY → google-api-key
        """
        return env_name.lower().replace("_", "-")

    @staticmethod
    def _from_kv_name(kv_name: str) -> str:
        """
        Convert Key Vault name back to env var format.
        google-api-key → GOOGLE_API_KEY
        """
        return kv_name.upper().replace("-", "_")


# ═══════════════════════════════════════════════════════════════════
# Module-level convenience
# ═══════════════════════════════════════════════════════════════════

_default_store: SecretStore | None = None
_store_lock = threading.Lock()


def get_secret(name: str, default: str = "") -> str:
    """Get a secret from the default store."""
    global _default_store
    with _store_lock:
        if _default_store is None:
            _default_store = SecretStore()
    return _default_store.get(name, default)


def get_store() -> SecretStore:
    """Get the default SecretStore instance."""
    global _default_store
    with _store_lock:
        if _default_store is None:
            _default_store = SecretStore()
    return _default_store


def reset_store():
    """Reset the default store (for testing)."""
    global _default_store
    with _store_lock:
        _default_store = None
