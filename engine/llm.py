"""
Cognitive Core — LLM Provider Factory

Single point of LLM construction for the entire framework.
Every module imports `create_llm` from here — nowhere else.

Configuration (in priority order):
  1. Explicit `provider` argument to create_llm()
  2. LLM_PROVIDER environment variable
  3. default_provider in llm_config.yaml
  4. Auto-detect from available API key env vars

Model aliasing:
  YAML configs and CLI use logical model names ("default", "fast",
  "standard", "strong"). The alias table in llm_config.yaml maps
  them to provider-specific identifiers. Provider-specific names
  also work as pass-through (e.g., "gpt-4o" routes automatically).

To switch providers:
  Option A: Set LLM_PROVIDER=azure_foundry (env var)
  Option B: Edit default_provider in llm_config.yaml
  No Python code changes required.

Supported providers:
  google         — Google Gemini (langchain-google-genai)
  azure          — Azure OpenAI Service (langchain-openai)
  azure_foundry  — Azure AI Foundry / Model Catalog (langchain-azure-ai)
  openai         — OpenAI direct (langchain-openai)
  bedrock        — Amazon Bedrock (langchain-aws)

Design rules:
  - Returns langchain BaseChatModel — all downstream code is provider-blind
  - No provider-specific imports at module level (lazy imports only)
  - Model aliases and provider mappings live in llm_config.yaml, not code
  - Unknown model strings pass through to the provider unchanged
"""

import os
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models.chat_models import BaseChatModel


# ═══════════════════════════════════════════════════════════════════════
# Configuration loading
# ═══════════════════════════════════════════════════════════════════════

_CONFIG_PATHS = [
    Path(os.environ.get("LLM_CONFIG_PATH", "")),       # explicit env var
    Path.cwd() / "llm_config.yaml",                    # working directory
    Path(__file__).parent.parent / "llm_config.yaml",   # repo root
]

_config_cache: dict | None = None


def _load_config() -> dict:
    """Load and cache the LLM config file. Falls back to built-in defaults."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    for p in _CONFIG_PATHS:
        if p and p.is_file():
            with open(p) as f:
                _config_cache = yaml.safe_load(f) or {}
            return _config_cache

    # Built-in fallback (no config file found)
    _config_cache = _BUILTIN_DEFAULTS
    return _config_cache


# Hardcoded fallback so the framework runs even without llm_config.yaml.
# This is intentionally minimal — the config file is the source of truth.
_BUILTIN_DEFAULTS: dict = {
    "default_provider": None,  # force auto-detect
    "aliases": {
        "default": {
            "google": "gemini-2.0-flash",
            "azure": "gpt-4o-mini",
            "azure_foundry": "gpt-4o-mini",
            "openai": "gpt-4o-mini",
            "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
        },
        "fast": {
            "google": "gemini-2.0-flash",
            "azure": "gpt-4o-mini",
            "azure_foundry": "gpt-4o-mini",
            "openai": "gpt-4o-mini",
            "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
        },
        "standard": {
            "google": "gemini-2.5-pro",
            "azure": "gpt-4o",
            "azure_foundry": "gpt-4o",
            "openai": "gpt-4o",
            "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        },
        "strong": {
            "google": "gemini-2.5-pro",
            "azure": "gpt-4o",
            "azure_foundry": "gpt-4o",
            "openai": "gpt-4o",
            "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        },
    },
    "model_to_provider": {
        "gemini-2.0-flash": "google",
        "gemini-2.5-pro": "google",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4.1": "openai",
        "gpt-4.1-mini": "openai",
    },
    "provider_defaults": {
        "google": "gemini-2.0-flash",
        "azure": "gpt-4o",
        "azure_foundry": "gpt-4o",
        "openai": "gpt-4o",
        "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
    "provider_settings": {},
}


# ═══════════════════════════════════════════════════════════════════════
# Convenience accessors (read from config)
# ═══════════════════════════════════════════════════════════════════════

def _aliases() -> dict[str, dict[str, str]]:
    return _load_config().get("aliases", {})

def _model_to_provider() -> dict[str, str]:
    return _load_config().get("model_to_provider", {})

def _provider_defaults() -> dict[str, str]:
    return _load_config().get("provider_defaults", {})

def _provider_settings() -> dict:
    return _load_config().get("provider_settings", {})


# ═══════════════════════════════════════════════════════════════════════
# Provider detection
# ═══════════════════════════════════════════════════════════════════════

def detect_provider() -> str:
    """
    Detect LLM provider. Priority:
      1. LLM_PROVIDER env var
      2. default_provider in llm_config.yaml
      3. Auto-detect from API key env vars
    """
    # 1. Explicit env var
    explicit = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if explicit:
        return explicit

    # 2. Config file default
    cfg_default = _load_config().get("default_provider")
    if cfg_default:
        return cfg_default.lower().strip()

    # 3. Auto-detect from API keys (in priority order for Navy Federal)
    if os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT"):
        return "azure_foundry"
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    if os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_PROFILE"):
        return "bedrock"

    raise EnvironmentError(
        "No LLM provider detected. Set one of:\n"
        "  LLM_PROVIDER=azure_foundry|azure|openai|google|bedrock\n"
        "  Or set default_provider in llm_config.yaml\n"
        "  Or set provider API key env vars:\n"
        "    AZURE_AI_FOUNDRY_ENDPOINT=...\n"
        "    AZURE_OPENAI_ENDPOINT=... + AZURE_OPENAI_API_KEY=...\n"
        "    OPENAI_API_KEY=...\n"
        "    GOOGLE_API_KEY=...\n"
        "    AWS_DEFAULT_REGION=... (for Bedrock)"
    )


def resolve_model(model: str, provider: str) -> str:
    """
    Resolve a model string to a provider-specific model ID.

    Handles:
      - Logical aliases: "default", "fast", "standard", "strong" (from config)
      - Provider-specific names: pass through unchanged
      - LLM_DEFAULT_MODEL env override when model is "default"
    """
    # Env override for the default model
    if model == "default":
        env_model = os.environ.get("LLM_DEFAULT_MODEL", "").strip()
        if env_model:
            return env_model

    # Check alias table from config
    aliases = _aliases()
    if model in aliases:
        alias_map = aliases[model]
        if provider in alias_map:
            return alias_map[provider]

    # Pass through as-is (provider-specific model name)
    return model


# ═══════════════════════════════════════════════════════════════════════
# Provider factories (lazy imports)
# ═══════════════════════════════════════════════════════════════════════

def _create_google(model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)


def _create_azure(model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI
    settings = _provider_settings().get("azure", {})
    return AzureChatOpenAI(
        azure_deployment=model,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get(
            "AZURE_OPENAI_VERSION",
            settings.get("api_version", "2024-12-01-preview"),
        ),
        temperature=temperature,
        **kwargs,
    )


def _create_azure_foundry(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """
    Azure AI Foundry — Model Catalog inference endpoint.

    Uses the Azure AI Inference SDK via langchain-azure-ai.
    This is the recommended path for Azure AI Foundry deployments,
    supporting both Azure OpenAI models and open-source models
    deployed through the Model Catalog.

    Environment variables:
      AZURE_AI_FOUNDRY_ENDPOINT  — Foundry project endpoint
                                   (e.g., https://<project>.services.ai.azure.com)
      AZURE_AI_FOUNDRY_API_KEY   — API key (optional if using DefaultAzureCredential)

    Falls back to AzureChatOpenAI if langchain-azure-ai is not installed,
    for backward compatibility during migration.
    """
    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", "")
    api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")

    if not endpoint:
        raise EnvironmentError(
            "AZURE_AI_FOUNDRY_ENDPOINT is required for azure_foundry provider.\n"
            "Set it to your Foundry project endpoint, e.g.:\n"
            "  https://<project>.services.ai.azure.com"
        )

    try:
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

        # Build credential — API key or DefaultAzureCredential
        cred_kwargs: dict[str, Any] = {}
        if api_key:
            cred_kwargs["api_key"] = api_key
        else:
            # Use Azure Identity (Managed Identity, CLI, etc.)
            from azure.identity import DefaultAzureCredential
            cred_kwargs["credential"] = DefaultAzureCredential()

        return AzureAIChatCompletionsModel(
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            **cred_kwargs,
            **kwargs,
        )

    except ImportError:
        # Graceful fallback: use AzureChatOpenAI if langchain-azure-ai
        # isn't installed yet. This lets teams migrate incrementally.
        import warnings
        warnings.warn(
            "langchain-azure-ai not installed. Falling back to AzureChatOpenAI.\n"
            "Install for full Foundry support: pip install langchain-azure-ai azure-identity",
            stacklevel=2,
        )
        from langchain_openai import AzureChatOpenAI
        settings = _provider_settings().get("azure_foundry", {})
        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.environ.get(
                "AZURE_OPENAI_VERSION",
                settings.get("api_version", "2024-12-01-preview"),
            ),
            temperature=temperature,
            **kwargs,
        )


def _create_openai(model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def _create_bedrock(model: str, temperature: float, **kwargs) -> BaseChatModel:
    from langchain_aws import ChatBedrock
    settings = _provider_settings().get("bedrock", {})
    return ChatBedrock(
        model_id=model,
        model_kwargs={"temperature": temperature},
        region_name=os.environ.get(
            "AWS_DEFAULT_REGION",
            settings.get("region", "us-east-1"),
        ),
        **kwargs,
    )


_FACTORIES = {
    "google":         _create_google,
    "azure":          _create_azure,
    "azure_foundry":  _create_azure_foundry,
    "openai":         _create_openai,
    "bedrock":        _create_bedrock,
}


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def create_llm(
    model: str = "default",
    temperature: float = 0.1,
    provider: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """
    Create an LLM instance for Cognitive Core.

    This is the ONLY function the rest of the codebase calls to get an LLM.

    Args:
        model:       Logical alias ("default", "fast", "standard", "strong")
                     or provider-specific model name ("gpt-4o", "gemini-2.0-flash").
        temperature: Sampling temperature.
        provider:    Force a provider. If None, auto-detected.
        **kwargs:    Passed through to the underlying LangChain constructor.

    Returns:
        BaseChatModel — ready to call with .invoke(), .ainvoke(), etc.
    """
    # Detect provider
    if provider is None:
        # Try to infer from model name first
        m2p = _model_to_provider()
        if model in m2p:
            provider = m2p[model]
        else:
            provider = detect_provider()

    provider = provider.lower().strip()
    if provider not in _FACTORIES:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(_FACTORIES.keys())}"
        )

    # Resolve model alias
    resolved = resolve_model(model, provider)

    # Apply timeout from config or env var (seconds)
    # Prevents hangs when LLM loops or connection stalls
    timeout = kwargs.pop("timeout", None)
    if timeout is None:
        env_timeout = os.environ.get("LLM_TIMEOUT_SECONDS", "").strip()
        if env_timeout:
            timeout = int(env_timeout)
        else:
            cfg_timeout = _load_config().get("timeout_seconds")
            if cfg_timeout:
                timeout = int(cfg_timeout)
    if timeout and "timeout" not in kwargs:
        kwargs["timeout"] = timeout

    # Create LLM
    factory = _FACTORIES[provider]
    return factory(resolved, temperature, **kwargs)


def get_provider_info() -> dict[str, str]:
    """Return current provider configuration for diagnostics."""
    try:
        provider = detect_provider()
    except EnvironmentError:
        provider = "none"

    defaults = _provider_defaults()
    return {
        "provider": provider,
        "default_model": os.environ.get(
            "LLM_DEFAULT_MODEL",
            defaults.get(provider, "unknown"),
        ),
        "env_override": os.environ.get("LLM_PROVIDER", ""),
        "config_file": _find_config_path() or "built-in defaults",
    }


def _find_config_path() -> str | None:
    """Return the path of the config file being used, or None."""
    for p in _CONFIG_PATHS:
        if p and p.is_file():
            return str(p)
    return None


def validate_config() -> list[str]:
    """
    Validate the current LLM configuration. Returns list of issues.
    Call at startup for early failure.
    """
    issues = []
    cfg = _load_config()

    # Check aliases have entries for all known providers
    aliases = cfg.get("aliases", {})
    providers = set(_FACTORIES.keys())
    for alias_name, mapping in aliases.items():
        missing = providers - set(mapping.keys())
        if missing:
            issues.append(
                f"Alias '{alias_name}' missing providers: {', '.join(sorted(missing))}"
            )

    # Check provider can be detected
    try:
        provider = detect_provider()
        # Check that the detected provider has a default model
        defaults = cfg.get("provider_defaults", {})
        if provider not in defaults:
            issues.append(f"No default model defined for provider '{provider}'")
    except EnvironmentError as e:
        issues.append(str(e))

    return issues


def reload_config() -> None:
    """Force reload of the config file. Useful for testing."""
    global _config_cache
    _config_cache = None


# ═══════════════════════════════════════════════════════════════════
# Retry-Aware LLM Invocation
# ═══════════════════════════════════════════════════════════════════

def create_fallback_llm(
    provider: str,
    temperature: float = 0.1,
    **kwargs,
) -> "BaseChatModel | None":
    """
    Create a fallback LLM for the given provider, if configured.

    Reads the `retry.<provider>.fallback_model` from llm_config.yaml.
    Returns None if no fallback is configured.
    """
    try:
        from engine.retry import get_retry_policy
    except ImportError:
        return None

    policy = get_retry_policy(provider)
    if not policy.fallback_model:
        return None

    try:
        return create_llm(
            model=policy.fallback_model,
            temperature=temperature,
            provider=provider,
            **kwargs,
        )
    except Exception:
        return None
