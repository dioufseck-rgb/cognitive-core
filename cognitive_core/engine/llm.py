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
try:
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:
    # Shim for environments without langchain_core.
    # BaseChatModel is only used as a return type annotation.
    class BaseChatModel:  # type: ignore[no-redef]
        """Minimal BaseChatModel shim for langchain-free operation."""
        def invoke(self, messages, **kwargs):
            raise NotImplementedError(
                "Real LLM calls require langchain_core. "
                "Install: pip install langchain-core"
            )


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
            "google": "gemini-2.0-flash-lite",
            "azure": "gpt-4o-mini",
            "azure_foundry": "gpt-4o-mini",
            "openai": "gpt-4o-mini",
            "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
        },
        "fast": {
            "google": "gemini-2.0-flash-lite",
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
        "gemini-2.0-flash-lite": "google",
        "gemini-2.5-pro": "google",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4.1": "openai",
        "gpt-4.1-mini": "openai",
    },
    "provider_defaults": {
        "google": "gemini-2.0-flash-lite",
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
    kwargs.setdefault("max_output_tokens", 16384)
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
# Model availability probe
# ═══════════════════════════════════════════════════════════════════════

_MODEL_NOT_AVAILABLE_PHRASES = (
    "not available",
    "not found",
    "does not exist",
    "no longer available",
    "deprecated",
    "404",
)


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Return True if this exception means the model name is invalid/deprecated."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _MODEL_NOT_AVAILABLE_PHRASES)


def _model_unavailable_message(model: str, provider: str) -> str:
    return (
        f"\n{'─' * 64}\n"
        f"  Model '{model}' is not available for your {provider} API key.\n"
        f"\n"
        f"  Fix — choose one of:\n"
        f"\n"
        f"  1. Check which models your key can access:\n"
        f"       python -m cognitive_core.engine.llm --list-models\n"
        f"\n"
        f"  2. Set a working model via environment variable:\n"
        f"       export LLM_DEFAULT_MODEL=<model-name>\n"
        f"\n"
        f"  3. Edit llm_config.yaml at the repo root and update the\n"
        f"     '{provider}' entry under 'aliases.default' (and 'fast')\n"
        f"     to a model name from step 1.\n"
        f"{'─' * 64}\n"
    )


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

    Raises:
        EnvironmentError: with a clear message if the model is unavailable.
    """
    # Detect provider
    if provider is None:
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

    # Create LLM — wrap in a proxy that gives clear errors on first call
    factory = _FACTORIES[provider]
    llm = factory(resolved, temperature, **kwargs)
    return _ModelWithClearErrors(llm, resolved, provider)


class _ModelWithClearErrors:
    """
    Thin wrapper around a BaseChatModel that intercepts model-not-available
    errors and re-raises them with a clear, actionable message.

    Transparent to all callers — proxies every attribute to the wrapped LLM.
    """

    def __init__(self, llm: BaseChatModel, model: str, provider: str):
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_provider", provider)

    def _wrap(self, result_or_exc):
        return result_or_exc

    def invoke(self, *args, **kwargs):
        try:
            return self._llm.invoke(*args, **kwargs)
        except Exception as exc:
            if _is_model_unavailable_error(exc):
                raise EnvironmentError(
                    _model_unavailable_message(self._model, self._provider)
                ) from exc
            raise

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self._llm.ainvoke(*args, **kwargs)
        except Exception as exc:
            if _is_model_unavailable_error(exc):
                raise EnvironmentError(
                    _model_unavailable_message(self._model, self._provider)
                ) from exc
            raise

    def stream(self, *args, **kwargs):
        try:
            return self._llm.stream(*args, **kwargs)
        except Exception as exc:
            if _is_model_unavailable_error(exc):
                raise EnvironmentError(
                    _model_unavailable_message(self._model, self._provider)
                ) from exc
            raise

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_llm"), name)

    def __setattr__(self, name: str, value):
        setattr(object.__getattribute__(self, "_llm"), name, value)


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


# ═══════════════════════════════════════════════════════════════════════
# CLI  —  python -m cognitive_core.engine.llm
# ═══════════════════════════════════════════════════════════════════════

def _cli_list_models() -> None:
    """List models available for the detected provider and API key."""
    try:
        provider = detect_provider()
    except EnvironmentError as e:
        print(f"Cannot detect provider: {e}")
        return

    print(f"\nProvider: {provider}")
    print(f"Config file: {_find_config_path() or 'built-in defaults'}\n")

    if provider == "google":
        try:
            import google.generativeai as genai
            import os
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            models = [
                m.name for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]
            print("Models available for your Google API key:")
            for m in sorted(models):
                print(f"  {m}")
            print(f"\nTo use one, either:")
            print(f"  export LLM_DEFAULT_MODEL={models[0] if models else '<model>'}")
            print(f"  or edit llm_config.yaml → aliases.default.google")
        except Exception as e:
            print(f"Could not list Google models: {e}")
            print("Check that GOOGLE_API_KEY is set.")

    elif provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI()
            models = sorted([m.id for m in client.models.list() if "gpt" in m.id])
            print("GPT models available for your OpenAI API key:")
            for m in models:
                print(f"  {m}")
        except Exception as e:
            print(f"Could not list OpenAI models: {e}")

    else:
        print(f"Model listing not yet supported for provider '{provider}'.")
        print("Check your provider's documentation for available model names.")
        print("Then set: export LLM_DEFAULT_MODEL=<model-name>")
        print("Or update llm_config.yaml → aliases.default and aliases.fast")


if __name__ == "__main__":
    import sys
    if "--list-models" in sys.argv or len(sys.argv) == 1:
        _cli_list_models()
    else:
        print("Usage: python -m cognitive_core.engine.llm --list-models")
