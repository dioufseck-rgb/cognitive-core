"""Provider regressions: retain real SDK request construction, mock only HTTP."""
from __future__ import annotations

import asyncio
from importlib.util import find_spec
import json
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import warnings

from cognitive_core.engine import llm


CONFIG = {
    "default_provider": "openai",
    "aliases": {"fast": {"openai": "gpt-4o-mini"}},
    "primitive_model_alias": {"retrieve": "fast"},
    "primitive_max_tokens": {"retrieve": 4096},
}


class TokenLimitFactoryTests(unittest.TestCase):
    def test_only_the_resolved_provider_parameter_reaches_its_factory(self):
        for provider in ("openai", "azure", "azure_foundry", "anthropic", "bedrock", "google"):
            for limits in (
                {"max_tokens": 4096},
                {"max_output_tokens": 4096},
                {"max_tokens": 4096, "max_output_tokens": 4096},
                {"max_tokens": None, "max_output_tokens": 4096},
            ):
                with self.subTest(provider=provider, limits=limits):
                    factory = Mock(return_value=object())
                    with patch.object(llm, "_load_config", return_value=CONFIG):
                        with patch.dict(llm._FACTORIES, {provider: factory}):
                            llm.create_llm(model="offline-model", provider=provider, timeout=7, **limits)
                    key = "max_output_tokens" if provider == "google" else "max_tokens"
                    factory.assert_called_once_with("offline-model", 0.1, timeout=7, **{key: 4096})

    def test_omitted_and_explicit_none_limits_remain_distinct(self):
        for provider in ("openai", "google"):
            for limits in ({}, {"max_tokens": None}, {"max_output_tokens": None}):
                with self.subTest(provider=provider, limits=limits):
                    factory = Mock(return_value=object())
                    with patch.object(llm, "_load_config", return_value=CONFIG):
                        with patch.dict(llm._FACTORIES, {provider: factory}):
                            llm.create_llm(model="offline-model", provider=provider, **limits)
                    key = "max_output_tokens" if provider == "google" else "max_tokens"
                    expected = {key: None} if limits else {}
                    factory.assert_called_once_with("offline-model", 0.1, **expected)

    def test_conflicting_limits_fail_before_provider_construction(self):
        factory = Mock()
        with patch.object(llm, "_load_config", return_value=CONFIG):
            with patch.dict(llm._FACTORIES, {"openai": factory}):
                with self.assertRaisesRegex(ValueError, "Conflicting output token limits"):
                    llm.create_llm(provider="openai", max_tokens=4096, max_output_tokens=8192)
        factory.assert_not_called()

    def test_google_adapter_preserves_the_configured_cap_and_its_default(self):
        # Strict constructor contract without Google's optional SDK or a key.
        module = ModuleType("langchain_google_genai")
        received = []

        def constructor(*, model, temperature, max_output_tokens):
            received.append(max_output_tokens)
            return object()

        module.ChatGoogleGenerativeAI = constructor
        with patch.dict(sys.modules, {"langchain_google_genai": module}):
            with patch.object(llm, "_load_config", return_value=CONFIG):
                llm.create_llm(provider="google", max_tokens=4096)
                llm.create_llm(provider="google")
                llm.create_llm(provider="google", max_tokens=None)
        self.assertEqual(received, [4096, 16384, None])


@unittest.skipUnless(
    find_spec("langchain_openai") is not None and find_spec("httpx") is not None,
    "Install langchain-openai and httpx to run real SDK request-contract tests.",
)
class OpenAIRequestContractTests(unittest.TestCase):
    def exercise_real_sdk(self, make_model):
        import httpx

        seen = []

        def handler(request):
            self.assertEqual(request.url.host, "openai.invalid")
            self.assertEqual(request.url.path, "/v1/chat/completions")
            payload = json.loads(request.content)
            seen.append(payload)
            self.assertNotIn("max_output_tokens", payload)
            self.assertEqual(payload.get("max_completion_tokens", payload.get("max_tokens")), 4096)
            self.assertEqual(payload["model"], "gpt-4o-mini")
            return httpx.Response(200, json={
                "id": "chatcmpl-offline", "object": "chat.completion",
                "created": 0, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        transport = httpx.MockTransport(handler)
        original_factory = llm._FACTORIES["openai"]
        with httpx.Client(transport=transport) as http_client:
            async_client = httpx.AsyncClient(transport=transport)

            def offline_factory(model, temperature, **kwargs):
                # Inject only transport. The real factory, ChatOpenAI, and
                # OpenAI SDK still reject any invalid request parameters.
                return original_factory(
                    model, temperature, api_key="sk-offline-test",
                    base_url="https://openai.invalid/v1",
                    http_client=http_client, http_async_client=async_client,
                    max_retries=0, **kwargs,
                )

            try:
                with patch.object(llm, "_load_config", return_value=CONFIG):
                    with patch.dict(llm._FACTORIES, {"openai": offline_factory}):
                        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "LLM_DEFAULT_MODEL": ""}):
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                model = make_model()
                                self.assertEqual(model.invoke("Return OK.").content, "OK")
                                self.assertFalse(any("max_output_tokens" in str(item.message) for item in caught))
            finally:
                asyncio.run(async_client.aclose())
        self.assertEqual(len(seen), 1)

    def test_native_retrieve_builds_a_valid_chat_completions_request(self):
        from cognitive_core.engine.profile_nodes import _model_for

        def make_model():
            model, alias = _model_for("retrieve", "default", 0.1)
            self.assertEqual(alias, "fast")
            return model

        self.exercise_real_sdk(make_model)

    def test_legacy_duplicate_aliases_do_not_leak_into_the_openai_request(self):
        self.exercise_real_sdk(lambda: llm.create_llm(
            model="fast", provider="openai", max_tokens=4096, max_output_tokens=4096,
        ))


if __name__ == "__main__":
    unittest.main()
