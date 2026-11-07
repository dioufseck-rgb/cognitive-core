"""The shared model configuration must load under Windows legacy locales."""

import builtins
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cognitive_core.engine import llm
from cognitive_core.primitives import registry


class TestLLMConfigEncoding(unittest.TestCase):
    def test_utf8_bom_config_loads_with_a_legacy_default_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_config.yaml"
            path.write_text("default_provider: openai\nlabel: 'Mamadou — Sénégal'\n", encoding="utf-8-sig")
            original_open = builtins.open

            def legacy_locale_open(file, *args, **kwargs):
                if Path(file) == path and "encoding" not in kwargs:
                    kwargs["encoding"] = "cp1252"
                return original_open(file, *args, **kwargs)

            with patch.object(llm, "_CONFIG_PATHS", [path]), patch.object(llm, "_config_cache", None):
                with patch("builtins.open", side_effect=legacy_locale_open):
                    config = llm._load_config()
            self.assertEqual(config["default_provider"], "openai")
            self.assertEqual(config["label"], "Mamadou — Sénégal")

    def test_unicode_prompt_loads_with_a_legacy_default_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.txt"
            expected = "Contexte — Sénégal; citation “à vérifier”.\n{question}\n"
            path.write_text(expected, encoding="utf-8")
            original_read_text = Path.read_text

            def legacy_locale_read_text(file, *args, **kwargs):
                if file == path and "encoding" not in kwargs:
                    kwargs["encoding"] = "cp1252"
                return original_read_text(file, *args, **kwargs)

            with patch.object(registry, "PROMPTS_DIR", Path(directory)):
                with patch.object(Path, "read_text", legacy_locale_read_text):
                    prompt = registry.get_prompt_template("frame")
            self.assertEqual(prompt, expected)


if __name__ == "__main__":
    unittest.main()
