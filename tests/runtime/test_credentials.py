"""Credentials must stay with the selected service and out of audit errors."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import run
from shared.settings import Settings
from shared import preflight
from stage1_inputs import client
from shared.secrets import scan


class CredentialRoutingTests(unittest.TestCase):
    def test_custom_variable_and_file_target_only_the_selected_provider(self):
        for provider, (_, native_env) in client.PROVIDERS.items():
            for source in ("custom_env", "file"):
                with self.subTest(provider=provider, source=source), tempfile.TemporaryDirectory() as tmp:
                    cfg = Settings()
                    cfg.lm.provider = provider
                    cfg.lm.api_key_env = "BENCH_TEST_CREDENTIAL"
                    if source == "file":
                        key_file = Path(tmp) / "credential"
                        key_file.write_text("file-credential\n")
                        cfg.lm.api_key_file = str(key_file)
                    with patch.dict(os.environ, {"BENCH_TEST_CREDENTIAL": "custom-credential"}, clear=True):
                        run.install_key(cfg)
                        self.assertEqual(client.resolve_provider(provider)[0], provider)
                        self.assertEqual(os.environ[native_env],
                                         "file-credential" if source == "file" else "custom-credential")
                        for other, (_, env) in client.PROVIDERS.items():
                            if other != provider:
                                self.assertNotIn(env, os.environ)

    def test_standard_environment_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "credential"
            key_file.write_text("file-fixture")
            cfg = Settings()
            cfg.lm.api_key_file = str(key_file)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "std-fixture"}, clear=True):
                run.install_key(cfg)
                self.assertEqual(client.resolve_provider("openrouter")[2], "std-fixture")

    def test_explicit_provider_never_falls_back(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "other-fixture"}, clear=True):
            self.assertIsNone(client.resolve_provider("deepseek"))
            self.assertIsNone(client.resolve_provider("unknown"))
            self.assertEqual(client.resolve_provider()[0], "openrouter")

    def test_preflight_does_not_accept_another_providers_key(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "other-fixture"}, clear=True):
            result = preflight.check(Path(tmp), want_modules=(), want_lm=True,
                                     provider="deepseek", model="fixture-model")
        self.assertFalse(result.lm_ready)
        self.assertEqual(result.lm_provider, "")
        self.assertIn("DEEPSEEK_API_KEY", result.render())

    def test_blank_standard_variable_allows_custom_variable(self):
        cfg = Settings()
        cfg.lm.api_key_env = "BENCH_TEST_CREDENTIAL"
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "  ",
                                     "BENCH_TEST_CREDENTIAL": "custom-fixture"}, clear=True):
            run.install_key(cfg)
            self.assertEqual(client.resolve_provider("openrouter")[2], "custom-fixture")

    def test_wrong_provider_variable_is_rejected(self):
        cfg = Settings()
        cfg.lm.provider = "deepseek"
        cfg.lm.api_key_env = "OPENROUTER_API_KEY"
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "other-fixture"}, clear=True):
            with self.assertRaises(SystemExit):
                run.install_key(cfg)
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)

    def test_empty_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "credential"
            key_file.write_text(" \n")
            cfg = Settings()
            cfg.lm.api_key_file = str(key_file)
            with self.assertRaises(SystemExit):
                run.install_key(cfg)

    def test_transport_error_never_persists_echoed_secrets(self):
        secret = "fake-credential-" + "x" * 30
        log = client.CallLog()
        error = urllib.error.HTTPError("https://example.invalid", 401, secret, {}, io.BytesIO())
        with patch.object(client, "resolve_provider", return_value=(
                "openrouter", "https://example.invalid/chat", secret)), \
                patch.object(client.urllib.request, "urlopen", side_effect=error), \
                patch.object(client.time, "sleep"):
            self.assertEqual(client.make_call_fn(log=log)("test"), "")
        self.assertEqual(log.errors, ["HTTPError: HTTP 401"])
        self.assertNotIn(secret, json.dumps(log.__dict__))


class CredentialScannerTests(unittest.TestCase):
    def test_hidden_and_ignored_data_are_scanned_without_values_in_output(self):
        secret = "sk-or-v1-" + "a1b2c3d4" * 8
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / ".env.local").write_text("API_KEY=" + secret)
            (root / ".git").mkdir()
            (root / ".git" / "ignored").write_text(secret)
            (root / "image.png").write_bytes(b"\x89PNG\0" + secret.encode())
            report = scan(root)
        self.assertEqual(report["status"], "review_required")
        self.assertEqual({row["file"] for row in report["findings"]}, {"data/.env.local"})
        self.assertNotIn(secret, json.dumps(report))
        self.assertEqual(report["git_history"], "not_scanned")

    def test_plaintext_assignments_and_private_keys_are_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.txt").write_text('client_secret="' + "abc123" * 5 + '"\n')
            (root / "local.key").write_text("private-material")
            report = scan(root)
        self.assertEqual({row["rule"] for row in report["findings"]},
                         {"credential_assignment", "credential_file"})


if __name__ == "__main__":
    unittest.main()
