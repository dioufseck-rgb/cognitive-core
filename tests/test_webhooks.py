"""Tests for S-014: Webhook Notifications."""

import importlib.util
import os
import sys
import threading
import time
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_mod_path = os.path.join(_base, "engine", "webhooks.py")
_spec = importlib.util.spec_from_file_location("engine.webhooks", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.webhooks"] = _mod
_spec.loader.exec_module(_mod)

WebhookConfig = _mod.WebhookConfig
WebhookNotifier = _mod.WebhookNotifier
_format_payload = _mod._format_payload


def _mock_http_success(**kwargs):
    """Mock HTTP client that always succeeds."""
    return {"success": True, "status_code": 200}


def _mock_http_fail(**kwargs):
    """Mock HTTP client that always fails."""
    return {"success": False, "status_code": 500, "error": "Internal Server Error"}


class TestWebhookConfig(unittest.TestCase):
    """Test webhook configuration."""

    def test_defaults(self):
        c = WebhookConfig(url="https://example.com/hook")
        self.assertTrue(c.enabled)
        self.assertEqual(c.format, "generic")
        self.assertIsNone(c.domains)
        self.assertIsNone(c.tiers)
        self.assertEqual(c.max_retries, 2)

    def test_custom_config(self):
        c = WebhookConfig(
            url="https://teams.webhook.office.com/123",
            format="teams",
            domains=["card_dispute"],
            tiers=["gate", "hold"],
        )
        self.assertEqual(c.format, "teams")
        self.assertEqual(c.domains, ["card_dispute"])


class TestNotifySuspension(unittest.TestCase):
    """Test suspension notification dispatch."""

    def test_notifies_all_matching_configs(self):
        calls = []

        def mock_http(url, payload, headers=None, timeout=10):
            calls.append({"url": url, "payload": payload})
            return {"success": True, "status_code": 200}

        configs = [
            WebhookConfig(url="https://hook1.example.com"),
            WebhookConfig(url="https://hook2.example.com"),
        ]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_123",
            workflow="card_dispute",
            domain="card_dispute",
            tier="gate",
            step="classify",
            reason="Needs review",
        )
        time.sleep(0.5)  # Wait for background threads
        self.assertEqual(len(calls), 2)

    def test_domain_filter(self):
        calls = []

        def mock_http(url, payload, headers=None, timeout=10):
            calls.append(url)
            return {"success": True, "status_code": 200}

        configs = [
            WebhookConfig(url="https://dispute-only.example.com", domains=["card_dispute"]),
            WebhookConfig(url="https://all-domains.example.com"),
        ]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_123",
            workflow="product_return",
            domain="electronics_return",  # Not card_dispute
            tier="gate",
        )
        time.sleep(0.5)
        # Only the all-domains hook should fire
        self.assertEqual(len(calls), 1)
        self.assertIn("all-domains", calls[0])

    def test_tier_filter(self):
        calls = []

        def mock_http(url, payload, headers=None, timeout=10):
            calls.append(url)
            return {"success": True, "status_code": 200}

        configs = [
            WebhookConfig(url="https://gate-only.example.com", tiers=["gate"]),
        ]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_123", workflow="w", domain="d",
            tier="spot_check",  # Not gate
        )
        time.sleep(0.5)
        self.assertEqual(len(calls), 0)

    def test_disabled_config_skipped(self):
        calls = []

        def mock_http(url, payload, headers=None, timeout=10):
            calls.append(url)
            return {"success": True, "status_code": 200}

        configs = [
            WebhookConfig(url="https://disabled.example.com", enabled=False),
            WebhookConfig(url="https://enabled.example.com", enabled=True),
        ]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_123", workflow="w", domain="d", tier="gate",
        )
        time.sleep(0.5)
        self.assertEqual(len(calls), 1)
        self.assertIn("enabled", calls[0])

    def test_no_configs_no_error(self):
        notifier = WebhookNotifier(configs=[])
        # Should not raise
        notifier.notify_suspension(
            instance_id="wf_123", workflow="w", domain="d", tier="gate",
        )

    def test_approve_url_generated(self):
        payloads = []

        def mock_http(url, payload, headers=None, timeout=10):
            payloads.append(payload)
            return {"success": True, "status_code": 200}

        configs = [WebhookConfig(url="https://hook.example.com")]
        notifier = WebhookNotifier(
            configs=configs, http_client=mock_http,
            base_url="https://api.example.com",
        )
        notifier.notify_suspension(
            instance_id="wf_abc", workflow="w", domain="d", tier="gate",
        )
        time.sleep(0.5)
        self.assertIn("approve_url", payloads[0])
        self.assertIn("wf_abc", payloads[0]["approve_url"])


class TestDeliveryRetry(unittest.TestCase):
    """Test retry and failure handling."""

    def test_retry_on_failure(self):
        attempt_count = {"n": 0}

        def mock_http(url, payload, headers=None, timeout=10):
            attempt_count["n"] += 1
            if attempt_count["n"] < 2:
                return {"success": False, "error": "timeout"}
            return {"success": True, "status_code": 200}

        configs = [WebhookConfig(url="https://flaky.example.com", max_retries=3)]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_retry", workflow="w", domain="d", tier="gate",
        )
        time.sleep(5)  # Allow retry delays
        self.assertEqual(attempt_count["n"], 2)  # Succeeded on second attempt
        dlv = notifier.deliveries
        self.assertEqual(len(dlv), 1)
        self.assertEqual(dlv[0]["status"], "delivered")

    def test_exhausted_retries(self):
        def mock_http(url, payload, headers=None, timeout=10):
            return {"success": False, "error": "always fails"}

        configs = [WebhookConfig(url="https://dead.example.com", max_retries=2)]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_fail", workflow="w", domain="d", tier="gate",
        )
        time.sleep(8)  # Allow all retries with backoff
        dlv = notifier.deliveries
        self.assertEqual(len(dlv), 1)
        self.assertEqual(dlv[0]["status"], "failed")
        self.assertEqual(dlv[0]["attempts"], 2)

    def test_exception_in_http_client(self):
        def mock_http(url, payload, headers=None, timeout=10):
            raise ConnectionError("network unreachable")

        configs = [WebhookConfig(url="https://crash.example.com", max_retries=1)]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_crash", workflow="w", domain="d", tier="gate",
        )
        time.sleep(2)
        dlv = notifier.deliveries
        self.assertEqual(dlv[0]["status"], "failed")
        self.assertIn("network unreachable", dlv[0]["error"])


class TestPayloadFormatters(unittest.TestCase):
    """Test platform-specific payload formatting."""

    def test_generic_format(self):
        p = _format_payload(
            "generic", "suspension", "wf_1", "card_dispute", "card_dispute",
            "gate", "classify", "needs review", "https://approve.me", None,
        )
        self.assertEqual(p["event_type"], "suspension")
        self.assertEqual(p["instance_id"], "wf_1")
        self.assertIn("approve_url", p)

    def test_teams_format(self):
        p = _format_payload(
            "teams", "suspension", "wf_1", "card_dispute", "card_dispute",
            "gate", "classify", "needs review", "https://approve.me", None,
        )
        self.assertEqual(p["@type"], "MessageCard")
        self.assertIn("sections", p)
        self.assertIn("potentialAction", p)

    def test_slack_format(self):
        p = _format_payload(
            "slack", "suspension", "wf_1", "card_dispute", "card_dispute",
            "gate", "classify", "needs review", "https://approve.me", None,
        )
        self.assertIn("blocks", p)
        # Should have header, section, reason, and action blocks
        self.assertGreaterEqual(len(p["blocks"]), 3)

    def test_generic_without_approve_url(self):
        p = _format_payload(
            "generic", "approval", "wf_1", "w", "d", "", "", "approved by admin", "", None,
        )
        self.assertNotIn("approve_url", p)

    def test_context_included(self):
        p = _format_payload(
            "generic", "suspension", "wf_1", "w", "d", "gate", "s",
            "reason", "", {"extra": "data"},
        )
        self.assertEqual(p["context"]["extra"], "data")


class TestDeliveryRecords(unittest.TestCase):
    """Test delivery record tracking."""

    def test_deliveries_list(self):
        def mock_http(url, payload, headers=None, timeout=10):
            return {"success": True, "status_code": 200}

        configs = [WebhookConfig(url="https://hook.example.com")]
        notifier = WebhookNotifier(configs=configs, http_client=mock_http)
        notifier.notify_suspension(
            instance_id="wf_1", workflow="w", domain="d", tier="gate",
        )
        notifier.notify_suspension(
            instance_id="wf_2", workflow="w", domain="d", tier="gate",
        )
        time.sleep(0.5)
        dlv = notifier.deliveries
        self.assertEqual(len(dlv), 2)
        instance_ids = {d["instance_id"] for d in dlv}
        self.assertEqual(instance_ids, {"wf_1", "wf_2"})


if __name__ == "__main__":
    unittest.main()
