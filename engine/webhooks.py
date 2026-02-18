"""
Cognitive Core — Webhook Notifications (S-014)

Sends notifications when workflow instances are suspended for approval.
Configurable targets: Teams incoming webhook, ServiceNow, or custom HTTP endpoint.

Features:
  - Fire-and-forget delivery (non-blocking)
  - Configurable per domain/tier
  - Retry with exponential backoff (optional)
  - Delivery logging for diagnostics
  - Approval link generation

Usage:
    from engine.webhooks import WebhookNotifier, WebhookConfig

    config = WebhookConfig(
        url="https://teams.webhook.office.com/...",
        format="teams",
    )
    notifier = WebhookNotifier(configs=[config])
    notifier.notify_suspension(
        instance_id="wf_abc",
        workflow="card_dispute",
        domain="card_dispute",
        tier="gate",
        step="classify",
        reason="High-value dispute requires review",
        approve_url="https://api.example.com/v1/approvals/wf_abc/approve",
    )
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cognitive_core.webhooks")


@dataclass
class WebhookConfig:
    """Configuration for a single webhook target."""
    url: str
    format: str = "generic"     # generic, teams, slack, servicenow
    enabled: bool = True
    domains: list[str] | None = None    # None = all domains
    tiers: list[str] | None = None      # None = all tiers
    max_retries: int = 2
    timeout_seconds: float = 10.0
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class DeliveryRecord:
    """Record of a webhook delivery attempt."""
    delivery_id: str
    webhook_url: str
    instance_id: str
    event_type: str
    status: str         # pending, delivered, failed
    attempts: int = 0
    last_attempt_at: float = 0.0
    error: str = ""
    created_at: float = 0.0


class WebhookNotifier:
    """
    Non-blocking webhook notification sender.

    Sends notifications in background threads.
    Supports multiple webhook targets with domain/tier filtering.
    """

    def __init__(
        self,
        configs: list[WebhookConfig] | None = None,
        http_client: Callable | None = None,
        base_url: str = "",
    ):
        self.configs = configs or []
        self._http_client = http_client or _default_http_client
        self.base_url = base_url.rstrip("/")
        self._deliveries: list[DeliveryRecord] = []
        self._lock = threading.Lock()

    def notify_suspension(
        self,
        instance_id: str,
        workflow: str,
        domain: str,
        tier: str,
        step: str = "",
        reason: str = "",
        approve_url: str = "",
        context: dict[str, Any] | None = None,
    ):
        """
        Notify all matching webhook targets of a workflow suspension.
        Non-blocking — dispatches to background threads.
        """
        if not self.configs:
            return

        # Build approve URL if not provided
        if not approve_url and self.base_url:
            approve_url = f"{self.base_url}/v1/approvals/{instance_id}/approve"

        for config in self.configs:
            if not config.enabled:
                continue
            if config.domains and domain not in config.domains:
                continue
            if config.tiers and tier not in config.tiers:
                continue

            # Format the payload
            payload = _format_payload(
                config.format,
                event_type="suspension",
                instance_id=instance_id,
                workflow=workflow,
                domain=domain,
                tier=tier,
                step=step,
                reason=reason,
                approve_url=approve_url,
                context=context,
            )

            # Dispatch in background thread
            record = DeliveryRecord(
                delivery_id=f"dlv_{uuid.uuid4().hex[:12]}",
                webhook_url=config.url,
                instance_id=instance_id,
                event_type="suspension",
                status="pending",
                created_at=time.time(),
            )
            with self._lock:
                self._deliveries.append(record)

            thread = threading.Thread(
                target=self._deliver,
                args=(config, payload, record),
                daemon=True,
            )
            thread.start()

    def notify_approval(
        self,
        instance_id: str,
        workflow: str,
        domain: str,
        approver: str,
        action: str = "approved",
    ):
        """Notify webhook targets of an approval/rejection."""
        for config in self.configs:
            if not config.enabled:
                continue

            payload = _format_payload(
                config.format,
                event_type="approval",
                instance_id=instance_id,
                workflow=workflow,
                domain=domain,
                tier="",
                step="",
                reason=f"{action} by {approver}",
                approve_url="",
                context={"approver": approver, "action": action},
            )

            record = DeliveryRecord(
                delivery_id=f"dlv_{uuid.uuid4().hex[:12]}",
                webhook_url=config.url,
                instance_id=instance_id,
                event_type="approval",
                status="pending",
                created_at=time.time(),
            )
            with self._lock:
                self._deliveries.append(record)

            thread = threading.Thread(
                target=self._deliver,
                args=(config, payload, record),
                daemon=True,
            )
            thread.start()

    def _deliver(self, config: WebhookConfig, payload: dict, record: DeliveryRecord):
        """Deliver a webhook with retry."""
        for attempt in range(1, config.max_retries + 1):
            record.attempts = attempt
            record.last_attempt_at = time.time()

            try:
                response = self._http_client(
                    url=config.url,
                    payload=payload,
                    headers=config.headers,
                    timeout=config.timeout_seconds,
                )
                if response.get("success"):
                    record.status = "delivered"
                    logger.info(
                        "Webhook delivered: %s → %s (attempt %d)",
                        record.delivery_id, config.url[:50], attempt,
                    )
                    return
                else:
                    record.error = response.get("error", "unknown error")
                    logger.warning(
                        "Webhook failed: %s → %s: %s (attempt %d/%d)",
                        record.delivery_id, config.url[:50], record.error,
                        attempt, config.max_retries,
                    )
            except Exception as e:
                record.error = str(e)[:200]
                logger.warning(
                    "Webhook error: %s → %s: %s (attempt %d/%d)",
                    record.delivery_id, config.url[:50], e,
                    attempt, config.max_retries,
                )

            # Exponential backoff
            if attempt < config.max_retries:
                time.sleep(min(2 ** attempt, 10))

        record.status = "failed"
        logger.error(
            "Webhook exhausted retries: %s → %s after %d attempts",
            record.delivery_id, config.url[:50], config.max_retries,
        )

    @property
    def deliveries(self) -> list[dict[str, Any]]:
        """Get delivery records for diagnostics."""
        with self._lock:
            return [
                {
                    "delivery_id": d.delivery_id,
                    "webhook_url": d.webhook_url[:50],
                    "instance_id": d.instance_id,
                    "event_type": d.event_type,
                    "status": d.status,
                    "attempts": d.attempts,
                    "error": d.error,
                }
                for d in self._deliveries[-100:]  # Last 100
            ]


# ═══════════════════════════════════════════════════════════════════
# Payload Formatters
# ═══════════════════════════════════════════════════════════════════

def _format_payload(
    fmt: str,
    event_type: str,
    instance_id: str,
    workflow: str,
    domain: str,
    tier: str,
    step: str,
    reason: str,
    approve_url: str,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Format webhook payload for the target platform."""

    if fmt == "teams":
        return _format_teams(
            event_type, instance_id, workflow, domain, tier,
            step, reason, approve_url, context,
        )
    elif fmt == "slack":
        return _format_slack(
            event_type, instance_id, workflow, domain, tier,
            step, reason, approve_url, context,
        )
    else:
        return _format_generic(
            event_type, instance_id, workflow, domain, tier,
            step, reason, approve_url, context,
        )


def _format_generic(event_type, instance_id, workflow, domain, tier,
                     step, reason, approve_url, context):
    payload = {
        "event_type": event_type,
        "instance_id": instance_id,
        "workflow": workflow,
        "domain": domain,
        "tier": tier,
        "step": step,
        "reason": reason,
        "timestamp": time.time(),
    }
    if approve_url:
        payload["approve_url"] = approve_url
    if context:
        payload["context"] = context
    return payload


def _format_teams(event_type, instance_id, workflow, domain, tier,
                   step, reason, approve_url, context):
    """Microsoft Teams Adaptive Card format."""
    title = f"🔔 Cognitive Core: {event_type.title()}"
    facts = [
        {"name": "Instance", "value": instance_id},
        {"name": "Workflow", "value": workflow},
        {"name": "Domain", "value": domain},
    ]
    if tier:
        facts.append({"name": "Tier", "value": tier})
    if step:
        facts.append({"name": "Step", "value": step})
    if reason:
        facts.append({"name": "Reason", "value": reason})

    card = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": title,
        "themeColor": "D63B00" if event_type == "suspension" else "2DC72D",
        "title": title,
        "sections": [{"facts": facts}],
    }
    if approve_url:
        card["potentialAction"] = [{
            "@type": "OpenUri",
            "name": "Approve",
            "targets": [{"os": "default", "uri": approve_url}],
        }]
    return card


def _format_slack(event_type, instance_id, workflow, domain, tier,
                   step, reason, approve_url, context):
    """Slack Block Kit format."""
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Cognitive Core: {event_type.title()}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Instance:* {instance_id}"},
                {"type": "mrkdwn", "text": f"*Workflow:* {workflow}"},
                {"type": "mrkdwn", "text": f"*Domain:* {domain}"},
                {"type": "mrkdwn", "text": f"*Tier:* {tier}"},
            ],
        },
    ]
    if reason:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Reason:* {reason}"},
        })
    if approve_url:
        blocks.append({
            "type": "actions",
            "elements": [{
                "type": "button",
                "text": {"type": "plain_text", "text": "Approve"},
                "url": approve_url,
                "style": "primary",
            }],
        })
    return {"blocks": blocks}


# ═══════════════════════════════════════════════════════════════════
# Default HTTP Client
# ═══════════════════════════════════════════════════════════════════

def _default_http_client(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """
    Default HTTP POST client using urllib.
    Returns {"success": bool, "status_code": int, "error": str}.
    """
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json"}
    if headers:
        all_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=all_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"success": resp.status < 400, "status_code": resp.status}
    except urllib.error.HTTPError as e:
        return {"success": False, "status_code": e.code, "error": str(e)}
    except Exception as e:
        return {"success": False, "status_code": 0, "error": str(e)}
