"""
Cognitive Core — SDA Policy Loader (TASK 6)

Loads Sequential Decision Analysis (SDA) policy artifacts from the registry
and serializes them for LLM prompts. The reward specification is read-only —
the LLM cannot modify it.

Tension detection between the policy recommendation and the causal finding from
the preceding investigate step is mandatory when both artifacts are active.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("cognitive_core.analytics.sda_policy")


# ── Policy Context Builder ────────────────────────────────────────────────────

_SDA_PROMPT_BLOCK = """\

=== SDA POLICY CONTEXT (READ-ONLY — DO NOT MODIFY) ===
Artifact: {artifact_name}
Version: {version}
Policy Class: {policy_class}
Decision Horizon: {horizon} steps
Reward Specification Version: {reward_spec_version}

REWARD SPECIFICATION:
{reward_spec_json}

DECISION HORIZON INSTRUCTIONS:
You are operating under a {policy_class} policy with a {horizon}-step horizon.
For each possible decision outcome, estimate the expected value using the
reward specification above.

CAUSAL FINDING FROM PRIOR INVESTIGATE STEP:
{causal_finding}

SDA REASONING INSTRUCTIONS:
Extend your JSON response with these additional fields:
1. policy_class: The policy class name above.
2. policy_version: The artifact version above.
3. reward_specification_version: The reward_spec_version above.
4. decision_horizon: The horizon integer above.
5. expected_value_by_horizon: A dict mapping each horizon step (1..N) to the
   estimated expected value under the optimal action.
6. policy_recommendation: The action the policy recommends given the reward spec
   and causal finding.
7. causal_consistency_check: "consistent" | "inconsistent" | "not_applicable".
   Compare policy_recommendation with causal finding. If they align, "consistent".
   If they conflict, "inconsistent". If no causal finding, "not_applicable".
8. tension_flags: A list of strings describing tensions between the policy
   recommendation and causal finding. Empty list if consistent.
=== END SDA POLICY CONTEXT ===
"""


def build_sda_context_block(
    artifact: dict[str, Any],
    causal_finding: str = "",
) -> str:
    """
    Build the SDA policy context block injected into the think prompt.

    Args:
        artifact: SDA artifact config from the registry
        causal_finding: The `finding` field from the preceding investigate step,
                        or "" if investigate did not run a causal artifact.
    """
    sda_config = artifact.get("sda_config", {})
    reward_spec = sda_config.get("reward_specification", {})

    return _SDA_PROMPT_BLOCK.format(
        artifact_name=artifact.get("artifact_name", ""),
        version=artifact.get("version", "1.0"),
        policy_class=sda_config.get("policy_class", "direct_lookahead"),
        horizon=sda_config.get("horizon", 3),
        reward_spec_version=artifact.get("version", "1.0"),
        reward_spec_json=json.dumps(reward_spec, indent=2),
        causal_finding=causal_finding or "No causal finding available from prior step.",
    )


# ── Tension Detector ──────────────────────────────────────────────────────────

# Keywords that signal divergent recommendations
_APPROVE_SIGNALS = {"approve", "legitimate", "not fraud", "no fraud", "close", "no action"}
_DENY_SIGNALS = {"deny", "block", "reject", "fraud", "confirmed fraud", "escalate", "file sar"}


def detect_tension(
    policy_recommendation: str,
    causal_finding: str,
) -> list[str]:
    """
    Detect tension between the policy recommendation and the causal finding.

    Returns a list of tension descriptions. Empty list if consistent or
    causal_finding is empty (not_applicable case).

    This is a structural check — full tension analysis is done by the LLM
    via the tension_flags field in the output. This function provides a
    deterministic backup for the proof event.
    """
    if not causal_finding or not policy_recommendation:
        return []

    rec_lower = policy_recommendation.lower()
    finding_lower = causal_finding.lower()

    rec_approve = any(s in rec_lower for s in _APPROVE_SIGNALS)
    rec_deny = any(s in rec_lower for s in _DENY_SIGNALS)
    finding_fraud = any(s in finding_lower for s in _DENY_SIGNALS)
    finding_legit = any(s in finding_lower for s in _APPROVE_SIGNALS)

    tensions = []
    if rec_approve and finding_fraud:
        tensions.append(
            "Policy recommends approval but causal finding indicates fraud"
        )
    if rec_deny and finding_legit:
        tensions.append(
            "Policy recommends denial but causal finding indicates legitimate activity"
        )
    return tensions


# ── Causal Finding Extractor ──────────────────────────────────────────────────

def extract_causal_finding(state: dict[str, Any]) -> str:
    """
    Extract the causal finding from the most recent investigate step in the
    workflow state. Returns "" if no investigate step with a causal artifact ran.
    """
    steps = state.get("steps", [])
    for step in reversed(steps):
        if step.get("primitive") == "investigate":
            output = step.get("output", {})
            # Only return finding if a causal artifact was active
            if "activated_paths" in output:
                return output.get("finding", "")
    return ""
