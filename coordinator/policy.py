"""
Cognitive Core — Coordinator Policy Engine

Evaluates governance tiers, delegation conditions, and need matching
against structured workflow output. Purely deterministic — no LLM calls.

The policy engine reads coordinator configuration (YAML) and evaluates
it against workflow instance results. It returns routing decisions that
the runtime coordinator executes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from coordinator.types import (
    GovernanceTier,
    GovernanceTierConfig,
    DelegationPolicy,
    DelegationCondition,
    Capability,
    Contract,
    ContractField,
)


# ─── Default Governance Tiers ────────────────────────────────────────

DEFAULT_TIERS: dict[str, GovernanceTierConfig] = {
    "auto": GovernanceTierConfig(
        tier=GovernanceTier.AUTO,
        hitl="none",
        sample_rate=0.0,
    ),
    "spot_check": GovernanceTierConfig(
        tier=GovernanceTier.SPOT_CHECK,
        hitl="post_completion",
        sample_rate=0.10,
        queue="qa_review",
        sla_seconds=7200,  # 2 hours
    ),
    "gate": GovernanceTierConfig(
        tier=GovernanceTier.GATE,
        hitl="before_act",
        queue="specialist_review",
        sla_seconds=14400,  # 4 hours
    ),
    "hold": GovernanceTierConfig(
        tier=GovernanceTier.HOLD,
        hitl="before_finalize",
        queue="compliance_review",
        sla_seconds=172800,  # 48 hours
    ),
}


# ─── Routing Decisions ───────────────────────────────────────────────

@dataclass
class GovernanceDecision:
    """Result of governance tier evaluation."""
    tier: str
    action: str  # "proceed", "queue_review", "suspend_for_approval"
    queue: str = ""
    reason: str = ""
    sampled: bool = False  # True if this was a random spot-check


@dataclass
class DelegationDecision:
    """Result of delegation policy evaluation."""
    policy_name: str
    target_workflow: str
    target_domain: str
    contract_name: str
    contract_version: int
    inputs: dict[str, Any]
    sla_seconds: float | None = None
    mode: str = "fire_and_forget"
    resume_at_step: str = ""


@dataclass
class NeedMatch:
    """Result of need-based capability matching."""
    need_type: str
    capability: Capability


# ─── Policy Engine ───────────────────────────────────────────────────

class PolicyEngine:
    """
    Evaluates coordinator policies against workflow output.
    Purely deterministic. No LLM calls.
    """

    def __init__(
        self,
        governance_tiers: dict[str, GovernanceTierConfig] | None = None,
        delegation_policies: list[DelegationPolicy] | None = None,
        capabilities: list[Capability] | None = None,
        contracts: dict[str, Contract] | None = None,
        overrides: dict[str, str] | None = None,
    ):
        self.governance_tiers = governance_tiers or dict(DEFAULT_TIERS)
        self.delegation_policies = delegation_policies or []
        self.capabilities = capabilities or []
        self.contracts = contracts or {}
        self.overrides = overrides or {}  # domain → tier override

    # ─── Governance Tier Evaluation ──────────────────────────────────

    def evaluate_governance(
        self,
        domain: str,
        governance_tier: str,
        workflow_result: dict[str, Any],
    ) -> GovernanceDecision:
        """
        Evaluate governance tier for a completed workflow instance.
        Returns a decision: proceed, queue for review, or suspend.
        """
        # Apply overrides (environment-specific)
        effective_tier = self.overrides.get(domain, governance_tier)
        # Wildcard override
        if "*" in self.overrides and domain not in self.overrides:
            effective_tier = self.overrides["*"]

        tier_config = self.governance_tiers.get(
            effective_tier,
            self.governance_tiers.get("gate"),  # safe default
        )

        if tier_config.tier == GovernanceTier.AUTO:
            return GovernanceDecision(
                tier=effective_tier,
                action="proceed",
                reason="Auto-proceed: low risk, validated use case",
            )

        if tier_config.tier == GovernanceTier.SPOT_CHECK:
            if random.random() < tier_config.sample_rate:
                return GovernanceDecision(
                    tier=effective_tier,
                    action="queue_review",
                    queue=tier_config.queue,
                    reason=f"Spot-check: randomly sampled ({tier_config.sample_rate:.0%} rate)",
                    sampled=True,
                )
            return GovernanceDecision(
                tier=effective_tier,
                action="proceed",
                reason=f"Spot-check: not sampled this time ({tier_config.sample_rate:.0%} rate)",
            )

        if tier_config.tier == GovernanceTier.GATE:
            return GovernanceDecision(
                tier=effective_tier,
                action="suspend_for_approval",
                queue=tier_config.queue,
                reason="Gate: mandatory pre-action review required",
            )

        if tier_config.tier == GovernanceTier.HOLD:
            return GovernanceDecision(
                tier=effective_tier,
                action="suspend_for_approval",
                queue=tier_config.queue,
                reason="Hold: mandatory expert sign-off before finalization",
            )

        # Unknown tier → safe default
        return GovernanceDecision(
            tier=effective_tier,
            action="suspend_for_approval",
            queue="unknown_tier_review",
            reason=f"Unknown governance tier '{effective_tier}', defaulting to gate",
        )

    # ─── Delegation Policy Evaluation ────────────────────────────────

    def evaluate_delegations(
        self,
        domain: str,
        workflow_output: dict[str, Any],
        lineage: list[str] | None = None,
    ) -> list[DelegationDecision]:
        """
        Evaluate all delegation policies against a completed workflow's output.
        Returns a list of delegations to execute (may be empty).
        """
        decisions = []
        steps = workflow_output.get("steps", [])

        for policy in self.delegation_policies:
            # Check if policy applies to this domain
            if not self._domain_matches(policy, domain):
                continue

            # Evaluate all conditions (must all match)
            if not self._conditions_match(policy.conditions, steps):
                continue

            # Cycle detection: check lineage
            if lineage and self._would_cycle(policy, lineage):
                continue

            # Resolve input mapping
            inputs = self._resolve_inputs(policy.input_mapping, workflow_output)

            decisions.append(DelegationDecision(
                policy_name=policy.name,
                target_workflow=policy.target_workflow,
                target_domain=policy.target_domain,
                contract_name=policy.contract_name,
                contract_version=policy.contract_version,
                inputs=inputs,
                sla_seconds=policy.sla_seconds,
                mode=policy.mode,
                resume_at_step=policy.resume_at_step,
            ))

        return decisions

    # ─── Need-Based Capability Matching ──────────────────────────────

    def match_needs(
        self,
        unresolved_needs: list[dict[str, Any]],
    ) -> list[NeedMatch]:
        """
        Match unresolved needs against the capability catalog.
        Returns matches for needs that have providers.
        """
        matches = []
        cap_by_need = {c.need_type: c for c in self.capabilities}

        for need in unresolved_needs:
            need_type = need.get("type", "")
            if need_type in cap_by_need:
                matches.append(NeedMatch(
                    need_type=need_type,
                    capability=cap_by_need[need_type],
                ))

        return matches

    # ─── Contract Validation ─────────────────────────────────────────

    def validate_work_order_inputs(
        self,
        contract_name: str,
        inputs: dict[str, Any],
    ) -> list[str]:
        """Validate work order inputs against contract schema."""
        contract = self.contracts.get(contract_name)
        if not contract:
            return [f"Unknown contract: {contract_name}"]
        return contract.validate_request(inputs)

    def validate_work_order_result(
        self,
        contract_name: str,
        outputs: dict[str, Any],
    ) -> list[str]:
        """Validate work order result against contract schema."""
        contract = self.contracts.get(contract_name)
        if not contract:
            return [f"Unknown contract: {contract_name}"]
        return contract.validate_response(outputs)

    # ─── Internal Helpers ────────────────────────────────────────────

    def _domain_matches(self, policy: DelegationPolicy, domain: str) -> bool:
        """Check if any condition targets this domain."""
        for cond in policy.conditions:
            if cond.domain == domain or cond.domain == "*":
                return True
        return False

    def _conditions_match(
        self,
        conditions: list[DelegationCondition],
        steps: list[dict[str, Any]],
    ) -> bool:
        """Evaluate all conditions against workflow steps. All must match."""
        for cond in conditions:
            if not self._single_condition_matches(cond, steps):
                return False
        return True

    def _single_condition_matches(
        self,
        cond: DelegationCondition,
        steps: list[dict[str, Any]],
    ) -> bool:
        """
        Evaluate a single condition using primitive-type selectors.
        Selectors scan steps by primitive type, not by step name.
        """
        selector = cond.selector
        # Parse selector: any_<prim>, last_<prim>, all_<prim>, final_output
        if selector.startswith("any_"):
            prim_type = selector[4:]
            matching_steps = [s for s in steps if s.get("primitive") == prim_type]
            return any(
                self._field_matches(s.get("output", {}), cond)
                for s in matching_steps
            )

        elif selector.startswith("last_"):
            prim_type = selector[5:]
            matching_steps = [s for s in steps if s.get("primitive") == prim_type]
            if not matching_steps:
                return False
            return self._field_matches(matching_steps[-1].get("output", {}), cond)

        elif selector.startswith("all_"):
            prim_type = selector[4:]
            matching_steps = [s for s in steps if s.get("primitive") == prim_type]
            if not matching_steps:
                return False
            return all(
                self._field_matches(s.get("output", {}), cond)
                for s in matching_steps
            )

        elif selector == "final_output":
            if steps:
                return self._field_matches(steps[-1].get("output", {}), cond)
            return False

        return False

    def _field_matches(self, output: dict[str, Any], cond: DelegationCondition) -> bool:
        """Evaluate a field condition against a step output."""
        if cond.operator == "exists":
            if not cond.field:
                return True  # just checking the step exists
            return self._get_nested(output, cond.field) is not None

        val = self._get_nested(output, cond.field)
        if val is None:
            return False

        if cond.operator == "eq":
            return val == cond.value

        if cond.operator == "gte":
            try:
                return float(val) >= float(cond.value)
            except (ValueError, TypeError):
                return False

        if cond.operator == "contains_any":
            if isinstance(val, (list, set)):
                target = set(cond.value) if isinstance(cond.value, list) else {cond.value}
                return bool(set(val) & target)
            if isinstance(val, str):
                target = cond.value if isinstance(cond.value, list) else [cond.value]
                return any(t in val for t in target)
            return False

        return False

    def _get_nested(self, obj: dict[str, Any], path: str) -> Any:
        """Navigate a dot-separated path into a nested dict."""
        if not path:
            return obj
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return None
            else:
                return None
        return current

    def _would_cycle(
        self,
        policy: DelegationPolicy,
        lineage: list[str],
    ) -> bool:
        """
        Check if executing this policy would create a cycle.
        Simple heuristic: if the target workflow type already
        appears in the lineage, skip.
        """
        # Lineage entries are instance IDs; we'd need to look up
        # their workflow types. For Phase 1, we store workflow_type
        # in lineage as "type:instance_id" format.
        target = policy.target_workflow
        for entry in lineage:
            if entry.startswith(f"{target}:"):
                return True
        return False

    def _resolve_inputs(
        self,
        mapping: dict[str, str],
        workflow_output: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve ${source.xxx} references in input mapping.
        Uses the same primitive-type selector pattern as conditions.
        """
        resolved = {}
        steps = workflow_output.get("steps", [])
        input_data = workflow_output.get("input", {})

        for target_field, source_ref in mapping.items():
            if not source_ref.startswith("${source."):
                resolved[target_field] = source_ref
                continue

            # Strip ${source. and trailing }
            path = source_ref[len("${source."):-1]
            resolved[target_field] = self._resolve_source_path(
                path, steps, input_data
            )

        return resolved

    def _resolve_source_path(
        self,
        path: str,
        steps: list[dict[str, Any]],
        input_data: dict[str, Any],
    ) -> Any:
        """
        Resolve a source path like 'last_retrieve.data.get_member.member_id'
        or 'input.member_id' against workflow output.
        """
        parts = path.split(".", 1)
        selector = parts[0]
        remaining = parts[1] if len(parts) > 1 else ""

        if selector == "input":
            return self._get_nested(input_data, remaining)

        # Primitive-type selectors
        if selector.startswith("last_"):
            prim_type = selector[5:]
            matching = [s for s in steps if s.get("primitive") == prim_type]
            if matching:
                output = matching[-1].get("output", {})
                return self._get_nested(output, remaining) if remaining else output

        if selector.startswith("any_"):
            prim_type = selector[4:]
            matching = [s for s in steps if s.get("primitive") == prim_type]
            # Return first match that has the field
            for s in matching:
                val = self._get_nested(s.get("output", {}), remaining)
                if val is not None:
                    return val

        return None


# ─── Configuration Loader ────────────────────────────────────────────

def load_policy_engine(config: dict[str, Any]) -> PolicyEngine:
    """
    Build a PolicyEngine from coordinator YAML config.

    Expected structure:
        governance_tiers:
          auto: {hitl: none, sample_rate: 0}
          spot_check: {hitl: post_completion, sample_rate: 0.10, queue: qa_review, sla: 7200}
          gate: {hitl: before_act, queue: specialist_review, sla: 14400}
          hold: {hitl: before_finalize, queue: compliance_review, sla: 172800}

        overrides:
          "*": gate   # everything gated by default
          debit_spending: auto

        delegations:
          - name: fraud_triggers_aml
            conditions:
              - domain: card_dispute
                selector: any_investigate
                field: evidence_flags
                operator: contains_any
                value: [foreign_ip, unknown_device]
            target_workflow: sar_investigation
            target_domain: card_fraud_referral
            contract: aml_referral_v1
            sla: 86400
            inputs:
              member_id: "${source.last_retrieve.data.get_member.member_id}"

        contracts:
          aml_referral_v1:
            version: 1
            request:
              - {name: member_id, type: string, required: true}
              - {name: referral_reason, type: string, required: true}
            response:
              - {name: filing_decision, type: enum, required: true,
                 enum_values: [file_sar, close_no_action, request_more_info]}

        capabilities:
          - need: industry_benchmarks
            type: workflow
            workflow: data_enrichment
            domain: industry_analysis
            contract: benchmarks_v1
    """
    # Governance tiers
    tiers = dict(DEFAULT_TIERS)
    for name, cfg in config.get("governance_tiers", {}).items():
        tiers[name] = GovernanceTierConfig(
            tier=GovernanceTier(name) if name in GovernanceTier.__members__.values() else GovernanceTier.GATE,
            hitl=cfg.get("hitl", "none"),
            sample_rate=cfg.get("sample_rate", 0.0),
            queue=cfg.get("queue", ""),
            sla_seconds=cfg.get("sla", 0),
        )

    # Overrides
    overrides = config.get("overrides", {})

    # Delegation policies
    delegations = []
    for d in config.get("delegations", []):
        conditions = []
        for c in d.get("conditions", []):
            conditions.append(DelegationCondition(
                domain=c.get("domain", "*"),
                selector=c.get("selector", "any_investigate"),
                field=c.get("field", ""),
                operator=c.get("operator", "exists"),
                value=c.get("value"),
            ))
        delegations.append(DelegationPolicy(
            name=d.get("name", "unnamed"),
            conditions=conditions,
            target_workflow=d.get("target_workflow", ""),
            target_domain=d.get("target_domain", ""),
            contract_name=d.get("contract", ""),
            contract_version=d.get("contract_version", 1),
            sla_seconds=d.get("sla"),
            mode=d.get("mode", "fire_and_forget"),
            resume_at_step=d.get("resume_at_step", ""),
            input_mapping=d.get("inputs", {}),
        ))

    # Contracts
    contracts = {}
    for name, c in config.get("contracts", {}).items():
        req_fields = [
            ContractField(
                name=f["name"], type=f.get("type", "string"),
                required=f.get("required", True),
                enum_values=f.get("enum_values"),
            )
            for f in c.get("request", [])
        ]
        resp_fields = [
            ContractField(
                name=f["name"], type=f.get("type", "string"),
                required=f.get("required", True),
                enum_values=f.get("enum_values"),
            )
            for f in c.get("response", [])
        ]
        contracts[name] = Contract(
            name=name,
            version=c.get("version", 1),
            request_fields=req_fields,
            response_fields=resp_fields,
        )

    # Capabilities
    capabilities = []
    for cap in config.get("capabilities", []):
        capabilities.append(Capability(
            need_type=cap.get("need", ""),
            provider_type=cap.get("type", "workflow"),
            workflow_type=cap.get("workflow", ""),
            domain=cap.get("domain", ""),
            contract_name=cap.get("contract", ""),
            queue=cap.get("queue", ""),
            endpoint=cap.get("endpoint", ""),
        ))

    return PolicyEngine(
        governance_tiers=tiers,
        delegation_policies=delegations,
        capabilities=capabilities,
        contracts=contracts,
        overrides=overrides,
    )
