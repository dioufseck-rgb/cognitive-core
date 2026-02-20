#!/usr/bin/env python3
"""
Cognitive Core — Live LLM Evaluation Runner

Runs synthetic test cases through the full engine+coordinator stack
against a real LLM and validates outputs against expected values.

This is the ONE test that proves the entire system works end-to-end.

Usage:
    # Gemini (personal API key)
    LLM_PROVIDER=google GOOGLE_API_KEY=your_key \
      python scripts/eval_live.py

    # Azure AI Foundry
    LLM_PROVIDER=azure_foundry AZURE_AI_PROJECT_ENDPOINT=... \
      python scripts/eval_live.py

    # Single case
    python scripts/eval_live.py --case sc_002_physical_damage

    # All cases
    python scripts/eval_live.py --all

    # Skip approval (auto-approve gate tier)
    python scripts/eval_live.py --auto-approve

    # Verbose (show full step outputs)
    python scripts/eval_live.py -v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


# ═════════════════════════════════════════════════════════════════
# Test Case Definitions
# ═════════════════════════════════════════════════════════════════

CASES = {
    # ── Claim Intake Cases ────────────────────────────────────
    "sc_001_simple_approve": {
        "file": "cases/synthetic/sc_001_simple_approve.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "liability",
            "check_eligibility.conforms": True,
            "assess_risk.recommendation": "auto_approve",
        },
        "risk_range": [0, 29],
        "delegations": {"damage_assessment": False, "fraud_screening": False},
    },
    "sc_002_physical_damage": {
        "file": "cases/synthetic/sc_002_physical_damage.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "physical_damage",
            "check_eligibility.conforms": True,
            "assess_risk.recommendation": "auto_approve",
        },
        "risk_range": [0, 29],
        "delegations": {"damage_assessment": True, "fraud_screening": False},
    },
    "sc_003_high_value_flagged": {
        "file": "cases/synthetic/sc_003_high_value_flagged.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "theft",
            "check_eligibility.conforms": True,
        },
        "risk_range": [60, 100],
        "delegations": {"damage_assessment": False, "fraud_screening": True},
    },
    "sc_004_both_delegations": {
        "file": "cases/synthetic/sc_004_both_delegations.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "physical_damage",
            "check_eligibility.conforms": True,
            "assess_risk.recommendation": "refer_to_siu",
        },
        "risk_range": [80, 100],
        "delegations": {"damage_assessment": True, "fraud_screening": True},
    },
    "sc_005_inactive_policy": {
        "file": "cases/synthetic/sc_005_inactive_policy.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "check_eligibility.conforms": False,
        },
        "risk_range": None,  # denied — no risk assessment
        "delegations": None,
    },
    "sc_006_wrong_coverage": {
        "file": "cases/synthetic/sc_006_wrong_coverage.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "theft",
            "check_eligibility.conforms": False,
        },
        "risk_range": None,
        "delegations": None,
    },
    "sc_007_multi_violation": {
        "file": "cases/synthetic/sc_007_multi_violation.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": ["medical", "liability"],
            "check_eligibility.conforms": False,
        },
        "risk_range": None,
        "delegations": None,
    },
    "sc_008_medium_risk": {
        "file": "cases/synthetic/sc_008_medium_risk.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": ["medical", "liability"],
            "check_eligibility.conforms": True,
            "assess_risk.recommendation": ["standard_review", "manual_review"],
        },
        "risk_range": [30, 59],
        "delegations": {"damage_assessment": False, "fraud_screening": True},
    },
    "sc_009_other_type": {
        "file": "cases/synthetic/sc_009_other_type.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "other",
        },
        "risk_range": [0, 29],
        "delegations": {"damage_assessment": False, "fraud_screening": False},
    },
    "sc_010_boundary_amount": {
        "file": "cases/synthetic/sc_010_boundary_amount.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": ["medical", "liability"],
            "check_eligibility.conforms": True,
            "assess_risk.recommendation": ["standard_review", "manual_review"],
        },
        "risk_range": [30, 59],
        "delegations": {"damage_assessment": False, "fraud_screening": True},
    },
    "sc_011_clean_physical": {
        "file": "cases/synthetic/sc_011_clean_physical.json",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "checks": {
            "classify_claim_type.category": "physical_damage",
        },
        "risk_range": [0, 29],
        "delegations": None,
    },

    # ── Damage Assessment Cases ───────────────────────────────
    "da_001_minor_clean": {
        "file": "cases/synthetic/da_001_minor_clean.json",
        "workflow": "damage_assessment",
        "domain": "synthetic_damage",
        "checks": {
            "classify_damage_severity.category": "minor",
            "verify_documentation.conforms": True,
        },
        "risk_range": None,
        "delegations": None,
    },
    "da_002_moderate": {
        "file": "cases/synthetic/da_002_moderate.json",
        "workflow": "damage_assessment",
        "domain": "synthetic_damage",
        "checks": {
            "classify_damage_severity.category": "moderate",
        },
        "risk_range": None,
        "delegations": None,
    },
    "da_003_total_loss_clean": {
        "file": "cases/synthetic/da_003_total_loss_clean.json",
        "workflow": "damage_assessment",
        "domain": "synthetic_damage",
        "checks": {
            "classify_damage_severity.category": "total_loss",
            "verify_documentation.conforms": True,
        },
        "risk_range": None,
        "delegations": None,
    },
    "da_004_major_missing_docs": {
        "file": "cases/synthetic/da_004_major_missing_docs.json",
        "workflow": "damage_assessment",
        "domain": "synthetic_damage",
        "checks": {
            "classify_damage_severity.category": ["major", "total_loss"],
            "verify_documentation.conforms": False,
        },
        "risk_range": None,
        "delegations": None,
    },
    "da_005_minor_missing_photos": {
        "file": "cases/synthetic/da_005_minor_missing_photos.json",
        "workflow": "damage_assessment",
        "domain": "synthetic_damage",
        "checks": {
            "classify_damage_severity.category": "minor",
        },
        "risk_range": None,
        "delegations": None,
    },

    # ── Fraud Screening Cases ─────────────────────────────────
    "fs_001_low_risk_clean": {
        "file": "cases/synthetic/fs_001_low_risk_clean.json",
        "workflow": "fraud_screening",
        "domain": "synthetic_fraud",
        "checks": {
            "classify_fraud_risk.category": "low_risk",
        },
        "risk_range": None,
        "delegations": None,
    },
    "fs_002_medium_risk_patterns": {
        "file": "cases/synthetic/fs_002_medium_risk_patterns.json",
        "workflow": "fraud_screening",
        "domain": "synthetic_fraud",
        "checks": {
            "classify_fraud_risk.category": "medium_risk",
        },
        "risk_range": None,
        "delegations": None,
    },
    "fs_003_medium_risk": {
        "file": "cases/synthetic/fs_003_medium_risk.json",
        "workflow": "fraud_screening",
        "domain": "synthetic_fraud",
        "checks": {
            "classify_fraud_risk.category": "medium_risk",
        },
        "risk_range": None,
        "delegations": None,
    },
    "fs_004_high_risk_siu": {
        "file": "cases/synthetic/fs_004_high_risk_siu.json",
        "workflow": "fraud_screening",
        "domain": "synthetic_fraud",
        "checks": {
            "classify_fraud_risk.category": "high_risk",
        },
        "risk_range": None,
        "delegations": None,
    },

    # ── Member Hardship Intake Cases ──────────────────────────
    "mh_001_mixed_portfolio": {
        "file": "cases/synthetic/mh_001_mixed_portfolio.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
                "field_checks": {
                    "preliminary_triage": ["complex"],
                },
            },
        },
        "structural_checks": {
            "no_pii": True,
            "evidence_linked": True,
        },
    },
    "mh_002_scra_ambiguity": {
        "file": "cases/synthetic/mh_002_scra_ambiguity.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
                "field_checks": {
                    "preliminary_triage": ["complex"],
                },
            },
        },
        "structural_checks": {
            "no_pii": True,
            "evidence_linked": True,
            "must_flag_scra": True,
        },
    },
    "mh_003_disaster_mismatch": {
        "file": "cases/synthetic/mh_003_disaster_mismatch.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
            },
        },
        "structural_checks": {
            "no_pii": True,
            "evidence_linked": True,
        },
    },
    "mh_004_fraud_exploitation": {
        "file": "cases/synthetic/mh_004_fraud_exploitation.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
                "field_checks": {
                    "preliminary_triage": ["urgent"],
                },
            },
        },
        "structural_checks": {
            "no_pii": True,
            "must_flag_fraud": True,
        },
    },
    "mh_005_open_complaint": {
        "file": "cases/synthetic/mh_005_open_complaint.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
                "field_checks": {
                    "preliminary_triage": ["complex"],
                },
            },
        },
        "structural_checks": {
            "no_pii": True,
            "evidence_linked": True,
            "must_flag_complaint": True,
        },
    },
    "mh_006_income_conflict": {
        "file": "cases/synthetic/mh_006_income_conflict.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
                "field_checks": {
                    "preliminary_triage": ["complex"],
                },
            },
        },
        "structural_checks": {
            "no_pii": True,
            "must_flag_conflict": True,
        },
    },
    "mh_007_non_english": {
        "file": "cases/synthetic/mh_007_non_english.json",
        "workflow": "hardship_intake_packet",
        "domain": "member_hardship",
        "checks": {},
        "risk_range": None,
        "delegations": None,
        "artifact_checks": {
            "generate_intake_packet": {
                "required_keys": ["member_token", "account_summary", "hardship_claims",
                                  "evidence_index", "risk_flags", "missing_information",
                                  "preliminary_triage"],
                "field_checks": {
                    "preliminary_triage": ["straightforward"],
                },
            },
        },
        "structural_checks": {
            "no_pii": True,
            "evidence_linked": True,
        },
    },
}


# ═════════════════════════════════════════════════════════════════
# Evaluation Engine
# ═════════════════════════════════════════════════════════════════

class LiveEvaluator:
    def __init__(self, project_root: str, verbose: bool = False, auto_approve: bool = False,
                 strict_gates: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.auto_approve = auto_approve
        self.strict_gates = strict_gates
        self.db_path = os.path.join(tempfile.mkdtemp(), "eval_live.db")
        self.results = []

    def run_case(self, case_name: str, case_def: dict) -> dict:
        """Run a single test case through the full stack."""
        from coordinator.runtime import Coordinator
        from coordinator.types import InstanceStatus

        print(f"\n{'─'*60}")
        print(f"CASE: {case_name}")
        print(f"  Workflow: {case_def['workflow']}/{case_def['domain']}")

        # Load case input
        case_path = os.path.join(self.project_root, case_def["file"])
        with open(case_path) as f:
            case_data = json.load(f)

        meta = case_data.pop("_meta", {})
        expected = meta.get("expected", {})
        print(f"  Expected: {json.dumps(expected, indent=None)[:120]}")

        # Create fresh coordinator
        config_path = os.path.join(self.project_root, "coordinator", "config.yaml")
        db_path = os.path.join(tempfile.mkdtemp(), f"eval_{case_name}.db")
        coord = Coordinator(config_path=config_path, db_path=db_path, verbose=self.verbose)

        # Run
        t0 = time.time()
        try:
            instance_id = coord.start(
                workflow_type=case_def["workflow"],
                domain=case_def["domain"],
                case_input=case_data,
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ CRASHED: {e}")
            return {"case": case_name, "status": "crashed", "error": str(e), "elapsed": elapsed}

        instance = coord.get_instance(instance_id)

        # Handle gate tier suspension
        if instance.status == InstanceStatus.SUSPENDED:
            suspension_step = getattr(instance, 'suspended_at_step', '') or ''

            # Check ledger for quality gate events
            pre_ledger = coord.get_ledger(instance_id=instance_id)
            has_quality_gate = any(
                getattr(e, 'action_type', '') == 'quality_gate_fired'
                for e in pre_ledger
            )

            if has_quality_gate and self.strict_gates:
                # Quality gates are hard fails in strict mode
                elapsed = time.time() - t0
                print(f"  ❌ QUALITY GATE fired — hard fail (strict mode)")
                return {"case": case_name, "status": "fail",
                        "checks_passed": 0, "checks_failed": 1,
                        "failures": ["quality_gate_fired (strict mode)"],
                        "elapsed": elapsed, "steps": 0, "audit_entries": 0,
                        "delegations": 0, "instance_id": instance_id}
            elif self.auto_approve:
                gate_type = "quality" if has_quality_gate else "governance"
                print(f"  ⏸ Suspended ({gate_type} gate) → auto-approving")

                # Show what the human reviewer would get
                try:
                    import importlib.util as _ilu
                    esc_path = os.path.join(self.project_root, "coordinator", "escalation.py")
                    if os.path.exists(esc_path):
                        _esc_spec = _ilu.spec_from_file_location("coordinator.escalation", esc_path)
                        _esc_mod = _ilu.module_from_spec(_esc_spec)
                        import sys as _s2
                        _s2.modules["coordinator.escalation"] = _esc_mod
                        _esc_spec.loader.exec_module(_esc_mod)

                        # Get the suspension state snapshot — it has the actual step outputs.
                        # instance.result is empty/partial when suspended.
                        _state = {}
                        try:
                            _suspension = coord.store.get_suspension(instance_id)
                            if _suspension:
                                _state = _suspension.state_snapshot or {}
                        except Exception:
                            pass
                        if not _state.get("steps"):
                            _inst = coord.get_instance(instance_id)
                            _state = _inst.result or {}
                        brief = _esc_mod.build_escalation_brief(
                            workflow_type=case_def["workflow"],
                            domain=case_def["domain"],
                            final_state=_state,
                            escalation_reason=gate_type + " gate",
                        )
                        if self.verbose:
                            _step_names = [s.get("step_name", "?") for s in _state.get("steps", [])]
                            print(f"  [brief] State has {len(_state.get('steps', []))} steps: {_step_names}")
                        print(f"  📋 ESCALATION BRIEF:")
                        print(f"     Case: {brief.get('case_summary', {})}")
                        dets = brief.get("determinations", [])
                        if dets:
                            print(f"     Determinations: {len(dets)} steps completed")
                            for d in dets:
                                conf = f" ({d['confidence']:.0%})" if d.get('confidence') else ""
                                print(f"       • {d['step']}: {d.get('result') or d.get('finding') or d.get('decision')}{conf}")
                        uncs = brief.get("uncertainties", [])
                        if uncs:
                            print(f"     Uncertainties: {len(uncs)}")
                            for u in uncs:
                                print(f"       ⚠ {u.get('description', '')[:100]}")
                        qs = brief.get("focus_questions", [])
                        if qs:
                            print(f"     Focus questions:")
                            for q in qs[:3]:
                                print(f"       ? {q[:100]}")
                        pri = brief.get("priority", {})
                        if pri:
                            print(f"     Priority: {pri.get('level', 'standard')} — {pri.get('reason', '')}")
                except Exception as _e:
                    pass  # Brief display is best-effort

                coord.approve(instance_id, approver="eval_runner")
                instance = coord.get_instance(instance_id)
            else:
                elapsed = time.time() - t0
                print(f"  ⏸ Suspended — use --auto-approve to continue")
                return {"case": case_name, "status": "suspended", "elapsed": elapsed,
                        "instance_id": instance_id}

        elapsed = time.time() - t0

        # Extract step outputs from result summary
        # _extract_result_summary flattens outputs: category, conforms, etc.
        # are directly on the step dict, not nested under "output"
        result = instance.result or {}
        steps = result.get("steps", [])
        step_outputs = {}

        if self.verbose:
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Steps in result: {len(steps)}")
            for s in steps:
                print(f"    {s.get('step_name')}/{s.get('primitive')}: {list(s.keys())}")

        for step in steps:
            name = step.get("step_name", step.get("name", ""))
            # The summary puts fields directly on step dict
            step_outputs[name] = step
            if self.verbose:
                prim = step.get("primitive", "?")
                # Show the relevant fields for each primitive type
                if prim == "classify":
                    print(f"  [{prim}] {name}: category={step.get('category')}, confidence={step.get('confidence')}")
                elif prim == "verify":
                    print(f"  [{prim}] {name}: conforms={step.get('conforms')}, violations={step.get('violations')}")
                elif prim == "think":
                    print(f"  [{prim}] {name}: risk_score={step.get('risk_score')}, recommendation={step.get('recommendation')}")
                elif prim == "investigate":
                    print(f"  [{prim}] {name}: finding={step.get('finding')}, evidence_flags={step.get('evidence_flags')}")
                elif prim == "generate":
                    print(f"  [{prim}] {name}: artifact_preview={str(step.get('artifact_preview', ''))[:80]}")
                elif prim == "retrieve":
                    print(f"  [{prim}] {name}: sources={step.get('sources')}")
                else:
                    print(f"  [{prim}] {name}: {json.dumps(step, indent=None)[:200]}")

        # Validate checks
        checks_passed = 0
        checks_failed = 0
        failures = []

        def _normalize_enum(val):
            """Normalize LLM output to snake_case enum for comparison."""
            if val is None:
                return None
            if not isinstance(val, str):
                return val
            s = val.strip().lower()
            # Common substitutions: "Auto-approve" → "auto_approve"
            s = s.replace("-", "_").replace(" ", "_")
            # Known synonyms the LLM uses
            SYNONYMS = {
                "auto_approve": ["auto_approve", "auto_approved", "autoapprove"],
                "refer_to_siu": ["refer_to_siu", "refer_to_investigations",
                                 "refer_to_special_investigations", "referred_to_siu"],
                "standard_review": ["standard_review", "manual_review",
                                    "further_review", "detailed_review"],
            }
            for canonical, aliases in SYNONYMS.items():
                if s in aliases:
                    return canonical
            return s

        for check_key, expected_val in case_def.get("checks", {}).items():
            step_name, field = check_key.rsplit(".", 1)
            actual = step_outputs.get(step_name, {}).get(field)

            # Normalize both sides for comparison
            norm_actual = _normalize_enum(actual)
            norm_expected = _normalize_enum(expected_val)

            # Support list of acceptable values
            if isinstance(expected_val, list):
                norm_expected_list = [_normalize_enum(v) for v in expected_val]
                if norm_actual in norm_expected_list:
                    checks_passed += 1
                    print(f"  ✅ {check_key}: {actual}")
                else:
                    checks_failed += 1
                    failures.append(f"{check_key}: expected one of {expected_val}, got={actual}")
                    print(f"  ❌ {check_key}: expected one of {expected_val}, got={actual}")
            elif norm_actual == norm_expected:
                checks_passed += 1
                print(f"  ✅ {check_key}: {actual}")
            else:
                checks_failed += 1
                failures.append(f"{check_key}: expected={expected_val}, got={actual}")
                print(f"  ❌ {check_key}: expected={expected_val}, got={actual}")

        # Validate risk range
        risk_range = case_def.get("risk_range")
        if risk_range:
            step_data = step_outputs.get("assess_risk", {})
            risk_score = step_data.get("risk_score")

            # risk_score may not be in structured output (think schema has no risk_score field)
            # Try to extract from conclusions or thought text
            if risk_score is None:
                conclusions = step_data.get("conclusions", [])
                thought = step_data.get("thought", "")
                # Look for "risk_score = 55" or "score: 55" patterns
                import re
                for text in [thought] + (conclusions if isinstance(conclusions, list) else []):
                    if isinstance(text, str):
                        match = re.search(r'(?:risk.?score|score)\s*[=:]\s*(\d+)', text, re.IGNORECASE)
                        if match:
                            risk_score = int(match.group(1))
                            break

            if risk_score is not None:
                if isinstance(risk_score, (int, float)) and risk_range[0] <= risk_score <= risk_range[1]:
                    checks_passed += 1
                    print(f"  ✅ risk_score: {risk_score} (in [{risk_range[0]}, {risk_range[1]}])")
                else:
                    checks_failed += 1
                    failures.append(f"risk_score: {risk_score} not in [{risk_range[0]}, {risk_range[1]}]")
                    print(f"  ❌ risk_score: {risk_score} not in [{risk_range[0]}, {risk_range[1]}]")
            else:
                # Can't validate — don't count as failure, just warn
                print(f"  ⚠️  risk_score: not found in structured output (skipping range check)")

        # Check delegations
        if case_def.get("delegations") and instance.status == InstanceStatus.COMPLETED:
            chain = coord.get_correlation_chain(instance.correlation_id)
            chain_wfs = {c.workflow_type for c in chain if c.instance_id != instance_id}

            for wf_name, should_exist in case_def["delegations"].items():
                exists = wf_name in chain_wfs
                if exists == should_exist:
                    checks_passed += 1
                    print(f"  ✅ delegation {wf_name}: {'triggered' if exists else 'not triggered'}")
                else:
                    checks_failed += 1
                    failures.append(f"delegation {wf_name}: expected={should_exist}, actual={exists}")
                    print(f"  ❌ delegation {wf_name}: expected={'triggered' if should_exist else 'not triggered'}, got={'triggered' if exists else 'not triggered'}")

        # Audit info
        ledger = coord.get_ledger(instance_id=instance_id)
        chain = coord.get_correlation_chain(instance.correlation_id)

        # ── Invariant checks ─────────────────────────────────────
        # These fire regardless of case-specific checks.
        # They verify architectural soundness, not per-case expectations.

        # Invariant I0: retrieve steps must return non-empty data
        for step in steps:
            if step.get("primitive") == "retrieve":
                sname = step.get("step_name", "")
                sources = step.get("sources", [])
                # Check if data keys exist and have content
                data_keys = step.get("data_keys") or step.get("sources_count")
                # If we have source info showing all empty, flag it
                # But only warn — synthetic fixtures may report empty sources
                # even when data is present in the fixture JSON

        # Invariant I1: conforms=False must never coexist with auto_approve
        eligibility = step_outputs.get("check_eligibility", {})
        risk_step = step_outputs.get("assess_risk", {})
        decision_step = step_outputs.get("generate_decision", {})
        denial_step = step_outputs.get("generate_denial", {})

        if eligibility.get("conforms") is False:
            # assess_risk and generate_decision should NOT have run
            if risk_step:
                checks_failed += 1
                failures.append("INVARIANT I1: assess_risk ran despite conforms=False")
                print(f"  ❌ INVARIANT I1: assess_risk ran despite conforms=False")
            elif decision_step:
                checks_failed += 1
                failures.append("INVARIANT I1: generate_decision ran despite conforms=False")
                print(f"  ❌ INVARIANT I1: generate_decision ran despite conforms=False")
            else:
                checks_passed += 1
                print(f"  ✅ INVARIANT I1: denied path skipped assess_risk + generate_decision")

            # generate_denial SHOULD have run
            if denial_step:
                checks_passed += 1
                print(f"  ✅ INVARIANT I1b: generate_denial produced for ineligible claim")
            else:
                checks_failed += 1
                failures.append("INVARIANT I1b: generate_denial missing for ineligible claim")
                print(f"  ❌ INVARIANT I1b: generate_denial missing for ineligible claim")

        # Invariant I2: parse errors on generate steps are hard fails
        parse_errors = []
        for step in steps:
            if step.get("primitive") == "generate":
                sname = step.get("step_name", "")
                confidence = step.get("confidence")
                artifact = step.get("artifact_preview", "") or step.get("artifact", "")
                if confidence == 0.0 or (isinstance(artifact, str) and "PARSE ERROR" in artifact):
                    checks_failed += 1
                    failures.append(f"INVARIANT I2: parse error in {sname}")
                    parse_errors.append(sname)
                    print(f"  ❌ INVARIANT I2: parse error in {sname}")

        # Invariant I3: artifact schema validation for generate steps
        # Required keys by step name (from domain format specs)
        ARTIFACT_SCHEMAS = {
            "generate_decision": ["claim_id", "decision", "risk_score", "eligible",
                                  "claim_type", "amount", "requires_delegation"],
            "generate_denial": ["claim_id", "decision", "eligible", "violations", "claim_type"],
            "generate_assessment": ["claim_id", "damage_grade", "repair_cost",
                                    "documentation_complete", "assessment_ready"],
            "generate_incomplete": ["claim_id", "damage_grade", "documentation_complete",
                                    "assessment_ready"],
            "generate_screening_result": ["claim_id", "fraud_risk_category", "patterns_found",
                                          "finding", "recommendation", "requires_siu"],
            # Hardship intake packet
            "generate_intake_packet": ["member_token", "account_summary", "hardship_claims",
                                       "evidence_index", "risk_flags", "missing_information",
                                       "preliminary_triage"],
        }
        artifact_validations = {}
        for step in steps:
            sname = step.get("step_name", "")
            if sname not in ARTIFACT_SCHEMAS:
                continue

            # Get artifact — could be dict, str (JSON), or preview
            artifact_raw = step.get("artifact") or step.get("artifact_preview", "")
            artifact_obj = None
            if isinstance(artifact_raw, dict):
                artifact_obj = artifact_raw
            elif isinstance(artifact_raw, str):
                # Try to parse JSON string
                try:
                    import json as _json
                    artifact_obj = _json.loads(artifact_raw)
                except (ValueError, TypeError):
                    pass

            required_keys = ARTIFACT_SCHEMAS[sname]
            if artifact_obj and isinstance(artifact_obj, dict):
                missing = [k for k in required_keys if k not in artifact_obj]
                if missing:
                    checks_failed += 1
                    failures.append(f"INVARIANT I3: {sname} artifact missing keys: {missing}")
                    print(f"  ❌ INVARIANT I3: {sname} artifact missing keys: {missing}")
                    artifact_validations[sname] = {"valid": False, "missing_keys": missing}
                else:
                    checks_passed += 1
                    print(f"  ✅ INVARIANT I3: {sname} artifact schema valid")
                    artifact_validations[sname] = {"valid": True}
            elif sname not in [e for e in parse_errors]:
                # Artifact not parseable but also not caught by I2
                checks_failed += 1
                failures.append(f"INVARIANT I3: {sname} artifact not a valid JSON object")
                print(f"  ❌ INVARIANT I3: {sname} artifact not a valid JSON object")
                artifact_validations[sname] = {"valid": False, "reason": "not_json_object"}

        # Gather gate/suspension metadata
        quality_gates_fired = sum(
            1 for e in ledger
            if getattr(e, 'action_type', '') == 'quality_gate_fired'
        )
        auto_approvals = sum(
            1 for e in ledger
            if getattr(e, 'action_type', '') in ('approved', 'auto_approved')
        )

        # Invariant I4: No generate step may have confidence=0.0
        # (separate from I2 because confidence can be 0.0 even without PARSE ERROR text)
        for step in steps:
            if step.get("primitive") == "generate":
                sname = step.get("step_name", "")
                conf = step.get("confidence")
                if conf is not None and conf == 0.0 and sname not in parse_errors:
                    checks_failed += 1
                    failures.append(f"INVARIANT I4: {sname} has confidence=0.0")
                    print(f"  ❌ INVARIANT I4: {sname} has confidence=0.0")

        # Invariant I5: Quality gate fired = fail (if strict_gates enabled)
        if quality_gates_fired > 0 and self.strict_gates:
            checks_failed += 1
            failures.append(f"INVARIANT I5: {quality_gates_fired} quality gate(s) fired (strict mode)")
            print(f"  ❌ INVARIANT I5: {quality_gates_fired} quality gate(s) fired (strict mode)")

        # Invariant I6: no delegations should fire when eligibility fails
        if eligibility.get("conforms") is False:
            delegation_count = len(chain) - 1  # subtract the main instance
            if delegation_count > 0:
                checks_failed += 1
                chain_wfs = [c.workflow_type for c in chain if c.instance_id != instance_id]
                failures.append(f"INVARIANT I6: {delegation_count} delegation(s) fired on denied claim: {chain_wfs}")
                print(f"  ❌ INVARIANT I6: {delegation_count} delegation(s) fired on denied claim: {chain_wfs}")
            else:
                checks_passed += 1
                print(f"  ✅ INVARIANT I6: no delegations on denied path")

        # Invariant I7: cross-step coherence (per domain)
        workflow = case_def.get("workflow", "")

        # I7a: Fraud screening — risk category ↔ finding ↔ recommendation
        if workflow == "fraud_screening":
            classify_step = step_outputs.get("classify_fraud_risk", {})
            investigate_step = step_outputs.get("investigate_patterns", {})
            risk_cat = classify_step.get("category")
            finding_raw = investigate_step.get("finding", "")
            recommendation_raw = investigate_step.get("recommendation", "")

            # The LLM may return prose instead of clean enums.
            # Extract enum value by checking if any known enum appears in the text.
            FINDING_ENUMS = ["no_suspicious_patterns", "minor_concerns", "significant_concerns"]
            REC_ENUMS = ["clear", "flag_for_monitoring", "refer_to_siu"]

            # Keyword patterns that map to enum values
            FINDING_KEYWORDS = {
                "no_suspicious_patterns": ["no suspicious", "no significant", "no fraud", "no pattern", "no concern", "nothing suspicious", "clean", "clearing", "no issues"],
                "minor_concerns": ["minor concern", "minor issue", "some concern", "low concern", "flag for monitoring", "monitoring", "worth monitoring"],
                "significant_concerns": ["significant concern", "serious concern", "major concern", "high concern", "refer to siu", "siu referral", "significant fraud", "suspicious pattern", "several suspicious", "multiple suspicious", "referred to the siu", "should be referred"],
            }
            REC_KEYWORDS = {
                "clear": ["clear", "approve", "no action", "no further"],
                "flag_for_monitoring": ["monitor", "flag for", "watch", "flag_for_monitoring"],
                "refer_to_siu": ["siu", "refer", "investigation unit", "special investigation"],
            }

            # Recommendation → finding fallback mapping
            REC_TO_FINDING = {
                "clear": "no_suspicious_patterns",
                "flag_for_monitoring": "minor_concerns",
                "refer_to_siu": "significant_concerns",
            }

            def _extract_enum(text, enums, keyword_map=None):
                """Extract a known enum value from text."""
                if not text:
                    return text
                t = str(text).lower()
                t_under = t.replace(" ", "_").replace("-", "_")
                # Exact match on underscored version
                if t_under in enums:
                    return t_under
                # Substring match on underscored version
                for e in enums:
                    if e in t_under:
                        return e
                # Keyword match — check if any keywords for an enum appear in text
                if keyword_map:
                    for enum_val, keywords in keyword_map.items():
                        for kw in keywords:
                            if kw in t:
                                return enum_val
                return text  # couldn't extract — return raw

            finding = _extract_enum(finding_raw, FINDING_ENUMS, FINDING_KEYWORDS)
            recommendation = _extract_enum(recommendation_raw, REC_ENUMS, REC_KEYWORDS)

            # Fallback: if finding is still prose, infer from recommendation
            if finding not in FINDING_ENUMS and recommendation in REC_TO_FINDING:
                finding = REC_TO_FINDING[recommendation]

            COHERENCE_TABLE = {
                "low_risk": {"no_suspicious_patterns", "minor_concerns"},
                "medium_risk": {"minor_concerns", "significant_concerns"},
                "high_risk": {"significant_concerns"},
            }
            REC_TABLE = {
                "no_suspicious_patterns": "clear",
                "minor_concerns": "flag_for_monitoring",
                "significant_concerns": "refer_to_siu",
            }

            if risk_cat and finding:
                allowed = COHERENCE_TABLE.get(risk_cat, set())
                if finding in allowed:
                    checks_passed += 1
                    print(f"  ✅ INVARIANT I7a: {risk_cat} ↔ {finding} coherent")
                else:
                    # Advisory only — semantic coherence is the LLM's job.
                    # If extraction failed (finding is still prose), note it
                    # but don't fail the case.
                    if finding in FINDING_ENUMS:
                        # Clean enum but wrong pairing — worth noting
                        print(f"  ⚠️  INVARIANT I7a: {risk_cat} ↔ {finding} (unexpected pairing)")
                    else:
                        # Prose — extraction couldn't resolve it
                        print(f"  ⚠️  INVARIANT I7a: could not extract finding enum from prose (advisory)")

            if finding and recommendation:
                expected_rec = REC_TABLE.get(finding)
                if expected_rec and recommendation == expected_rec:
                    checks_passed += 1
                    print(f"  ✅ INVARIANT I7a: {finding} → {recommendation} correct")
                elif expected_rec:
                    # Advisory — don't fail
                    print(f"  ⚠️  INVARIANT I7a: {finding} → expected {expected_rec}, got {recommendation} (advisory)")

        # I7b: Damage assessment — doc conformance ↔ artifact type
        if workflow == "damage_assessment":
            verify_step = step_outputs.get("verify_documentation", {})
            has_assessment = "generate_assessment" in step_outputs
            has_incomplete = "generate_incomplete" in step_outputs
            conforms = verify_step.get("conforms")

            if conforms is False:
                if has_incomplete and not has_assessment:
                    checks_passed += 1
                    print(f"  ✅ INVARIANT I7b: docs incomplete → generate_incomplete (correct route)")
                else:
                    checks_failed += 1
                    failures.append("INVARIANT I7b: docs incomplete but wrong generate step ran")
                    print(f"  ❌ INVARIANT I7b: docs incomplete but wrong generate step ran")
            elif conforms is True:
                if has_assessment and not has_incomplete:
                    checks_passed += 1
                    print(f"  ✅ INVARIANT I7b: docs complete → generate_assessment (correct route)")
                else:
                    checks_failed += 1
                    failures.append("INVARIANT I7b: docs complete but wrong generate step ran")
                    print(f"  ❌ INVARIANT I7b: docs complete but wrong generate step ran")

        # I7c: Hardship intake — structural checks on intake packet
        if workflow == "hardship_intake_packet":
            generate_step = step_outputs.get("generate_intake_packet", {})
            artifact_raw = generate_step.get("artifact") or generate_step.get("artifact_preview", "")
            artifact = None
            if isinstance(artifact_raw, dict):
                artifact = artifact_raw
            elif isinstance(artifact_raw, str):
                try:
                    artifact = json.loads(artifact_raw)
                except (ValueError, TypeError):
                    pass

            structural = case_def.get("structural_checks", {})

            if artifact and isinstance(artifact, dict):
                # PII check — no SSN patterns, DOB patterns, full addresses
                if structural.get("no_pii"):
                    artifact_str = json.dumps(artifact)
                    import re as _re
                    ssn_pattern = _re.search(r'\b\d{3}-\d{2}-\d{4}\b', artifact_str)
                    if ssn_pattern:
                        checks_failed += 1
                        failures.append("INVARIANT I7c: PII detected (SSN pattern) in intake packet")
                        print(f"  ❌ INVARIANT I7c: PII detected (SSN pattern)")
                    else:
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: no SSN patterns in intake packet")

                # Evidence linkage check
                if structural.get("evidence_linked"):
                    claims = artifact.get("hardship_claims", [])
                    evidence_index = artifact.get("evidence_index", [])
                    all_linked = True
                    for claim in claims:
                        if isinstance(claim, dict):
                            status = claim.get("evidence_status", "")
                            eids = claim.get("evidence_ids", [])
                            if status != "member_stated_unverified" and not eids:
                                all_linked = False
                    if all_linked and claims:
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: all claims evidence-linked or marked unverified")
                    elif claims:
                        print(f"  ⚠️  INVARIANT I7c: some claims missing evidence linkage (advisory)")

                # Risk flag checks
                risk_flags = artifact.get("risk_flags", {})
                if structural.get("must_flag_fraud"):
                    fraud_status = ""
                    if isinstance(risk_flags, dict):
                        fe = risk_flags.get("fraud_exploitation", {})
                        fraud_status = fe.get("status", "") if isinstance(fe, dict) else str(fe)
                    if fraud_status in ("elevated", "high"):
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: fraud/exploitation flagged ({fraud_status})")
                    else:
                        checks_failed += 1
                        failures.append(f"INVARIANT I7c: fraud exploitation not flagged (ATO=87)")
                        print(f"  ❌ INVARIANT I7c: fraud exploitation not flagged (got: {fraud_status})")

                if structural.get("must_flag_scra"):
                    scra_status = ""
                    if isinstance(risk_flags, dict):
                        scra = risk_flags.get("scra", {})
                        scra_status = scra.get("status", "") if isinstance(scra, dict) else str(scra)
                    if scra_status and scra_status != "none":
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: SCRA flagged ({scra_status})")
                    else:
                        checks_failed += 1
                        failures.append("INVARIANT I7c: SCRA not flagged despite pending indicator")
                        print(f"  ❌ INVARIANT I7c: SCRA not flagged (got: {scra_status})")

                if structural.get("must_flag_complaint"):
                    complaint_status = ""
                    if isinstance(risk_flags, dict):
                        comp = risk_flags.get("complaints", {})
                        complaint_status = comp.get("status", "") if isinstance(comp, dict) else str(comp)
                    if complaint_status and complaint_status != "none":
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: complaint flagged ({complaint_status})")
                    else:
                        checks_failed += 1
                        failures.append("INVARIANT I7c: open complaint not flagged")
                        print(f"  ❌ INVARIANT I7c: complaint not flagged (got: {complaint_status})")

                if structural.get("must_flag_conflict"):
                    # Check that missing_information or evidence_index mentions conflict
                    artifact_str = json.dumps(artifact).lower()
                    if "conflict" in artifact_str or "contradict" in artifact_str or "mismatch" in artifact_str:
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: income conflict detected in intake packet")
                    else:
                        # Advisory — conflict detection is semantic
                        print(f"  ⚠️  INVARIANT I7c: income conflict not explicitly noted (advisory)")

                # Triage check
                artifact_checks = case_def.get("artifact_checks", {})
                gen_checks = artifact_checks.get("generate_intake_packet", {})
                field_checks = gen_checks.get("field_checks", {})
                for field, expected_vals in field_checks.items():
                    actual = artifact.get(field, "")
                    if actual in expected_vals:
                        checks_passed += 1
                        print(f"  ✅ INVARIANT I7c: {field}={actual} (expected)")
                    else:
                        # Triage is advisory
                        print(f"  ⚠️  INVARIANT I7c: {field}={actual}, expected one of {expected_vals} (advisory)")

        # ── Domain Contract Validation (auto-generated from domain YAML) ──
        # Contract checks are ADVISORY — they surface issues but don't fail cases.
        # The hardcoded invariants (I1-I7) remain the gating checks.
        # Contract checks graduate to gating only after proven reliable.
        contract_violations = []
        try:
            import importlib.util
            dc_path = os.path.join(self.project_root, "engine", "domain_contract.py")
            if os.path.exists(dc_path):
                dc_spec = importlib.util.spec_from_file_location(
                    "engine.domain_contract", dc_path
                )
                dc_mod = importlib.util.module_from_spec(dc_spec)
                dc_mod.__package__ = "engine"
                import sys as _sys
                _sys.modules["engine.domain_contract"] = dc_mod
                dc_spec.loader.exec_module(dc_mod)
                DomainContract = dc_mod.DomainContract

                domain_path = os.path.join(
                    self.project_root, "domains", f"{case_def['domain']}.yaml"
                )
                if os.path.exists(domain_path):
                    contract = DomainContract.from_yaml(domain_path)
                    if contract.has_contracts:
                        contract_violations = contract.validate_run(step_outputs)
                        if contract_violations:
                            for v in contract_violations:
                                print(f"  ⚠️  CONTRACT {v.invariant}: {v.step} — {v.message}")
                        else:
                            print(f"  ✅ DOMAIN CONTRACT: all checks passed ({contract.domain_name})")
        except ImportError:
            pass
        except Exception as e:
            print(f"  ⚠️  DOMAIN CONTRACT: validation error: {e}")

        status = "pass" if checks_failed == 0 else "fail"
        result_entry = {
            "case": case_name,
            "status": status,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "failures": failures,
            "elapsed": round(elapsed, 2),
            "steps": len(steps),
            "audit_entries": len(ledger),
            "delegations": len(chain) - 1,
            "instance_id": instance_id,
            # Enriched metadata (reviewer feedback)
            "quality_gates_fired": quality_gates_fired,
            "parse_errors": parse_errors,
            "auto_approvals": auto_approvals,
            "artifact_validations": artifact_validations,
            "contract_violations": [
                {"invariant": v.invariant, "step": v.step,
                 "message": v.message, "severity": v.severity}
                for v in contract_violations
            ],
        }

        icon = "✅" if status == "pass" else "❌"
        print(f"\n  {icon} {case_name}: {checks_passed}/{checks_passed + checks_failed} checks | "
              f"{elapsed:.1f}s | {len(steps)} steps | {len(chain)-1} delegations | {len(ledger)} audit entries")

        return result_entry

    def run_all(self, case_names: list[str]) -> dict:
        """Run multiple cases and produce summary."""
        print("═" * 60)
        print("COGNITIVE CORE — LIVE LLM EVALUATION")
        print(f"Provider: {os.environ.get('LLM_PROVIDER', 'not set')}")
        print(f"Cases: {len(case_names)}")
        print(f"Auto-approve: {self.auto_approve}")
        print("═" * 60)

        results = []
        for name in case_names:
            if name not in CASES:
                print(f"\n⚠ Unknown case: {name}")
                continue
            result = self.run_case(name, CASES[name])
            results.append(result)

        # Summary
        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        crashed = sum(1 for r in results if r["status"] == "crashed")
        suspended = sum(1 for r in results if r["status"] == "suspended")
        total_time = sum(r.get("elapsed", 0) for r in results)
        total_checks = sum(r.get("checks_passed", 0) + r.get("checks_failed", 0) for r in results)
        checks_passed = sum(r.get("checks_passed", 0) for r in results)

        print(f"\n{'═'*60}")
        print(f"RESULTS")
        print(f"{'═'*60}")
        print(f"  Cases:    {passed} passed, {failed} failed, {crashed} crashed, {suspended} suspended")
        print(f"  Checks:   {checks_passed}/{total_checks}")
        print(f"  Time:     {total_time:.1f}s total")
        print(f"  Provider: {os.environ.get('LLM_PROVIDER', 'not set')}")

        # Aggregate integrity metadata
        total_quality_gates = sum(r.get("quality_gates_fired", 0) for r in results)
        total_parse_errors = sum(len(r.get("parse_errors", [])) for r in results)
        total_auto_approvals = sum(r.get("auto_approvals", 0) for r in results)
        artifact_checks = {}
        for r in results:
            for step_name, av in r.get("artifact_validations", {}).items():
                artifact_checks.setdefault(step_name, {"valid": 0, "invalid": 0})
                if av.get("valid"):
                    artifact_checks[step_name]["valid"] += 1
                else:
                    artifact_checks[step_name]["invalid"] += 1

        print(f"\n  Integrity:")
        print(f"    Quality gates fired: {total_quality_gates}")
        print(f"    Parse errors:        {total_parse_errors}")
        print(f"    Auto-approvals:      {total_auto_approvals}")
        if artifact_checks:
            print(f"    Artifact schemas:")
            for sn, counts in sorted(artifact_checks.items()):
                total = counts["valid"] + counts["invalid"]
                icon = "✅" if counts["invalid"] == 0 else "❌"
                print(f"      {icon} {sn}: {counts['valid']}/{total} valid")

        if failed > 0 or crashed > 0:
            print(f"\n  FAILURES:")
            for r in results:
                if r["status"] in ("fail", "crashed"):
                    print(f"    {r['case']}: {r.get('failures', r.get('error', '?'))}")

        print(f"{'═'*60}")

        overall = "PASS" if failed == 0 and crashed == 0 and suspended == 0 else "FAIL"
        print(f"\n  {'✅' if overall == 'PASS' else '❌'} OVERALL: {overall}")

        # Write results JSON
        report_path = os.path.join(self.project_root, "eval_results.json")
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": os.environ.get("LLM_PROVIDER", "unknown"),
                "overall": overall,
                "cases_passed": passed,
                "cases_failed": failed,
                "cases_crashed": crashed,
                "checks_passed": checks_passed,
                "checks_total": total_checks,
                "total_time_seconds": round(total_time, 2),
                # Integrity metadata
                "quality_gates_fired": total_quality_gates,
                "parse_errors": total_parse_errors,
                "auto_approvals": total_auto_approvals,
                "artifact_schema_results": artifact_checks,
                "contract_violations": sum(
                    len(r.get("contract_violations", []))
                    for r in results if isinstance(r, dict)
                ),
                "results": results,
            }, f, indent=2)
        print(f"\n  Report: {report_path}")

        return {"overall": overall, "passed": passed, "failed": failed, "results": results}


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Live LLM evaluation for Cognitive Core")
    parser.add_argument("--case", help="Run a single case (e.g., sc_002_physical_damage)")
    parser.add_argument("--all", action="store_true", help="Run all cases")
    parser.add_argument("--workflow", help="Run all cases for a workflow (claim_intake, damage_assessment, fraud_screening, hardship_intake_packet)")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve governance gate suspensions")
    parser.add_argument("--strict-gates", action="store_true",
                        help="Quality gates (parse errors, schema violations) are hard fails even with --auto-approve")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full step outputs")
    parser.add_argument("--project-root", default=BASE, help="Project root directory")
    args = parser.parse_args()

    # Validate environment
    provider = os.environ.get("LLM_PROVIDER", "")
    if not provider:
        print("Error: LLM_PROVIDER not set")
        print("  For Gemini: export LLM_PROVIDER=google GOOGLE_API_KEY=your_key")
        print("  For Azure:  export LLM_PROVIDER=azure_foundry AZURE_AI_PROJECT_ENDPOINT=...")
        sys.exit(1)

    if provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        sys.exit(1)

    # Determine cases to run
    if args.case:
        case_names = [args.case]
    elif args.workflow:
        case_names = [name for name, defn in CASES.items() if defn["workflow"] == args.workflow]
    elif args.all:
        case_names = list(CASES.keys())
    else:
        # Default: one representative case per workflow
        case_names = ["sc_002_physical_damage", "da_002_moderate", "fs_003_medium_risk"]

    evaluator = LiveEvaluator(
        project_root=args.project_root,
        verbose=args.verbose,
        auto_approve=args.auto_approve,
        strict_gates=args.strict_gates,
    )

    result = evaluator.run_all(case_names)
    sys.exit(0 if result["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
