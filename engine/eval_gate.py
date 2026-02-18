"""
Cognitive Core — Model Versioning & Eval-Gated Deployment

Gates model/provider changes behind eval pack results. Compares
against both absolute thresholds AND stored baselines.

Design: Option C — must pass absolute thresholds AND not regress >5%
from baseline.

Workflow:
  1. Run eval pack against candidate model
  2. Compare against absolute quality gates (existing acceptance_criteria)
  3. Compare against stored baseline (last passing run)
  4. If both pass → approve, save as new baseline
  5. If either fails → reject, exit 1

Baseline storage: JSON file per pack in evals/baselines/
  evals/baselines/product_return.json
  evals/baselines/card_dispute.json

Usage:
    from engine.eval_gate import EvalGate
    gate = EvalGate(baselines_dir="evals/baselines")
    result = gate.evaluate(eval_result, model_version="gemini-2.0-flash")
    # result.approved → True/False
    # result.exit_code → 0 (pass) or 1 (fail)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cognitive_core.eval_gate")


# ═══════════════════════════════════════════════════════════════════
# Baseline
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Baseline:
    """Stored baseline from last passing eval run."""
    pack_name: str
    model_version: str
    timestamp: float
    gate_scores: dict[str, float]  # gate_name → actual percentage
    total_cases: int
    passed_cases: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "gate_scores": self.gate_scores,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Baseline:
        return cls(
            pack_name=d["pack_name"],
            model_version=d["model_version"],
            timestamp=d["timestamp"],
            gate_scores=d["gate_scores"],
            total_cases=d["total_cases"],
            passed_cases=d["passed_cases"],
        )


# ═══════════════════════════════════════════════════════════════════
# Gate Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RegressionDetail:
    """Detail for a single gate's regression check."""
    gate: str
    baseline_score: float
    current_score: float
    delta: float
    max_regression: float
    regressed: bool


@dataclass
class GateVerdict:
    """Full eval gate verdict."""
    approved: bool
    model_version: str
    pack_name: str

    # Absolute threshold results
    gates_passed: bool
    gate_results: dict[str, dict[str, Any]]

    # Regression results
    baseline_exists: bool
    regression_passed: bool
    regressions: list[RegressionDetail] = field(default_factory=list)

    # Summary
    reasons: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 0 if self.approved else 1

    def summary(self) -> str:
        lines = []
        status = "APPROVED" if self.approved else "REJECTED"
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  EVAL GATE: {status}")
        lines.append(f"  Model: {self.model_version}  Pack: {self.pack_name}")
        lines.append(f"{'=' * 60}")

        lines.append(f"\n  Absolute Gates: {'PASS' if self.gates_passed else 'FAIL'}")
        for gate, result in self.gate_results.items():
            mark = "✓" if result["passed"] else "✗"
            lines.append(
                f"    {mark} {gate}: {result['actual']:.1f}% "
                f"(threshold: {result['threshold']:.1f}%)"
            )

        if self.baseline_exists:
            lines.append(f"\n  Regression Check: {'PASS' if self.regression_passed else 'FAIL'}")
            for r in self.regressions:
                if r.regressed:
                    lines.append(
                        f"    ✗ {r.gate}: {r.current_score:.1f}% → "
                        f"dropped {abs(r.delta):.1f}% from baseline {r.baseline_score:.1f}% "
                        f"(max allowed: {r.max_regression:.1f}%)"
                    )
        else:
            lines.append("\n  Regression Check: SKIPPED (no baseline)")

        if self.reasons:
            lines.append(f"\n  Reasons:")
            for reason in self.reasons:
                lines.append(f"    - {reason}")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Eval Gate
# ═══════════════════════════════════════════════════════════════════

class EvalGate:
    """
    Eval-gated deployment controller.

    Compares eval results against:
      1. Absolute quality gate thresholds (from acceptance_criteria)
      2. Stored baseline from last passing run (regression check)
    """

    def __init__(
        self,
        baselines_dir: str = "evals/baselines",
        max_regression_pct: float = 5.0,
    ):
        self.baselines_dir = Path(baselines_dir)
        self.max_regression_pct = max_regression_pct

    def _baseline_path(self, pack_name: str) -> Path:
        safe_name = pack_name.replace("/", "_").replace(" ", "_").lower()
        return self.baselines_dir / f"{safe_name}.json"

    def load_baseline(self, pack_name: str) -> Baseline | None:
        """Load stored baseline for a pack."""
        path = self._baseline_path(pack_name)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return Baseline.from_dict(json.load(f))
        except Exception as e:
            logger.warning("Failed to load baseline %s: %s", path, e)
            return None

    def save_baseline(self, baseline: Baseline) -> None:
        """Save baseline after a passing run."""
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_path(baseline.pack_name)
        with open(path, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2)
        logger.info("Baseline saved: %s", path)

    def evaluate(
        self,
        eval_result: Any,  # EvalResult from evals.runner
        model_version: str = "unknown",
    ) -> GateVerdict:
        """
        Evaluate an eval result against gates and baseline.

        Args:
            eval_result: EvalResult with .gate_results() and .all_gates_pass
            model_version: Model version being tested

        Returns:
            GateVerdict with approval status and details
        """
        pack_name = eval_result.pack_name
        gate_results = eval_result.gate_results()
        gates_passed = eval_result.all_gates_pass

        # Build current scores dict
        current_scores = {
            gate: result["actual"]
            for gate, result in gate_results.items()
        }

        # Load baseline
        baseline = self.load_baseline(pack_name)
        baseline_exists = baseline is not None

        # Regression check
        regressions = []
        regression_passed = True
        reasons = []

        if not gates_passed:
            failed_gates = [g for g, r in gate_results.items() if not r["passed"]]
            reasons.append(f"Absolute gates failed: {', '.join(failed_gates)}")

        if baseline_exists:
            for gate, current in current_scores.items():
                baseline_score = baseline.gate_scores.get(gate)
                if baseline_score is None:
                    continue  # New gate, no baseline

                delta = current - baseline_score
                regressed = delta < -self.max_regression_pct

                regressions.append(RegressionDetail(
                    gate=gate,
                    baseline_score=baseline_score,
                    current_score=current,
                    delta=round(delta, 1),
                    max_regression=self.max_regression_pct,
                    regressed=regressed,
                ))

                if regressed:
                    regression_passed = False
                    reasons.append(
                        f"{gate} regressed {abs(delta):.1f}% "
                        f"(baseline: {baseline_score:.1f}%, "
                        f"current: {current:.1f}%, "
                        f"max allowed: {self.max_regression_pct}%)"
                    )
        else:
            regression_passed = True  # No baseline → can't regress

        approved = gates_passed and regression_passed

        verdict = GateVerdict(
            approved=approved,
            model_version=model_version,
            pack_name=pack_name,
            gates_passed=gates_passed,
            gate_results=gate_results,
            baseline_exists=baseline_exists,
            regression_passed=regression_passed,
            regressions=regressions,
            reasons=reasons,
        )

        # Save new baseline if approved
        if approved:
            new_baseline = Baseline(
                pack_name=pack_name,
                model_version=model_version,
                timestamp=time.time(),
                gate_scores=current_scores,
                total_cases=eval_result.total,
                passed_cases=eval_result.passed,
            )
            self.save_baseline(new_baseline)
            logger.info("Eval gate APPROVED: %s (model: %s)", pack_name, model_version)
        else:
            logger.warning(
                "Eval gate REJECTED: %s (model: %s) — %s",
                pack_name, model_version, "; ".join(reasons),
            )

        return verdict
