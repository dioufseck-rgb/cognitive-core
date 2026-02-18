"""
Cognitive Core — Cost Tracking per Workflow

Tracks tokens and costs per LLM call, aggregates per step, per workflow,
per domain.

Pricing table lives in llm_config.yaml (Design: Option A — co-located
with provider config).

Usage:
    tracker = CostTracker()
    tracker.record_call(
        model="gemini-2.0-flash",
        input_tokens=1500,
        output_tokens=500,
        step_name="classify_return_type",
    )
    summary = tracker.summary()
    # {"total_cost_usd": 0.00035, "total_input_tokens": 1500, ...}
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("cognitive_core.cost")


# ═══════════════════════════════════════════════════════════════════
# Pricing
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelPricing:
    """Pricing per million tokens for a model."""
    model: str
    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        return input_cost + output_cost


def load_pricing(config_path: str | None = None) -> dict[str, ModelPricing]:
    """Load pricing from llm_config.yaml."""
    path = config_path or _find_config_path()
    if not path:
        return {}

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    pricing_cfg = cfg.get("pricing", {})
    result = {}
    for model, prices in pricing_cfg.items():
        result[model] = ModelPricing(
            model=model,
            input_per_million=prices.get("input_per_million", 0.0),
            output_per_million=prices.get("output_per_million", 0.0),
        )
    return result


# ═══════════════════════════════════════════════════════════════════
# Call Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CallRecord:
    """Single LLM call cost record."""
    model: str
    step_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float


# ═══════════════════════════════════════════════════════════════════
# Cost Tracker
# ═══════════════════════════════════════════════════════════════════

class BudgetExceededError(Exception):
    """Raised when a workflow exceeds its cost budget."""
    pass


# Default cost estimate for unknown models (per million tokens)
UNKNOWN_MODEL_INPUT_PER_MILLION = 1.00   # Conservative: ~$1/M input
UNKNOWN_MODEL_OUTPUT_PER_MILLION = 3.00  # Conservative: ~$3/M output


class CostTracker:
    """
    Tracks LLM call costs for a workflow execution.

    Thread-safe. One tracker per workflow instance.
    Budget cap: if set, raises BudgetExceededError when total_cost exceeds budget.
    Unknown models: logs WARNING and uses conservative estimate (not silent $0).
    """

    def __init__(
        self,
        config_path: str | None = None,
        budget_usd: float | None = None,
        unknown_model_action: str = "warn",  # "warn" or "fail"
    ):
        self._pricing = load_pricing(config_path)
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()
        self.budget_usd = budget_usd
        self.unknown_model_action = unknown_model_action
        self._unknown_models_seen: set[str] = set()

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        step_name: str = "",
    ) -> CallRecord:
        """
        Record a single LLM call.

        Raises BudgetExceededError if budget is set and would be exceeded.
        """
        pricing = self._pricing.get(model)
        if pricing:
            cost = pricing.cost(input_tokens, output_tokens)
        else:
            # Unknown model — estimate conservatively, don't silently use $0
            estimated_cost = (
                (input_tokens / 1_000_000) * UNKNOWN_MODEL_INPUT_PER_MILLION
                + (output_tokens / 1_000_000) * UNKNOWN_MODEL_OUTPUT_PER_MILLION
            )

            if self.unknown_model_action == "fail":
                raise ValueError(
                    f"No pricing configured for model '{model}'. "
                    f"Add it to llm_config.yaml → pricing section."
                )

            # warn (default)
            if model not in self._unknown_models_seen:
                logger.warning(
                    "No pricing for model '%s' — using conservative estimate "
                    "($%.2f/M input, $%.2f/M output). Add to llm_config.yaml → pricing.",
                    model, UNKNOWN_MODEL_INPUT_PER_MILLION, UNKNOWN_MODEL_OUTPUT_PER_MILLION,
                )
                self._unknown_models_seen.add(model)
            cost = estimated_cost

        record = CallRecord(
            model=model,
            step_name=step_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=time.time(),
        )

        with self._lock:
            self._records.append(record)

            # Budget enforcement
            if self.budget_usd is not None:
                total = sum(r.cost_usd for r in self._records)
                if total > self.budget_usd:
                    raise BudgetExceededError(
                        f"Workflow cost ${total:.4f} exceeds budget "
                        f"${self.budget_usd:.4f} at step '{step_name}'"
                    )

        return record

    def summary(self) -> dict[str, Any]:
        """Aggregate cost summary for the workflow."""
        with self._lock:
            records = list(self._records)

        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)

        by_step: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.step_name not in by_step:
                by_step[r.step_name] = {
                    "calls": 0, "input_tokens": 0,
                    "output_tokens": 0, "cost_usd": 0.0,
                }
            s = by_step[r.step_name]
            s["calls"] += 1
            s["input_tokens"] += r.input_tokens
            s["output_tokens"] += r.output_tokens
            s["cost_usd"] += r.cost_usd

        by_model: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "calls": 0, "input_tokens": 0,
                    "output_tokens": 0, "cost_usd": 0.0,
                }
            m = by_model[r.model]
            m["calls"] += 1
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["cost_usd"] += r.cost_usd

        return {
            "total_calls": len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 6),
            "by_step": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_step.items()},
            "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_model.items()},
        }

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._records)


def _find_config_path() -> str | None:
    """Find llm_config.yaml."""
    candidates = [
        os.environ.get("LLM_CONFIG_PATH", ""),
        "llm_config.yaml",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_config.yaml"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None
