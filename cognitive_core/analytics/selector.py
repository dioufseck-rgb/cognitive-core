"""
Cognitive Core — Artifact Selector (TASK 4)

Selects the best-matching artifact for a primitive via LLM confidence scoring.
Uses the `fast` model alias — never `strong`.

Selection result types:
  selected            — an artifact was chosen with confidence ≥ threshold
  abstained           — eligible artifacts exist but confidence < threshold
  no_eligible_artifact — no artifact matches the eligibility predicates

Fallback behaviour on abstention (CC_ANALYTICS_FALLBACK):
  skip      → continue without the artifact
  escalate  → escalate governance tier (default)
  fail      → raise ArtifactSelectionFailedError

Special rule: at `hold` tier, abstention ALWAYS escalates to human regardless
of CC_ANALYTICS_FALLBACK.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.analytics.selector")

_DEFAULT_THRESHOLD = 0.75
_FALLBACK_DEFAULT = "escalate"


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class SelectionResult:
    outcome: str          # "selected" | "abstained" | "no_eligible_artifact"
    artifact: dict[str, Any] | None = None
    confidence: float = 0.0
    reason: str = ""
    fallback_action: str = ""   # populated when outcome == "abstained"


class ArtifactSelectionFailedError(RuntimeError):
    """Raised when CC_ANALYTICS_FALLBACK=fail and selection abstains."""


# ── Scorer ────────────────────────────────────────────────────────────────────

_SCORE_PROMPT = """\
You are an analytics artifact matcher. Given a list of candidate artifacts and
case context, score how well each artifact matches this case.

Case context:
{context}

Candidates:
{candidates}

For each candidate return a confidence score between 0.0 and 1.0 indicating
how appropriate this artifact is for this case. Respond with JSON only:
{{"scores": {{"<artifact_name>": <score>, ...}}}}
"""


def _score_candidates(
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    llm: Any,
) -> dict[str, float]:
    """
    Call the LLM (fast model) to score each candidate artifact.
    Returns artifact_name → confidence.
    Falls back to 1.0 for all candidates if LLM call fails.
    """
    if not candidates:
        return {}

    candidate_summary = json.dumps(
        [
            {
                "artifact_name": a["artifact_name"],
                "artifact_type": a["artifact_type"],
                "version": a["version"],
            }
            for a in candidates
        ],
        indent=2,
    )
    context_summary = json.dumps(
        {k: v for k, v in context.items() if not isinstance(v, dict)},
        indent=2,
    )
    prompt = _SCORE_PROMPT.format(
        context=context_summary,
        candidates=candidate_summary,
    )

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)

        # Extract JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            scores = parsed.get("scores", {})
            return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        logger.warning("Artifact confidence scoring LLM call failed: %s", e)

    # Fallback: assign 1.0 to all — eligibility already filtered them
    return {a["artifact_name"]: 1.0 for a in candidates}


# ── Selector ──────────────────────────────────────────────────────────────────

class ArtifactSelector:
    """
    Selects a single artifact for a primitive using LLM confidence scoring.
    """

    def __init__(
        self,
        threshold: float | None = None,
        fallback: str | None = None,
    ):
        self.threshold = threshold if threshold is not None else float(
            os.environ.get("CC_ANALYTICS_SELECTION_THRESHOLD", str(_DEFAULT_THRESHOLD))
        )
        self.fallback = fallback or os.environ.get("CC_ANALYTICS_FALLBACK", _FALLBACK_DEFAULT)

    def select(
        self,
        registry: Any,           # AnalyticsRegistry
        artifact_type: str,
        context: dict[str, Any],
        tier: str = "auto",
        llm: Any = None,
    ) -> SelectionResult:
        """
        Select the best artifact for a primitive.

        Args:
            registry: AnalyticsRegistry instance
            artifact_type: e.g. "causal_dag", "sequential_decision"
            context: case input + metadata (used for eligibility + scoring)
            tier: governance tier — affects abstention handling
            llm: LLM for confidence scoring (fast model); if None, skips scoring

        Returns:
            SelectionResult with outcome, artifact, and confidence.
        """
        candidates = registry.list_eligible(context, artifact_type=artifact_type)

        if not candidates:
            return SelectionResult(
                outcome="no_eligible_artifact",
                reason=f"No eligible {artifact_type} artifact for context",
            )

        # Score candidates
        if llm is not None:
            scores = _score_candidates(candidates, context, llm)
        else:
            # No LLM provided — treat all eligible artifacts as fully confident
            scores = {a["artifact_name"]: 1.0 for a in candidates}

        # Pick the highest-scoring candidate
        best = max(candidates, key=lambda a: scores.get(a["artifact_name"], 0.0))
        best_score = scores.get(best["artifact_name"], 0.0)

        if best_score >= self.threshold:
            return SelectionResult(
                outcome="selected",
                artifact=best,
                confidence=best_score,
            )

        # Abstention path
        return self._handle_abstention(best, best_score, tier)

    def _handle_abstention(
        self, best: dict, score: float, tier: str
    ) -> SelectionResult:
        """
        Apply fallback policy when confidence < threshold.
        At hold tier, always escalates regardless of CC_ANALYTICS_FALLBACK.
        """
        fallback = self.fallback
        reason = (
            f"Confidence {score:.2f} below threshold {self.threshold:.2f} "
            f"for artifact '{best['artifact_name']}'"
        )

        # hold tier invariant — never degrade silently
        if tier == "hold":
            logger.warning(
                "Analytics abstention at HOLD tier — forcing escalation. %s", reason
            )
            return SelectionResult(
                outcome="abstained",
                artifact=best,
                confidence=score,
                reason=reason,
                fallback_action="escalate",
            )

        if fallback == "fail":
            raise ArtifactSelectionFailedError(reason)

        if fallback == "skip":
            return SelectionResult(
                outcome="abstained",
                artifact=best,
                confidence=score,
                reason=reason,
                fallback_action="skip",
            )

        # default: escalate
        return SelectionResult(
            outcome="abstained",
            artifact=best,
            confidence=score,
            reason=reason,
            fallback_action="escalate",
        )
