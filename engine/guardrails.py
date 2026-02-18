"""
Cognitive Core — Input Guardrails (S-009)

Prompt injection defense for case inputs. Hybrid strategy:
  1. Deterministic patterns (regex + keyword): catches known injection techniques
  2. LLM classifier (optional): evaluates ambiguous inputs via guard model

Detection categories:
  - instruction_override: "ignore previous", "disregard above", "new instructions"
  - role_play: "you are now", "pretend you are", "act as"
  - delimiter_attack: "---", triple backticks, XML tags, system tokens
  - encoding_evasion: base64 instructions, Unicode homoglyphs
  - output_manipulation: "respond only with", "format your response as"

Risk levels:
  - CLEAR: no patterns detected, proceed normally
  - AMBIGUOUS: low-confidence patterns, may escalate to LLM classifier
  - HIGH: definite injection patterns, auto-escalate to 'hold' tier

Usage:
    from engine.guardrails import InputGuardrail, scan_case_input

    guard = InputGuardrail()
    result = guard.scan(case_input)
    if result.risk == "HIGH":
        # Auto-escalate or reject
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.guardrails")


@dataclass
class GuardrailResult:
    """Result of an input guardrail scan."""
    risk: str                              # CLEAR, AMBIGUOUS, HIGH
    score: float                           # 0.0 (safe) to 1.0 (definite injection)
    patterns_matched: list[str]            # Which patterns triggered
    categories: list[str]                  # Which categories matched
    scan_method: str                       # "deterministic", "llm", "hybrid"
    scan_latency_ms: float = 0.0
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "score": self.score,
            "patterns_matched": self.patterns_matched,
            "categories": self.categories,
            "scan_method": self.scan_method,
            "scan_latency_ms": self.scan_latency_ms,
            "details": self.details,
        }


# ═══════════════════════════════════════════════════════════════════
# Pattern Library
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InjectionPattern:
    """A single injection detection pattern."""
    name: str
    category: str
    pattern: re.Pattern
    weight: float  # 0.0-1.0, contribution to overall score
    description: str = ""


# Build patterns at module load time
_PATTERNS: list[InjectionPattern] = []


def _build_patterns() -> list[InjectionPattern]:
    patterns = []

    # ── Instruction Override ─────────────────────────────────
    instruction_overrides = [
        (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|guidelines?|prompts?)",
         0.9, "ignore_previous_instructions"),
        (r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
         0.9, "disregard_previous"),
        (r"(new|updated|revised)\s+instructions?\s*:",
         0.8, "new_instructions"),
        (r"override\s+(the\s+)?(system|instructions?|rules?|prompt)",
         0.9, "override_system"),
        (r"forget\s+(everything|all|what)\s+(you|I)",
         0.8, "forget_context"),
        (r"do\s+not\s+follow\s+(the\s+)?(above|previous|original)",
         0.9, "do_not_follow"),
        (r"(instead|actually|really)\s*,?\s*(you\s+should|please|I\s+want\s+you\s+to)",
         0.5, "redirect_instruction"),
    ]
    for pat, weight, name in instruction_overrides:
        patterns.append(InjectionPattern(
            name=name, category="instruction_override",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))

    # ── Role Play ────────────────────────────────────────────
    role_plays = [
        (r"you\s+are\s+(now|actually|really)\s+(a|an|the)",
         0.8, "role_reassignment"),
        (r"pretend\s+(you\s+are|to\s+be)",
         0.7, "pretend_to_be"),
        (r"act\s+as\s+(a|an|if\s+you)",
         0.6, "act_as"),
        (r"(assume|take\s+on)\s+the\s+(role|persona|identity)",
         0.7, "assume_role"),
        (r"from\s+now\s+on\s*,?\s*(you|your)",
         0.7, "from_now_on"),
        (r"enter\s+(DAN|jailbreak|god|developer|admin)\s+mode",
         0.95, "jailbreak_mode"),
        (r"(DAN|jailbreak)\s+mode",
         0.9, "dan_mode"),
    ]
    for pat, weight, name in role_plays:
        patterns.append(InjectionPattern(
            name=name, category="role_play",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))

    # ── Delimiter Attacks ────────────────────────────────────
    delimiter_attacks = [
        (r"```\s*(system|instruction|prompt|override)",
         0.8, "backtick_system"),
        (r"<\|?(im_start|system|endoftext|INST)\|?>",
         0.95, "special_tokens"),
        (r"-{3,}\s*(system|new\s+instructions?|override)",
         0.7, "delimiter_system"),
        (r"<system>|</system>|<instruction>|<prompt>",
         0.85, "xml_injection_tags"),
        (r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]",
         0.9, "llama_tokens"),
        (r"</?system_prompt\s*/?>",
         0.95, "system_prompt_tags"),
        (r"-{5,}.*\n.*(approve|ignore|override|comply|everything)",
         0.7, "separator_then_instruction"),
    ]
    for pat, weight, name in delimiter_attacks:
        patterns.append(InjectionPattern(
            name=name, category="delimiter_attack",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))

    # ── Output Manipulation ──────────────────────────────────
    output_manipulations = [
        (r"respond\s+(only\s+)?with\s*(the\s+word|yes|no|true|approved)",
         0.7, "force_output"),
        (r"(output|return|say|reply\s+with)\s+(only|exactly|nothing\s+but)",
         0.6, "constrain_output"),
        (r"format\s+your\s+(response|answer|output)\s+as",
         0.3, "format_output"),  # Low weight — legitimate in some contexts
    ]
    for pat, weight, name in output_manipulations:
        patterns.append(InjectionPattern(
            name=name, category="output_manipulation",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))

    # ── Encoding Evasion ─────────────────────────────────────
    encoding_evasions = [
        (r"base64\s*:\s*[A-Za-z0-9+/]{20,}",
         0.6, "base64_payload"),
        (r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){3,}",
         0.5, "hex_encoded"),
        (r"&#\d{2,4};(&#\d{2,4};){3,}",
         0.5, "html_entity_chain"),
    ]
    for pat, weight, name in encoding_evasions:
        patterns.append(InjectionPattern(
            name=name, category="encoding_evasion",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))


    # ── Data Extraction Attempts ─────────────────────────────
    data_extraction = [
        (r"(reveal|show|display|print|output)\s+(your|the)\s+(system\s+)?prompt",
         0.9, "reveal_system_prompt"),
        (r"what\s+are\s+your\s+(instructions|rules|guidelines|directives)",
         0.8, "query_instructions"),
        (r"(output|repeat|print|show)\s+(everything|all|the\s+text)\s+(above|before|verbatim)",
         0.8, "output_verbatim"),
        (r"(copy|paste|reproduce)\s+(the|your)\s+(above|system|original)",
         0.7, "reproduce_prompt"),
        (r"(tell|show)\s+me\s+(your|the)\s+(system|original|initial)\s+(prompt|instructions)",
         0.9, "tell_system_prompt"),
    ]
    for pat, weight, name in data_extraction:
        patterns.append(InjectionPattern(
            name=name, category="data_extraction",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))

    # ── Direct Manipulation ──────────────────────────────────
    direct_manipulation = [
        (r"(approve|accept|authorize)\s+(this|the|it|immediately|right\s+now)",
         0.7, "force_approve"),
        (r"set\s+(confidence|score|rating|probability)\s+(to|=|at)\s+",
         0.9, "set_confidence"),
        (r"skip\s+(verification|compliance|review|check|validation)",
         0.85, "skip_checks"),
        (r"(bypass|circumvent|avoid)\s+(the\s+)?(governance|review|approval|verification)",
         0.9, "bypass_governance"),
        (r"without\s+(review|verification|approval|checking|compliance)",
         0.6, "without_review"),
        (r"(mark|flag|classify)\s+(this|it|the\s+case)\s+as\s+(approved|safe|low.risk|legitimate)",
         0.8, "force_classification"),
    ]
    for pat, weight, name in direct_manipulation:
        patterns.append(InjectionPattern(
            name=name, category="direct_manipulation",
            pattern=re.compile(pat, re.IGNORECASE),
            weight=weight,
        ))
    return patterns


_PATTERNS = _build_patterns()


# ═══════════════════════════════════════════════════════════════════
# Deterministic Scanner
# ═══════════════════════════════════════════════════════════════════

def _extract_text_fields(data: Any, max_depth: int = 5) -> list[str]:
    """Recursively extract all string values from a dict/list structure."""
    texts = []
    if max_depth <= 0:
        return texts
    if isinstance(data, str):
        texts.append(data)
    elif isinstance(data, dict):
        for v in data.values():
            texts.extend(_extract_text_fields(v, max_depth - 1))
    elif isinstance(data, (list, tuple)):
        for item in data:
            texts.extend(_extract_text_fields(item, max_depth - 1))
    return texts


def deterministic_scan(case_input: dict[str, Any]) -> GuardrailResult:
    """
    Scan case input using deterministic patterns.
    Returns a GuardrailResult with risk level.
    """
    t0 = time.time()

    # Extract all text from the case input
    texts = _extract_text_fields(case_input)
    combined_text = " ".join(texts)

    if not combined_text.strip():
        return GuardrailResult(
            risk="CLEAR", score=0.0, patterns_matched=[],
            categories=[], scan_method="deterministic",
            scan_latency_ms=(time.time() - t0) * 1000,
        )

    # Run all patterns
    matched_patterns = []
    matched_categories = set()
    total_score = 0.0

    for pattern in _PATTERNS:
        if pattern.pattern.search(combined_text):
            matched_patterns.append(pattern.name)
            matched_categories.add(pattern.category)
            total_score += pattern.weight

    # Normalize score to 0-1 range (cap at 1.0)
    score = min(total_score, 1.0)

    # Determine risk level
    if score >= 0.7:
        risk = "HIGH"
    elif score >= 0.3:
        risk = "AMBIGUOUS"
    else:
        risk = "CLEAR"

    latency_ms = (time.time() - t0) * 1000

    return GuardrailResult(
        risk=risk,
        score=round(score, 3),
        patterns_matched=matched_patterns,
        categories=sorted(matched_categories),
        scan_method="deterministic",
        scan_latency_ms=round(latency_ms, 2),
    )


# ═══════════════════════════════════════════════════════════════════
# LLM Classifier Interface (pluggable)
# ═══════════════════════════════════════════════════════════════════

class LLMGuardClassifier:
    """
    Interface for LLM-based guard model.
    Concrete implementation requires an LLM client.
    """

    def classify(self, text: str) -> GuardrailResult:
        """Classify text using a guard model. Override in subclass."""
        raise NotImplementedError("LLM guard classifier not configured")


class NoOpClassifier(LLMGuardClassifier):
    """Passthrough — used when LLM guard is disabled."""
    def classify(self, text: str) -> GuardrailResult:
        return GuardrailResult(
            risk="CLEAR", score=0.0, patterns_matched=[],
            categories=[], scan_method="llm_noop",
        )


# ═══════════════════════════════════════════════════════════════════
# Hybrid Guardrail (S-009 Option C)
# ═══════════════════════════════════════════════════════════════════

class InputGuardrail:
    """
    Hybrid input guardrail.

    1. Deterministic scan (< 1ms)
    2. If AMBIGUOUS, optionally escalate to LLM classifier
    3. HIGH risk → auto-escalate governance tier

    Config:
        CC_GUARDRAILS_ENABLED  — enable/disable (default: true)
        CC_GUARDRAILS_LLM      — enable LLM for ambiguous (default: false)
    """

    def __init__(
        self,
        enabled: bool | None = None,
        llm_enabled: bool | None = None,
        llm_classifier: LLMGuardClassifier | None = None,
    ):
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.environ.get("CC_GUARDRAILS_ENABLED", "true").lower() in ("true", "1")

        if llm_enabled is not None:
            self.llm_enabled = llm_enabled
        else:
            self.llm_enabled = os.environ.get("CC_GUARDRAILS_LLM", "false").lower() in ("true", "1")

        self.llm_classifier = llm_classifier or NoOpClassifier()

    def scan(self, case_input: dict[str, Any]) -> GuardrailResult:
        """
        Scan case input for injection patterns.

        Returns GuardrailResult. Caller decides what to do with HIGH risk.
        """
        if not self.enabled:
            return GuardrailResult(
                risk="CLEAR", score=0.0, patterns_matched=[],
                categories=[], scan_method="disabled",
            )

        # Phase 1: Deterministic
        result = deterministic_scan(case_input)

        # Phase 2: LLM for ambiguous (if enabled)
        if result.risk == "AMBIGUOUS" and self.llm_enabled:
            texts = _extract_text_fields(case_input)
            combined = " ".join(texts)
            try:
                llm_result = self.llm_classifier.classify(combined)
                # Merge results
                return GuardrailResult(
                    risk=llm_result.risk,
                    score=max(result.score, llm_result.score),
                    patterns_matched=result.patterns_matched + llm_result.patterns_matched,
                    categories=sorted(set(result.categories + llm_result.categories)),
                    scan_method="hybrid",
                    scan_latency_ms=result.scan_latency_ms + llm_result.scan_latency_ms,
                )
            except Exception as e:
                logger.warning("LLM guard classifier failed: %s — using deterministic result", e)
                result.details = f"LLM classifier error: {e}"
                return result

        return result
