"""
Cognitive Core — Live LLM Test Harness

Runs synthetic claim cases through Claude via the Anthropic API.
Each case has deterministic expected outputs. The harness:
  1. Renders the prompt exactly as the engine would
  2. Calls Claude
  3. Parses the structured JSON response
  4. Grades against expected values

Usage: python tests/test_live_llm.py
"""

import json
import os
import re
import subprocess
import sys
import time
import yaml

# ─── Config ──────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"
API_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 2048

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Prompt Templates ────────────────────────────────────────
def load_template(primitive):
    path = os.path.join(_base, "registry", "prompts", f"{primitive}.txt")
    with open(path) as f:
        return f.read()

TEMPLATES = {}
for p in ["classify", "verify", "think", "investigate", "generate", "retrieve"]:
    try:
        TEMPLATES[p] = load_template(p)
    except FileNotFoundError:
        pass

# ─── API Call ────────────────────────────────────────────────
def call_claude(prompt, temperature=0.0):
    """Call Claude via curl (no pip deps needed)."""
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    })

    result = subprocess.run(
        ["curl", "-s", "-X", "POST", API_URL,
         "-H", "Content-Type: application/json",
         "-H", "x-api-key: dummy",  # handled by proxy
         "-H", "anthropic-version: 2023-06-01",
         "-d", payload],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")

    resp = json.loads(result.stdout)
    if "content" not in resp:
        raise RuntimeError(f"API error: {json.dumps(resp, indent=2)}")

    return resp["content"][0]["text"]


def extract_json_from_response(text):
    """Extract JSON object from LLM response."""
    # Try code block first
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    brace = text.find('{')
    if brace == -1:
        raise ValueError(f"No JSON in response: {text[:200]}")

    depth = 0
    in_str = False
    esc = False
    for i in range(brace, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == '\\' and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[brace:i+1])

    raise ValueError(f"Unterminated JSON: {text[:200]}")


# ─── Test Case Definitions ───────────────────────────────────
def load_case(filename):
    path = os.path.join(_base, "cases", "synthetic", filename)
    with open(path) as f:
        return json.load(f)

def load_domain(filename):
    path = os.path.join(_base, "domains", filename)
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Step Runners ────────────────────────────────────────────
def run_classify(domain_cfg, case_data, step_name):
    """Run a classify step and return parsed output."""
    cfg = domain_cfg[step_name]
    template = TEMPLATES["classify"]

    # Build context from case data (simulates retrieve output)
    context = json.dumps({k: v for k, v in case_data.items() if not k.startswith("_")}, indent=2)

    prompt = template.format(
        context=context,
        input=f"Claim data:\n{context}",
        categories=cfg["categories"],
        criteria=cfg["criteria"],
        additional_instructions="Follow the DETERMINISTIC RULES exactly. No interpretation.",
        confidence_threshold=cfg.get("confidence_threshold", "0.8"),
    )

    raw = call_claude(prompt)
    return extract_json_from_response(raw), raw


def run_verify(domain_cfg, case_data, step_name, prior_outputs):
    """Run a verify step."""
    cfg = domain_cfg[step_name]
    template = TEMPLATES["verify"]

    # Build subject from case data + prior outputs
    subject_parts = []
    if "classify_claim_type" in prior_outputs:
        subject_parts.append(f"Claim type: {prior_outputs['classify_claim_type']['category']}")
    for k, v in case_data.items():
        if k.startswith("get_"):
            for field, val in v.items():
                subject_parts.append(f"{k}.{field}: {val}")

    prompt = template.format(
        context=json.dumps(case_data, indent=2),
        input="\n".join(subject_parts),
        rules=cfg["rules"],
        additional_instructions="Check ALL rules. Report ALL violations. This is deterministic.",
    )

    raw = call_claude(prompt)
    return extract_json_from_response(raw), raw


def run_think(domain_cfg, case_data, step_name, prior_outputs):
    """Run a think step."""
    cfg = domain_cfg[step_name]
    template = TEMPLATES["think"]

    context_parts = []
    if "classify_claim_type" in prior_outputs:
        context_parts.append(f"Claim type: {prior_outputs['classify_claim_type']['category']}")
    context_parts.append(f"Claim amount: {case_data['get_claim']['amount']}")
    context_parts.append(f"Policy tenure months: {case_data['get_policy']['tenure_months']}")
    context_parts.append(f"Prior claims count: {case_data['get_policy']['prior_claims']}")
    context_parts.append(f"Flags: {case_data['get_claim']['flags']}")

    prompt = template.format(
        context="\n".join(context_parts),
        input="\n".join(context_parts),
        instruction=cfg["question"],
        additional_instructions=cfg.get("constraints", ""),
    )

    raw = call_claude(prompt)
    return extract_json_from_response(raw), raw


def run_investigate(domain_cfg, case_data, step_name):
    """Run an investigate step."""
    cfg = domain_cfg[step_name]
    template = TEMPLATES["investigate"]

    context = json.dumps({k: v for k, v in case_data.items() if not k.startswith("_")}, indent=2)

    evidence_parts = []
    if "get_claim_history" in case_data:
        evidence_parts.append(f"Prior claims: {json.dumps(case_data['get_claim_history']['claims'])}")
    if "get_flags" in case_data:
        evidence_parts.append(f"Flag count: {case_data['get_flags']['count']}")
        evidence_parts.append(f"Flag types: {case_data['get_flags']['types']}")
    if "get_claim" in case_data:
        evidence_parts.append(f"Claim amount: {case_data['get_claim']['amount']}")
    if "get_policy" in case_data:
        evidence_parts.append(f"Days since policy start: {case_data['get_policy'].get('days_active', 'unknown')}")

    prompt = template.format(
        context=context,
        input="\n".join(evidence_parts),
        question=cfg["question"],
        scope=cfg["question"],
        hypotheses=cfg["hypotheses"],
        additional_instructions="Follow the DETERMINISTIC pattern checks exactly.",
    )

    raw = call_claude(prompt)
    return extract_json_from_response(raw), raw


# ─── Grading ─────────────────────────────────────────────────
def grade(case_id, step, field, expected, actual, op="equals"):
    """Grade a single assertion. Returns (pass, message)."""
    if op == "equals":
        passed = actual == expected
        return passed, f"{'✓' if passed else '✗'} {case_id}/{step}.{field}: expected={expected}, got={actual}"
    elif op == "gte":
        passed = actual >= expected
        return passed, f"{'✓' if passed else '✗'} {case_id}/{step}.{field}: expected>={expected}, got={actual}"
    elif op == "in":
        passed = actual in expected
        return passed, f"{'✓' if passed else '✗'} {case_id}/{step}.{field}: expected in {expected}, got={actual}"
    elif op == "range":
        lo, hi = expected
        passed = lo <= actual <= hi
        return passed, f"{'✓' if passed else '✗'} {case_id}/{step}.{field}: expected [{lo},{hi}], got={actual}"
    elif op == "contains":
        if isinstance(actual, list):
            passed = any(expected in str(v) for v in actual)
        else:
            passed = expected in str(actual)
        return passed, f"{'✓' if passed else '✗'} {case_id}/{step}.{field}: expected contains '{expected}', got={actual}"
    return False, f"? {case_id}/{step}.{field}: unknown operator {op}"


# ─── Test Scenarios ──────────────────────────────────────────

CLAIM_INTAKE_TESTS = [
    {
        "case_file": "sc_001_simple_approve.json",
        "domain_file": "synthetic_claim.yaml",
        "assertions": [
            ("classify_claim_type", "category", "liability", "equals"),
            ("check_eligibility", "conforms", True, "equals"),
            ("assess_risk", "recommendation", "auto_approve", "equals"),
        ],
    },
    {
        "case_file": "sc_002_physical_damage.json",
        "domain_file": "synthetic_claim.yaml",
        "assertions": [
            ("classify_claim_type", "category", "physical_damage", "equals"),
            ("check_eligibility", "conforms", True, "equals"),
        ],
    },
    {
        "case_file": "sc_003_high_value_flagged.json",
        "domain_file": "synthetic_claim.yaml",
        "assertions": [
            ("classify_claim_type", "category", "theft", "equals"),
            ("check_eligibility", "conforms", True, "equals"),
            ("assess_risk", "recommendation", ["enhanced_review", "refer_to_siu"], "in"),
        ],
    },
    {
        "case_file": "sc_005_inactive_policy.json",
        "domain_file": "synthetic_claim.yaml",
        "assertions": [
            ("classify_claim_type", "category", "liability", "equals"),
            ("check_eligibility", "conforms", False, "equals"),
        ],
    },
    {
        "case_file": "sc_006_wrong_coverage.json",
        "domain_file": "synthetic_claim.yaml",
        "assertions": [
            ("classify_claim_type", "category", "theft", "equals"),
            ("check_eligibility", "conforms", False, "equals"),
        ],
    },
    {
        "case_file": "sc_007_multi_violation.json",
        "domain_file": "synthetic_claim.yaml",
        "assertions": [
            ("classify_claim_type", "category", "medical", "equals"),
            ("check_eligibility", "conforms", False, "equals"),
        ],
    },
]

FRAUD_SCREENING_TESTS = [
    {
        "case_file": "fs_003_medium_risk.json",
        "domain_file": "synthetic_fraud.yaml",
        "assertions": [
            ("classify_fraud_risk", "category", "medium_risk", "equals"),
        ],
    },
    {
        "case_file": "fs_004_high_risk_siu.json",
        "domain_file": "synthetic_fraud.yaml",
        "assertions": [
            ("classify_fraud_risk", "category", "high_risk", "equals"),
            ("investigate_patterns", "finding", "significant_concerns", "equals"),
            ("investigate_patterns", "recommendation", "refer_to_siu", "equals"),
        ],
    },
]

DAMAGE_TESTS = [
    {
        "case_file": "da_002_moderate.json",
        "domain_file": "synthetic_damage.yaml",
        "assertions": [
            ("classify_damage_severity", "category", "moderate", "equals"),
            ("verify_documentation", "conforms", True, "equals"),
        ],
    },
    {
        "case_file": "da_004_major_missing_docs.json",
        "domain_file": "synthetic_damage.yaml",
        "assertions": [
            ("classify_damage_severity", "category", "major", "equals"),
            ("verify_documentation", "conforms", False, "equals"),
        ],
    },
]


# ─── Main ────────────────────────────────────────────────────
def run_all():
    total = 0
    passed = 0
    failed = 0
    errors = 0
    results = []

    all_tests = (
        [("CLAIM_INTAKE", t) for t in CLAIM_INTAKE_TESTS]
        + [("FRAUD_SCREENING", t) for t in FRAUD_SCREENING_TESTS]
        + [("DAMAGE_ASSESSMENT", t) for t in DAMAGE_TESTS]
    )

    print(f"\n{'═'*70}")
    print(f"  COGNITIVE CORE — LIVE LLM TEST RUN")
    print(f"  Model: {MODEL}")
    print(f"  Cases: {len(all_tests)}")
    print(f"{'═'*70}\n")

    for workflow_name, test in all_tests:
        case_id = test["case_file"].replace(".json", "")
        case_data = load_case(test["case_file"])
        domain = load_domain(test["domain_file"])
        meta = case_data.get("_meta", {})

        print(f"─── {case_id} ({workflow_name}) ───")
        print(f"    {meta.get('description', '')[:80]}")

        outputs = {}

        for step, field, expected, op in test["assertions"]:
            total += 1
            try:
                # Run the step if not already cached
                if step not in outputs:
                    t0 = time.time()
                    if step.startswith("classify_"):
                        outputs[step], raw = run_classify(domain, case_data, step)
                    elif step.startswith("check_") or step.startswith("verify_"):
                        outputs[step], raw = run_verify(domain, case_data, step, outputs)
                    elif step.startswith("assess_"):
                        outputs[step], raw = run_think(domain, case_data, step, outputs)
                    elif step.startswith("investigate_"):
                        outputs[step], raw = run_investigate(domain, case_data, step)
                    else:
                        print(f"    ⚠ Unknown step type: {step}")
                        errors += 1
                        continue
                    elapsed = time.time() - t0
                    print(f"    [{elapsed:.1f}s] {step} → {json.dumps(outputs[step], default=str)[:120]}")

                actual = outputs[step].get(field)
                ok, msg = grade(case_id, step, field, expected, actual, op)
                results.append((ok, msg))
                print(f"    {msg}")
                if ok:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                errors += 1
                results.append((False, f"✗ {case_id}/{step}.{field}: ERROR — {e}"))
                print(f"    ✗ {case_id}/{step}.{field}: ERROR — {e}")

        print()

    # ─── Summary ─────────────────────────────────────────────
    print(f"{'═'*70}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed, {errors} errors")
    pct = (passed / total * 100) if total > 0 else 0
    print(f"  PASS RATE: {pct:.0f}%")
    print(f"{'═'*70}")

    if failed > 0 or errors > 0:
        print(f"\n  FAILURES:")
        for ok, msg in results:
            if not ok:
                print(f"    {msg}")

    return failed + errors == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
