# Compliance Review

**Pattern:** P07 — Compliance and Conformance Review  
**Overlays:** exception, governance  
**Coordinator:** simple  

Reviews whether a case, decision, artifact, or business process conforms to policy, regulation, or statute. Verify-first design enables early exit for clean conformance — most routine cases close after the verify step without investigation or deliberation.

## What it does

Six steps, two conditional branches:

1. **Retrieve** — pulls the record, artifact, or case being reviewed
2. **Verify** — checks against explicit numbered rules immediately
3. **Investigate** — fires only when violations found or conformance is ambiguous; analyses severity, pattern, and remediation requirements
4. **Deliberate** — produces a finding level (conforms / minor_violation / material_violation / critical_violation) with warrant
5. **Generate** — produces a compliance memo with violations, remediation requirements, and observations
6. **Govern** — routes to SPOT_CHECK (clean), GATE (any violation), or HOLD (material or critical)

**Early exit:** When verify finds full conformance with no ambiguity, the workflow skips investigation and deliberation and generates the memo directly. Clean cases run faster and consume fewer tokens.

## Files

```
compliance-review/
  compliance_review.yaml              ← fill in your rules, finding options, memo format
  workflow.yaml            ← the workflow (do not edit)
  coordinator_config.yaml  ← governance queues and SLAs
  run.py                   ← runner script
  cases/
    example_reviews.json   ← 3 cases: clean, material violation, minor violation
```

## Setup

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your_key_here
python library/domain-packs/compliance-review/run.py
```

## Customising

Open `compliance_review.yaml` and fill in:

| Section | What to change |
|---------|---------------|
| `gather_subject.specification` | What records to retrieve for this review type |
| `verify_conformance.rules` | Your explicit numbered rules |
| `verify_conformance.early_exit_condition` | When to skip investigation |
| `investigate_violations.scope` | What to analyse when rules fail |
| `deliberate_finding.instruction` | Your finding levels and warrant requirements |
| `generate_memo.requirements` | Required sections in your compliance memo |
| `govern_outcome.governance_context` | When each tier applies |

## The three example reviews

| ID | Type | Expected path |
|----|------|---------------|
| COMP-001 | Clean prime loan approval | conforms → early exit → SPOT_CHECK |
| COMP-002 | Denial missing adverse action notice | material_violation → investigate → HOLD |
| COMP-003 | Fair lending review with incomplete documentation | minor_violation → investigate → GATE |

## Adapting to different compliance domains

This pack ships with generic rules that apply across most compliance reviews. Replace `verify_conformance.rules` with your domain-specific requirements:

**Fair lending / ECOA:** Replace with disparate treatment and disparate impact rules, adverse action notice requirements, ECOA-specific prohibited bases.

**BSA / AML:** Replace with SAR filing thresholds, CIP requirements, beneficial ownership rules, 314(a)/(b) obligations.

**HIPAA:** Replace with PHI access controls, breach notification requirements, minimum necessary standard, BAA requirements.

**Model governance:** Replace with SR-11-7 model risk management requirements, validation frequency, challenger model requirements.

## Scaffold boundary

The scaffold covers rule definitions, investigation scope, finding levels, memo format, and governance thresholds — the large majority of compliance review variation.

Structural changes requiring workflow edits:
- Adding a parallel legal review lane (wait-for-result coordinator template)
- A two-stage review where first-line produces a draft and second-line approves (two-stage-review coordinator template)
- Adding a remediation tracking workflow that fires after the memo is issued (fire-and-forget coordinator template)
