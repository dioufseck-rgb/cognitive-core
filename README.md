# Cognitive Core

Composable AI workflows from five cognitive primitives.
Three-layer architecture: **Workflow** × **Domain** × **Case**.

## Quick Start

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key

# Card dispute (fraud)
python -m engine.runner \
  -w workflows/dispute_resolution.yaml \
  -d domains/card_dispute.yaml \
  -c cases/card_clear_fraud.json

# SAR investigation (structuring)
python -m engine.runner \
  -w workflows/sar_investigation.yaml \
  -d domains/structuring_sar.yaml \
  -c cases/sar_structuring.json

# Regulatory impact (AVM rule)
python -m engine.runner \
  -w workflows/regulatory_impact.yaml \
  -d domains/avm_regulation.yaml \
  -c cases/avm_regulation.json

# Loan hardship (military transition)
python -m engine.runner \
  -w workflows/loan_hardship.yaml \
  -d domains/military_hardship.yaml \
  -c cases/military_hardship_reeves.json

# Nurse triage (cardiac)
python -m engine.runner \
  -w workflows/nurse_triage.yaml \
  -d domains/cardiac_triage.yaml \
  -c cases/cardiac_chest_pain.json

# ACH dispute (revoked authorization)
python -m engine.runner \
  -w workflows/dispute_resolution.yaml \
  -d domains/ach_dispute.yaml \
  -c cases/ach_revoked_authorization.json
```

## Three-Layer Architecture

```
workflows/               domains/                 cases/
  dispute_resolution ──→   card_dispute        ──→  card_clear_fraud.json
                     ──→   ach_dispute          ──→  ach_revoked_auth.json
  sar_investigation  ──→   structuring_sar      ──→  sar_structuring.json
  regulatory_impact  ──→   avm_regulation       ──→  avm_regulation.json
  loan_hardship      ──→   military_hardship    ──→  military_hardship_reeves.json
  nurse_triage       ──→   cardiac_triage       ──→  cardiac_chest_pain.json
```

**Workflow** — the cognitive pattern. Which primitives, in what order,
with what transitions. Reusable across domains. Owned by AI engineers.

**Domain** — the subject matter expertise. Categories, rules, constraints.
Domain-specific but case-independent. Owned by SMEs.

**Case** — runtime data. The specific member, transaction, patient.
Comes from production systems. Never hand-edited in prod.

Multiplication: 5 workflows × 30 domains × unlimited cases.

## Primitives

| Primitive       | Question               | Key Output Fields                   |
|-----------------|------------------------|-------------------------------------|
| **Classify**    | What is this?          | category, confidence, alternatives  |
| **Investigate** | What's true here?      | finding, hypotheses, actions        |
| **Verify**      | Does this conform?     | conforms, violations, rules_checked |
| **Generate**    | Write this properly    | artifact, constraints_checked       |
| **Challenge**   | Can this survive?      | survives, vulnerabilities, strengths|

## Agentic Capabilities

Workflows support three transition modes per step:

- **Deterministic** (`when`/`goto`) — evaluated first, no LLM call
- **Agent** (`agent_decide`) — LLM chooses among options
- **Default** — fallback if nothing else matches

Plus: loops with `max_loops`, early termination with `__end__`,
escalation paths to human specialists.

## Live Tracing

Every run shows real-time progress:

```
──────────────────────────────────────────────────────────────────────
  dispute_resolution_card_dispute  (three-layer)
  model: gemini-2.0-flash
  steps: classify_dispute_type → verify_against_records → ...
──────────────────────────────────────────────────────────────────────
  [  0.0s] 🏷️  classify_dispute_type
  [  0.1s]     ↳ calling LLM (2,341 chars)...
  [  3.2s]     ↳ response received (847 chars, 3.1s)
  [  3.2s]     → unauthorized_transaction (confidence: 0.95)
  [  3.2s]     ⚡ route → classify_resolution_fast (deterministic)
  [  3.2s] 🏷️  classify_resolution_fast
  ...
```

Disable with `--no-trace`.

## CLI Reference

```
python -m engine.runner -w WORKFLOW -d DOMAIN -c CASE [options]

Options:
  -w, --workflow    Workflow YAML
  -d, --domain      Domain YAML
  -c, --case        Case JSON/YAML
  -m, --model       Gemini model (default: gemini-2.0-flash)
  -v, --verbose     Detailed output
  -o, --output      Save full state to JSON
  --no-trace        Disable live progress
  --validate-only   Check config without running
```

## Project Structure

```
cognitive-core/
├── engine/
│   ├── composer.py     # Three-layer merge + LangGraph compilation
│   ├── nodes.py        # Primitive execution + tracing
│   ├── runner.py       # CLI with live trace
│   └── state.py        # Shared workflow state
├── registry/
│   ├── primitives.py   # Primitive registry + prompt rendering
│   ├── schemas.py      # Pydantic output contracts
│   └── prompts/        # Base prompt templates
├── workflows/          # Layer 1: cognitive patterns
├── domains/            # Layer 2: subject matter expertise
└── cases/              # Layer 3: runtime data
```
