# Cognitive Core

Composable AI workflows from eight cognitive primitives.
Three-layer architecture: **Workflow** × **Domain** × **Case**.
Platform-agnostic — runs on Google, Azure, OpenAI, or Bedrock.

## Quick Start

```bash
pip install -r requirements.txt

# Install ONE provider:
pip install langchain-google-genai     # Google Gemini
pip install langchain-openai           # Azure OpenAI / OpenAI
pip install langchain-aws              # Amazon Bedrock

# Set credentials for your provider:
export GOOGLE_API_KEY=your_key                          # Google
# — or —
export AZURE_OPENAI_ENDPOINT=https://your.openai.azure.com  # Azure
export AZURE_OPENAI_API_KEY=your_key
# — or —
export OPENAI_API_KEY=your_key                          # OpenAI
# — or —
export AWS_DEFAULT_REGION=us-east-1                     # Bedrock

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

# Loan hardship (military transition)
python -m engine.runner \
  -w workflows/loan_hardship.yaml \
  -d domains/military_hardship.yaml \
  -c cases/military_hardship_reeves.json

# Check clearing complaint with Act primitive
python -m engine.runner \
  -w workflows/complaint_resolution_act.yaml \
  -d domains/check_clearing_complaint.yaml \
  -c cases/check_clearing_complaint_diouf.json

# For live email delivery via Act primitive (optional):
export SMTP_SENDER=your-email@gmail.com
export SMTP_APP_PASSWORD=your-app-password
# SMTP_HOST and SMTP_PORT default to smtp.gmail.com:587
```

## Three-Layer Architecture

```
workflows/               domains/                 cases/
  dispute_resolution ──→   card_dispute        ──→  card_clear_fraud.json
                     ──→   ach_dispute          ──→  ach_revoked_authorization.json
  sar_investigation  ──→   structuring_sar      ──→  sar_structuring.json
  regulatory_impact  ──→   avm_regulation       ──→  avm_regulation.json
  loan_hardship      ──→   military_hardship    ──→  military_hardship_reeves.json
  nurse_triage       ──→   cardiac_triage       ──→  cardiac_chest_pain.json
  spending_advisor   ──→   debit_spending       ──→  spending_advisor_williams.json
  complaint_res_act  ──→   check_clearing       ──→  check_clearing_complaint_diouf.json
```

**Workflow** — the cognitive pattern. Which primitives, in what order,
with what transitions. Reusable across domains. Owned by AI engineers.

**Domain** — the subject matter expertise. Categories, rules, constraints.
Domain-specific but case-independent. Owned by SMEs.

**Case** — runtime data. The specific member, transaction, patient.
Comes from production systems. Never hand-edited in prod.

Multiplication: workflows × domains × unlimited cases.

## Primitives

| # | Primitive       | Question               | Key Output Fields                   | Boundary |
|---|-----------------|------------------------|-------------------------------------|----------|
| 1 | **Retrieve**    | What data exists?      | data, sources_queried, retrieval_plan | Read     |
| 2 | **Classify**    | What is this?          | category, confidence, alternatives  | Read     |
| 3 | **Investigate** | What's true here?      | finding, hypotheses, actions        | Read     |
| 4 | **Think**       | What should we do?     | thought, conclusions, decision      | Read     |
| 5 | **Verify**      | Does this conform?     | conforms, violations, rules_checked | Read     |
| 6 | **Generate**    | Write this properly    | artifact, constraints_checked       | Read     |
| 7 | **Challenge**   | Can this survive?      | survives, vulnerabilities, strengths| Read     |
| 8 | **Act**         | Execute this action    | actions_taken, authorization_checks | **Write**|

Primitives 1–7 are read-only. Only Act (8) crosses the read-write boundary,
with authorization enforcement, dry-run by default, and reversibility declarations.

## LLM Provider Configuration

The framework auto-detects your provider from environment variables.
No code changes needed to switch providers.

### Model Aliases

YAML configs and CLI use logical aliases that resolve per-provider:

| Alias      | Google            | Azure / OpenAI | Bedrock                |
|------------|-------------------|----------------|------------------------|
| `default`  | gemini-2.0-flash  | gpt-4o-mini    | claude-3.5-haiku       |
| `fast`     | gemini-2.0-flash  | gpt-4o-mini    | claude-3.5-haiku       |
| `standard` | gemini-2.5-pro    | gpt-4o         | claude-3.5-sonnet      |
| `strong`   | gemini-2.5-pro    | gpt-4o         | claude-3.5-sonnet      |

Provider-specific model names also work as pass-through:
```bash
python -m engine.runner -m gpt-4o ...       # auto-detects OpenAI/Azure
python -m engine.runner -m gemini-2.5-pro ...  # auto-detects Google
```

### Environment Overrides

```bash
LLM_PROVIDER=azure          # Force provider (skip auto-detection)
LLM_DEFAULT_MODEL=gpt-4.1   # Override what "default" resolves to
```

## Agentic Capabilities

Two execution modes:

- **Sequential** (production): Steps in predetermined order with
  deterministic or LLM-assisted routing
- **Agentic** (discovery): LLM orchestrator chooses step sequence
  at runtime using hub-and-spoke graph

Sequential workflows support three transition modes per step:

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
  provider: azure  model: gpt-4o-mini
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
  -m, --model       Model alias (default/fast/standard/strong) or
                    provider-specific name (gpt-4o, gemini-2.0-flash)
  -p, --provider    Force provider: google, azure, openai, bedrock
  -v, --verbose     Detailed output
  -o, --output      Save full state to JSON
  --no-trace        Disable live progress
  --validate-only   Check config without running
```

## Project Structure

```
cognitive-core/
├── engine/
│   ├── llm.py          # Provider factory — single point of LLM construction
│   ├── composer.py      # Three-layer merge + LangGraph compilation
│   ├── nodes.py         # Primitive execution + tracing
│   ├── agentic.py       # Hub-and-spoke orchestrator for agentic mode
│   ├── runner.py        # CLI with live trace
│   ├── state.py         # Shared workflow state + parameter resolution
│   ├── actions.py       # Action registry with authorization enforcement
│   ├── tools.py         # Tool registry for Retrieve primitive
│   └── providers.py     # API, Vector, and MCP tool providers
├── registry/
│   ├── primitives.py    # Primitive registry + prompt rendering
│   ├── schemas.py       # Pydantic output contracts
│   └── prompts/         # Base prompt templates (9 files)
├── mcp_servers/
│   ├── compliance_server.py  # Read-side MCP server
│   └── actions_server.py     # Write-side MCP server
├── workflows/           # Layer 1: cognitive patterns (7 sequential + 2 agentic)
├── domains/             # Layer 2: subject matter expertise (10 configs)
├── cases/               # Layer 3: runtime data (9 case files)
└── requirements.txt
```
