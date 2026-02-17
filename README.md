# Cognitive Core

Composable AI workflows from eight cognitive primitives.
Four-layer architecture: **Workflow** × **Domain** × **Case** × **Coordinator**.
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

# Run through the engine (single workflow)
python -m engine.runner \
  -w workflows/dispute_resolution.yaml \
  -d domains/card_dispute.yaml \
  -c cases/card_clear_fraud.json

# Run through the coordinator (governance + delegation)
python -m coordinator.cli run \
  -w dispute_resolution \
  -d card_dispute \
  -c cases/card_clear_fraud.json
```

## Four-Layer Architecture

```
workflows/               domains/                 cases/
  dispute_resolution ──→   card_dispute    [spot] ──→  card_clear_fraud.json
                     ──→   ach_dispute     [spot] ──→  ach_revoked_authorization.json
  sar_investigation  ──→   structuring_sar [hold] ──→  sar_structuring.json
  regulatory_impact  ──→   avm_regulation  [gate] ──→  avm_regulation.json
  loan_hardship      ──→   military_hardship[gate]──→  military_hardship_reeves.json
  nurse_triage       ──→   cardiac_triage  [gate] ──→  cardiac_chest_pain.json
  spending_advisor   ──→   debit_spending  [auto] ──→  spending_advisor_williams.json
  complaint_res_act  ──→   check_clearing  [spot] ──→  check_clearing_complaint_diouf.json

coordinator/
  config.yaml        ──→  governance tiers, delegation policies, contracts, capabilities
```

| Layer | Purpose | Owner | Changes |
|-------|---------|-------|---------|
| **Workflow** | Primitive sequence, transitions, routing | AI Engineers | Rarely |
| **Domain** | Categories, rules, risk tier, need vocabulary | SMEs + Engineers | Per use case |
| **Case** | Intake trigger (slim: member ID, complaint, alert) | Production APIs | Every execution |
| **Coordinator** | Governance tiers, A2A delegation, HITL, SLA | Risk / Compliance / Ops | Per policy change |

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

Primitives 1–7 are read-only (cognitive phases). Act (8) crosses the read-write
boundary with authorization enforcement, dry-run by default, and reversibility
declarations. The engine structurally prevents Act from executing with pending
delegations or unresolved work orders.

## Data Architecture

Case data is separated into **slim intake triggers** (what arrives) and a
**service registry** of 16 API-shaped tools (what gets looked up).

Three-tier sourcing — the engine auto-selects the best available:

| Tier | Source | When |
|------|--------|------|
| 1 | MCP Server (production APIs) | `DATA_MCP_URL` set |
| 2 | Fixture DB (16 SQLite tables) | `cognitive_core.db` exists |
| 3 | Case passthrough (legacy) | Neither above |

The Retrieve primitive calls `get_member`, `get_transactions`, `get_fraud_score`
etc. regardless of which tier is active. Zero changes to switch.

## Runtime Coordinator

The coordinator is the fourth layer — manages multi-workflow execution,
governance, and cross-workflow delegation through brokered asynchronous
communication (A2A).

**No persistent agents.** Workflow instances are short-lived: born, execute,
produce output, die. What persists is state — case files, action ledger,
instance registry.

### Governance Tiers

Every domain declares a tier. The coordinator applies the posture:

| Tier | HITL | Example |
|------|------|---------|
| `auto` | None | Spending advisor |
| `spot_check` | 10% sampled post-completion | Card disputes |
| `gate` | Mandatory pre-action review | Complaint resolution |
| `hold` | Mandatory expert sign-off | SAR investigation |

### Delegation (A2A)

Workflows don't know each other exist. The coordinator evaluates output
against delegation policies and spawns new instances when conditions match.
Typed contracts define the interface. Correlation chains link everything
for audit.

### CLI

```
python -m coordinator.cli run -w WORKFLOW -d DOMAIN -c CASE [options]
python -m coordinator.cli stats
python -m coordinator.cli chain INSTANCE_ID
python -m coordinator.cli ledger [--instance ID] [--correlation ID]
```

## LLM Provider Configuration

Auto-detects provider from environment variables. No code changes to switch.

| Alias      | Google            | Azure / OpenAI | Bedrock                |
|------------|-------------------|----------------|------------------------|
| `default`  | gemini-2.0-flash  | gpt-4o-mini    | claude-3.5-haiku       |
| `fast`     | gemini-2.0-flash  | gpt-4o-mini    | claude-3.5-haiku       |
| `standard` | gemini-2.5-pro    | gpt-4o         | claude-3.5-sonnet      |
| `strong`   | gemini-2.5-pro    | gpt-4o         | claude-3.5-sonnet      |

```bash
LLM_PROVIDER=azure          # Force provider
LLM_DEFAULT_MODEL=gpt-4.1   # Override "default" alias
```

## Execution Modes

- **Sequential** (production): Predetermined step order with deterministic
  or LLM-assisted routing, fast-paths, investigation loops, escalation
- **Agentic** (discovery): LLM orchestrator chooses step sequence at runtime.
  Prototype in agentic, crystallize to sequential for production.

## Project Structure

```
cognitive-core/
├── engine/                    # Core execution engine
│   ├── llm.py                 # Provider factory (Google/Azure/OpenAI/Bedrock/Ollama)
│   ├── composer.py            # Three-layer merge + LangGraph compilation
│   ├── nodes.py               # Primitive execution + tracing
│   ├── agentic.py             # Hub-and-spoke orchestrator for agentic mode
│   ├── runner.py              # Engine CLI with live trace
│   ├── state.py               # Workflow state + parameter resolution
│   ├── actions.py             # Action registry with authorization enforcement
│   ├── tools.py               # Tool registry for Retrieve primitive
│   └── providers.py           # API, Vector, and MCP tool providers
├── coordinator/               # Fourth layer: governance + A2A
│   ├── runtime.py             # Coordinator: start/resume/checkpoint/terminate
│   ├── policy.py              # Policy engine: tiers, delegation, needs, contracts
│   ├── store.py               # SQLite persistence for instances + work orders
│   ├── types.py               # Data structures for all coordinator concepts
│   ├── cli.py                 # Coordinator CLI
│   └── config.yaml            # Governance tiers, delegation policies, contracts
├── registry/
│   ├── primitives.py          # Primitive registry + prompt rendering
│   ├── schemas.py             # Pydantic output contracts
│   └── prompts/               # Base prompt templates (9 files)
├── fixtures/
│   ├── api.py                 # Service registry (16 tools)
│   ├── db.py                  # Fixture database builder
│   └── cognitive_core.db      # SQLite with member data for all domains
├── mcp_servers/
│   ├── data_services.py       # MCP server exposing 16 data tools
│   ├── compliance_server.py   # Regulation search MCP server
│   └── actions_server.py      # Write-side MCP server (email, credits)
├── workflows/                 # Layer 1: cognitive patterns (7 sequential + 2 agentic)
├── domains/                   # Layer 2: expertise + governance tier (10 configs)
├── cases/                     # Layer 3: slim intake triggers (9 case files)
├── demo.sh                    # 10 progressive demo use cases
└── requirements.txt
```

## Engine CLI Reference

```
python -m engine.runner -w WORKFLOW -d DOMAIN -c CASE [options]

Options:
  -w, --workflow    Workflow YAML
  -d, --domain      Domain YAML
  -c, --case        Case JSON/YAML
  -m, --model       Model alias or provider-specific name
  -p, --provider    Force provider: google, azure, openai, bedrock
  -v, --verbose     Detailed output
  -o, --output      Save full state to JSON
  --no-trace        Disable live progress
  --validate-only   Check config without running
```
