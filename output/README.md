# Cognitive Core

An AI workflow engine that composes LLM-driven workflows from YAML configuration. Configure with YAML, integrate with MCP, let the engine orchestrate.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run the simple insurance claim (smoke test)
python run_batch_test.py --case simple --n 1

# Run all three complexity tiers
python run_batch_test.py --case simple medium hard --n 5

# Start the API server (dev mode, no Redis)
CC_WORKER_MODE=inline uvicorn api.server:app --reload --port 8080
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLM_PROVIDER` | LLM backend | `azure_foundry`, `google`, `openai`, `bedrock` |
| `LLM_DEFAULT_MODEL` | Override default model | `gpt-4o`, `gemini-2.5-pro` |
| `CLAIMS_MCP_CMD` | Claims MCP server (stdio, dev) | `python mcp_servers/claims_services.py` |
| `CLAIMS_MCP_URL` | Claims MCP server (http, prod) | `https://claims-mcp.internal/mcp` |
| `DATA_MCP_URL` | Data services MCP (http, prod) | `https://data-mcp.internal/mcp` |
| `CC_WORKER_MODE` | API worker mode | `inline` (dev) or `arq` (prod) |

## How It Works

Every workflow is defined by three files:

1. **Workflow YAML** (`workflows/`) — Structure: steps, transitions, loop limits
2. **Domain YAML** (`domains/`) — Expertise: prompts, rules, tool declarations
3. **Case JSON** (`cases/`) — Data: identity and input for a specific instance

The engine merges these three layers, builds a LangGraph state machine, and executes it step by step. At each step, an LLM reasons using one of eight cognitive primitives. If a step can't proceed without external input, it produces a `ResourceRequest`. The coordinator dispatches whatever is needed — another workflow, a human specialist, an external service — then resumes the original workflow when results arrive.

```
Workflow YAML ─┐
               ├─ merge → StateGraph → step-by-step execution
Domain YAML  ──┤                         ↕
               │                    Coordinator
Case JSON ─────┘                    (governance, delegation, HITL)
```

## Eight Cognitive Primitives

Every step in every workflow uses exactly one primitive:

| Primitive | Purpose | Example |
|-----------|---------|---------|
| **Retrieve** | Pull data from tools/APIs | Get claim details, policy data |
| **Classify** | Categorize with confidence | Claim type, fraud risk level |
| **Think** | Analyze and reason | Coverage analysis, damage assessment |
| **Investigate** | Deep-dive with evidence | Forensic review, root cause analysis |
| **Generate** | Produce structured output | Settlement recommendation, reports |
| **Verify** | Check against rules | Compliance verification, policy limits |
| **Act** | Execute side effects | Send notification, update system |
| **Route** | Decide next step | Complexity-based branching |

Primitives are defined in `registry/primitives.py` with prompt templates in `registry/prompts/`.

## Project Structure

```
cognitive_core/
├── engine/              # Runtime: primitives, composition, state, LLM clients
│   ├── composer.py      #   Three-layer merge, graph compilation
│   ├── nodes.py         #   Node factories for each primitive
│   ├── stepper.py       #   Step-by-step executor with interrupt detection
│   ├── providers.py     #   MCP, API, and vector tool providers
│   ├── tools.py         #   Tool registry (unified interface)
│   ├── llm.py           #   Multi-provider LLM client
│   ├── resume.py        #   Mid-graph resume (subgraph compilation)
│   └── ...              #   Audit, retry, guardrails, PII, cache, etc.
│
├── coordinator/         # Orchestration: governance, delegation, HITL
│   ├── runtime.py       #   Coordinator class (start/suspend/resume/approve)
│   ├── policy.py        #   Policy engine (governance tiers, need matching)
│   ├── store.py         #   SQLite persistence
│   ├── types.py         #   Core types (InstanceState, WorkOrder, Suspension)
│   └── config.yaml      #   Governance tiers, capabilities, contracts
│
├── mcp_servers/         # MCP tool servers (FastMCP)
│   ├── data_services.py #   Banking data (members, accounts, transactions)
│   ├── claims_services.py#  Claims data + settlement calculator
│   ├── actions_server.py#   Write-side actions (email, disputes, SAR)
│   └── compliance_server.py# Regulations, examination findings
│
├── api/                 # FastAPI HTTP server + async worker
├── registry/            # Primitive catalog + prompt templates + schemas
├── fixtures/            # SQLite test data (16 tables, 775+ rows)
├── evals/               # Evaluation framework (50 test cases)
├── cases/               # Active case files + fixture data
├── domains/             # Active domain configs
├── workflows/           # Active workflow configs
├── config/              # Environment configs (dev, prod, LLM aliases)
├── tests/               # 43 test files, 19K+ lines
├── archive/             # Previous domains/workflows/cases (reference)
└── docs/                # Design documents
```

## MCP Integration

Tools are exposed as MCP servers. The engine discovers and calls them through the standard Model Context Protocol. In dev, servers run locally via stdio. In production, they're HTTP endpoints.

```bash
# Run claims MCP server standalone
python mcp_servers/claims_services.py

# Run with HTTP transport
python mcp_servers/claims_services.py --http --port 8300
```

The `calculate_settlement` tool on the claims server is deterministic — it does arithmetic that the LLM should never do itself. The LLM provides structured input (line items, deductibles, sublimits, coinsurance), and the tool returns exact numbers.

## Governance

The coordinator enforces four governance tiers:

| Tier | Behavior |
|------|----------|
| `auto` | No human review |
| `spot_check` | Random sampling (configurable rate) |
| `gate` | Human approval required before completion |
| `hold` | Human approval required before execution |

Tiers are configured per domain in `coordinator/config.yaml`. The `logic_breaker` module can auto-upgrade tiers when quality degrades.

## Delegation

When a workflow step needs something it doesn't have, it produces a `ResourceRequest`. The coordinator:

1. Matches the need to a registered capability
2. Creates a work order
3. Dispatches the provider (another workflow or human task)
4. Suspends the source workflow
5. Resumes when all providers complete

This is demand-driven: the front-office workflow starts first and pulls what it needs, rather than back-office processes pushing results forward.

## Running Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Eval packs
python -m evals --pack card_dispute
python -m evals --pack product_return

# Batch test (insurance claims)
python run_batch_test.py --case simple --n 5
```

## Configuration

LLM models are configured in `llm_config.yaml` with logical aliases (default, fast, standard, strong) mapped to provider-specific model IDs. No Python code changes needed to switch providers or models.

Production tool endpoints are configured in `config/production_registry.json`.
