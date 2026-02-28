# Cognitive Core

AI Application Factory — composable cognitive workflows from YAML configuration with demand-driven multi-agent coordination.

Cognitive Core lets you define AI workflows as YAML specifications that are merged at runtime from three layers (workflow structure, domain expertise, case data), executed step-by-step through eight cognitive primitives, and coordinated across agents through a demand-driven delegation protocol where agents describe what they need and the coordinator finds providers.

## Quick Start

```bash
# Clone and install core dependencies
pip install -r requirements.txt

# Run the interactive insurance claim demo (no LLM required)
python demo_insurance_claim.py

# Run the live coordinator demo (real state machine, simulated LLM)
python demo_live_coordinator.py

# Run the batch test (uses LLM if configured, simulation fallback otherwise)
python run_batch_test.py --case simple --n 5
```

### With LLM Execution

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your provider credentials

# Run with real LLM
python run_batch_test.py --case simple --n 5
python run_batch_test.py --case simple medium hard --n 5
```

## How It Works

Every workflow is assembled from three YAML layers merged at load time:

```
workflows/claim_adjudication.yaml   →  Structure: steps, transitions, loop limits
domains/claims_processing.yaml      →  Expertise: prompts, thresholds, tools, capabilities
cases/insurance_claim_simple.json   →  Data: identity and inputs for this execution
```

The engine merges these, builds a state graph, and executes step by step. Each step uses one of eight cognitive primitives (retrieve, think, generate, verify, decide, act, reflect, delegate). When a step encounters something it cannot resolve, it emits a `ResourceRequest`. The coordinator matches the need to a registered capability, dispatches a provider (another workflow, a human task, a solver), suspends the source workflow, and resumes it when results arrive.

```
Agent runs forward through steps
  ↓
Step 2: "I need the equipment schedule"     →  ResourceRequest
  ↓
Coordinator matches need → dispatches provider workflow
  ↓
Provider completes → Coordinator resumes agent at Step 2
  ↓
Agent continues with new data
```

The complexity of any execution is determined by what the case actually requires, not by what was anticipated at design time. A simple claim runs straight through. A complex claim triggers three interrupt/dispatch/resume cycles across five back-office agents.

## Project Structure

```
cognitive-core/
├── coordinator/              # Orchestration layer
│   ├── runtime.py            #   Coordinator (start/suspend/resume/complete)
│   ├── policy.py             #   Governance tiers, capability matching
│   ├── store.py              #   SQLite persistence
│   ├── tasks.py              #   Human task queue
│   ├── types.py              #   Core types (InstanceState, WorkOrder, Suspension)
│   ├── contracts.py          #   Schema validation for delegation contracts
│   ├── escalation.py         #   Escalation briefs for human review
│   ├── cli.py                #   Coordinator CLI
│   └── config.yaml           #   Governance tiers, capabilities, delegation rules
│
├── engine/                   # Execution engine
│   ├── composer.py           #   Three-layer merge, graph compilation
│   ├── nodes.py              #   Node factories for each primitive
│   ├── stepper.py            #   Step-by-step executor with interrupt detection
│   ├── state.py              #   WorkflowState, StepResult types
│   ├── tools.py              #   Tool registry (case fixtures + MCP)
│   ├── actions.py            #   Action registry for side effects
│   ├── llm.py                #   Multi-provider LLM factory
│   ├── resume.py             #   Mid-graph resume (subgraph compilation)
│   ├── trace.py              #   Lightweight execution tracing
│   ├── settlement.py         #   Deterministic settlement calculator
│   ├── governance.py         #   Governance pipeline (guardrails + all gates)
│   ├── guardrails.py         #   Input validation, prompt injection detection
│   ├── pii.py                #   PII redaction
│   ├── cost.py               #   Token and cost accounting
│   ├── kill_switch.py        #   Runtime emergency stop
│   ├── audit.py              #   Audit trail
│   ├── db.py                 #   Database backends (SQLite, Postgres)
│   ├── providers.py          #   MCP, API, vector tool providers
│   ├── retry.py              #   Retry policies with backoff
│   ├── validate.py           #   Config validation
│   └── ...                   #   (30+ modules total)
│
├── registry/                 # Primitive catalog
│   ├── primitives.py         #   Eight primitives: config, prompts, schemas
│   ├── schemas.py            #   Pydantic output schemas per primitive
│   └── prompts/              #   Prompt templates per primitive
│
├── langgraph/                # Local LangGraph shim for offline execution
│   └── graph.py              #   Minimal StateGraph/CompiledGraph
│
├── mcp_servers/              # MCP tool servers
│   └── claims_services.py    #   Claims data + settlement calculator
│
├── workflows/                # Workflow definitions (26 workflows)
├── domains/                  # Domain configurations (27 domains)
├── cases/                    # Test cases with fixture data
│   ├── fixtures/             #   MCP mock responses per case
│   └── *.json                #   Case files (simple, medium, hard, meridian)
│
├── tests/                    # Test suite
│   └── test_demand_driven.py #   Demand-driven delegation tests
│
├── demo_insurance_claim.py   # Interactive CLI demo (no deps)
├── demo_live_coordinator.py  # Live coordinator state machine demo
├── smoke_test.py             # Single-case smoke test
├── run_batch_test.py         # Multi-case batch test harness
├── llm_config.yaml           # LLM provider/model configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package configuration
├── Dockerfile                # Container build
├── Makefile                  # Development commands
└── .env.example              # Environment variable template
```

## Eight Cognitive Primitives

Every step in every workflow uses exactly one primitive:

| Primitive | Purpose | Output Contract |
|-----------|---------|-----------------|
| **retrieve** | Pull structured data from tools/APIs | `{data: {...}, sources: [...]}` |
| **think** | Analyze, reason, assess | `{analysis: str, confidence: float, resource_requests?: [...]}` |
| **generate** | Produce structured output | `{artifact: {...}}` |
| **verify** | Validate against criteria | `{conforms: bool, findings: [...]}` |
| **decide** | Binary or categorical branch | `{decision: str, reasoning: str}` |
| **act** | Execute side effects | `{action_taken: bool, result: {...}}` |
| **reflect** | Self-evaluate prior output | `{assessment: str, revisions: [...]}` |
| **delegate** | Explicit external handoff | `{need: str, context: {...}}` |

## Coordination Protocol

The demand-driven delegation protocol operates through five phases:

1. **Interrupt** — Agent produces a ResourceRequest; stepper pauses; workflow suspends
2. **Match** — Coordinator maps each need to a registered capability
3. **Dispatch** — Work orders created; providers started (workflows, human tasks, solvers)
4. **Wait** — Workflow stays suspended; providers may themselves delegate (nested chains)
5. **Resume** — All providers complete; results injected; workflow resumes at interrupted step

Three dispatch patterns: **sequential** (single need), **parallel** (independent needs dispatched simultaneously), **staged** (dependent needs dispatched in waves).

## Governance Tiers

| Tier | Behavior | Use Case |
|------|----------|----------|
| `auto` | No human review | Low-risk lookups, data retrieval |
| `log` | Record but don't gate | Medium-risk, audit trail |
| `spot_check` | Random sampling | Quality monitoring |
| `gate` | Human approval required | High-stakes decisions (claims, loans) |

Tiers are configured per domain in `coordinator/config.yaml`. The logic breaker module auto-escalates tiers when quality metrics degrade.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_PROVIDER` | For LLM execution | `azure_foundry`, `azure`, `openai`, `google`, `bedrock` |
| `AZURE_AI_FOUNDRY_ENDPOINT` | If using Azure Foundry | Project endpoint URL |
| `AZURE_AI_FOUNDRY_KEY` | If using Azure Foundry | API key |
| `OPENAI_API_KEY` | If using OpenAI | API key |
| `GOOGLE_API_KEY` | If using Google | API key |
| `CLAIMS_MCP_CMD` | For MCP (dev) | `python mcp_servers/claims_services.py` |
| `CLAIMS_MCP_URL` | For MCP (prod) | HTTP endpoint URL |
| `DATABASE_URL` | For Postgres | Connection string |

See `.env.example` for the full list.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,azure]"

# Run all checks
make check

# Run tests
make test

# Run batch validation
make batch-all

# Build container
make docker
```

## Architecture Documentation

Detailed technical documentation is in `output/`:

- `output/ARCHITECTURE.md` — Full system architecture with component details
- `output/SYSTEM.md` — System-level design decisions and constraints
- `CHANGES.md` — Session-by-session change log
