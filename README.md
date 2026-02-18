# Cognitive Core

Composable AI workflows from eight cognitive primitives.
Four-layer architecture: **Workflow** × **Domain** × **Case** × **Coordinator**.
Platform-agnostic — runs on Google, Azure, OpenAI, or Bedrock.

## Quick Start

```bash
pip install -r requirements.txt --break-system-packages

# Install ONE provider:
pip install langchain-google-genai     # Google Gemini
pip install langchain-openai           # Azure OpenAI / OpenAI
pip install langchain-aws              # Amazon Bedrock

# Set provider and credentials:
export LLM_PROVIDER=google
export GOOGLE_API_KEY=your_key

# Validate (no LLM calls)
python3 engine/validate.py --root .
python3 -m unittest discover -s tests   # 486 tests

# Run a workflow
LLM_PROVIDER=google python -m engine.runner \
  -w workflows/product_return.yaml \
  -d domains/electronics_return.yaml \
  -c cases/laptop_return_suspicious.json

# Run via coordinator (governance + delegation)
LLM_PROVIDER=google python -m coordinator.cli run \
  -w product_return -d electronics_return \
  -c cases/laptop_return_suspicious.json

# Run eval pack (live LLM)
LLM_PROVIDER=google python -m evals.runner \
  --pack evals/packs/product_return.yaml --project-root .
```

## Architecture

| Layer | Purpose | Owner | Artifact |
|-------|---------|-------|----------|
| **Workflow** | Primitive sequence, transitions, routing | AI Engineers | `workflows/*.yaml` |
| **Domain** | Categories, rules, risk tier, vocabulary | SMEs + Engineers | `domains/*.yaml` |
| **Case** | Runtime input (member ID, complaint, alert) | Production APIs | `cases/*.json` |
| **Coordinator** | Governance tiers, A2A delegation, HITL | Risk / Compliance | `coordinator/config.yaml` |

## Primitives

| # | Primitive | Question | Boundary |
|---|-----------|----------|----------|
| 1 | Retrieve | What data exists? | Read |
| 2 | Classify | What is this? | Read |
| 3 | Investigate | What's true here? | Read |
| 4 | Think | What should we do? | Read |
| 5 | Verify | Does this conform? | Read |
| 6 | Generate | Write this properly | Read |
| 7 | Challenge | Can this survive? | Read |
| 8 | Act | Execute this action | **Write** |

## Enterprise Readiness

| Module | File | Purpose | Tests |
|--------|------|---------|-------|
| Retry + Fallback | `engine/retry.py` | Backoff, same-provider fallback, circuit breaker | 30 |
| Structured Logging | `engine/logging.py` | JSON lines, OTel-compatible trace_id/span_id | 26 |
| PII Redaction | `engine/pii.py` | Regex + case-entity hybrid at LLM chokepoint | 28 |
| Rate Limiting | `engine/rate_limit.py` | Per-provider semaphore + token bucket | 13 |
| Health Checks | `engine/health.py` | /health, /ready, /startup for K8s | 16 |
| Audit Trail | `engine/audit.py` | Append-only SHA-256 hash chain | 16 |
| Eval-Gated Deploy | `engine/eval_gate.py` | Absolute + regression check, CI/CD exit codes | 13 |
| Cost Tracking | `engine/cost.py` | Per-call token/cost by step and model | 13 |

All enterprise modules configured in `llm_config.yaml`.

## Governance Tiers

| Tier | HITL | Example |
|------|------|---------|
| auto | None | Spending advisor |
| spot_check | 10% sampled | Card disputes |
| gate | Pre-action review | Complaint resolution |
| hold | Expert sign-off | SAR investigation |

## Environment Variables

```bash
LLM_PROVIDER            # google | azure | azure_foundry | openai | bedrock
GOOGLE_API_KEY          # Google Gemini
AZURE_OPENAI_ENDPOINT   # Azure OpenAI endpoint
AZURE_OPENAI_API_KEY    # Azure OpenAI key
OPENAI_API_KEY          # OpenAI key
AWS_DEFAULT_REGION      # Bedrock region
LLM_DEFAULT_MODEL       # Override default model alias
LLM_TIMEOUT_SECONDS     # Per-call timeout
LLM_CONFIG_PATH         # Path to llm_config.yaml
CC_VERSION              # Service version for structured logs
```
