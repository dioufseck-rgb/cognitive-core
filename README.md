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
python3 -m unittest discover -s tests   # 1,097 tests

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

## API Server

```bash
pip install fastapi uvicorn arq redis

# Dev mode (in-process, no Redis)
CC_WORKER_MODE=inline uvicorn api.server:app --port 8080

# Production (arq + Redis worker)
uvicorn api.server:app --host 0.0.0.0 --port 8080  # API
python -m api.arq_worker                             # Worker

# Submit a case
curl -X POST http://localhost:8080/v1/cases \
  -H "Content-Type: application/json" \
  -d '{"workflow":"product_return","domain":"electronics_return","case_input":{"member_id":"M123","complaint":"broken laptop"}}'

# Poll status
curl http://localhost:8080/v1/cases/{instance_id}

# List pending approvals
curl http://localhost:8080/v1/approvals
```

## Container Deployment

```bash
docker build -f Dockerfile.api -t cognitive-core-api .
docker build -f Dockerfile.worker -t cognitive-core-worker .

# Run
docker run -p 8080:8080 -e LLM_PROVIDER=google -e GOOGLE_API_KEY=... cognitive-core-api
```

## Architecture

| Layer | Purpose | Owner | Artifact |
|-------|---------|-------|----------|
| **Workflow** | Primitive sequence, transitions, routing | AI Engineers | `workflows/*.yaml` |
| **Domain** | Natural-language policy: what to look for, how to handle it | SMEs + Engineers | `domains/*.yaml` |
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

## Enterprise Modules (1,097 tests)

| Module | File | Purpose | Tests |
|--------|------|---------|-------|
| Retry + Fallback | `engine/retry.py` | Backoff, same-provider fallback, circuit breaker | 30 |
| Structured Logging | `engine/logging.py` | JSON lines, OTel-compatible trace_id/span_id | 26 |
| PII Redaction | `engine/pii.py` | Regex + case-entity hybrid at LLM chokepoint | 30 |
| Rate Limiting | `engine/rate_limit.py` | Per-provider semaphore + token bucket | 13 |
| Health Checks | `engine/health.py` | /health, /ready, /startup for K8s | 16 |
| Audit Trail | `engine/audit.py` | Append-only SHA-256 hash chain + tiered payload | 27 |
| Eval-Gated Deploy | `engine/eval_gate.py` | Absolute + regression check, CI/CD exit codes | 13 |
| Cost Tracking | `engine/cost.py` | Budget caps, conservative unknown pricing | 21 |
| API Server | `api/server.py` | FastAPI REST endpoints for case submission | 37 |
| Worker Backends | `api/worker.py` | Inline, ThreadPool, arq+Redis | — |
| Config Loader | `engine/config_loader.py` | Azure App Config → overlay → env vars | 35 |
| Secrets | `engine/secrets.py` | Key Vault + Managed Identity, env var fallback | 21 |
| Kill Switches | `engine/kill_switch.py` | Runtime disable per domain/workflow/primitive | 30 |
| Spec-Locking | `engine/manifest.py` | SHA-256 manifest with full YAML snapshots | 28 |
| Guardrails | `engine/guardrails.py` | Prompt injection: regex + LLM hybrid | 36 |
| Logic Breakers | `engine/logic_breaker.py` | Sliding window quality monitor + tier upgrade | 30 |
| State Replay | `engine/replay.py` | Checkpoint snapshots + replay from any step | 18 |
| Webhooks | `engine/webhooks.py` | Suspension notifications (Teams, Slack, generic) | 17 |
| Semantic Cache | `engine/semantic_cache.py` | Exact-match + vector similarity, on/off config | 35 |
| HITL Routing | `engine/hitl_routing.py` | Capability-based approval queue routing | 28 |

## Environment Configuration

```bash
CC_ENV=dev|staging|prod     # Active profile (loads config/{env}.yaml overlay)
CC_WORKER_MODE=inline|thread|arq   # Worker backend
CC_PROJECT_ROOT=.           # Project root for YAML resolution

# Azure (optional — falls back to env vars)
AZURE_APP_CONFIG_ENDPOINT=  # Azure App Configuration endpoint
AZURE_KEY_VAULT_URL=        # Key Vault for secrets

# LLM Provider
LLM_PROVIDER=google|azure|azure_foundry|openai|bedrock
GOOGLE_API_KEY=             # Google Gemini
AZURE_OPENAI_ENDPOINT=      # Azure OpenAI
AZURE_OPENAI_API_KEY=
OPENAI_API_KEY=
```

## Governance Tiers

| Tier | HITL | Example |
|------|------|---------|
| auto | None | Spending advisor |
| spot_check | 10% sampled | Card disputes |
| gate | Pre-action review | Complaint resolution |
| hold | Expert sign-off | SAR investigation |

## Escalation Briefs

When a workflow suspends (governance gate or quality gate), the coordinator
builds a structured escalation brief for the human reviewer:

| Section | Purpose |
|---------|---------|
| Case Summary | What case this is (ID, amount, description) |
| Determinations | What automation decided at each step, with confidence |
| Uncertainties | Where automation was unsure (low confidence, conflicting signals) |
| Focus Questions | Specific questions for the reviewer to answer |
| Evidence | All data gathered — reviewer doesn't re-investigate |
| Priority | High/medium/standard based on uncertainty count |

The brief makes the human faster: legwork is done, ambiguity is identified,
and the reviewer has a checklist instead of a blank slate.

## Domain Philosophy

Domain specs are **natural-language policy documents**, not pseudo-code.
Write them like you're training a smart new hire:

```yaml
classify_fraud_risk:
  categories: |
    - low_risk: No fraud flags on the claim.
    - medium_risk: One or two fraud flags.
    - high_risk: Three or more fraud flags.
```

The LLM reads the policy and applies judgment. The framework handles routing,
artifact validation, audit trails, and escalation. More precise specs get
higher automation rates (85% → 95%). When the LLM is unsure, it escalates
with a structured brief.

## Live Eval Harness

```bash
# Run all synthetic cases with live LLM
python scripts/eval_live.py --all --auto-approve

# Run specific workflow
python scripts/eval_live.py --workflow fraud_screening --auto-approve

# Run single case with verbose output
python scripts/eval_live.py --case sc_001_simple_approve --auto-approve -v
```

The eval harness validates structural invariants (routing correctness,
artifact schemas, parse reliability) and reports semantic checks as advisory.

