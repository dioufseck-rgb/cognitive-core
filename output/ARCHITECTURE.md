# Cognitive Core — Architecture Documentation

**Date:** February 26, 2026
**Version:** Post-restructure (stepper integration, MCP migration, fixture separation)

---

## 1. Design Philosophy

Three rules govern every design decision:

1. **YAML configures, MCP integrates, engine orchestrates.** No bespoke code per use case. A new workflow is a new YAML file, not a new Python module.
2. **The LLM reasons, the platform enforces.** Arithmetic, schema validation, routing logic, and loop limits are deterministic. The LLM provides judgment; the platform provides correctness.
3. **If it's a stable, reusable capability, it's a configured service.** Settlement calculation is an MCP tool, not an inline function. Equipment lookup is a back-office workflow, not a special-case branch.

---

## 2. Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     COORDINATOR                             │
│  Governance · Delegation · HITL · Work Orders · Persistence │
├─────────────────────────────────────────────────────────────┤
│                       ENGINE                                │
│  Primitives · Composition · State · Routing · LLM Clients   │
├─────────────────────────────────────────────────────────────┤
│                   DOMAIN CONFIGURATION                      │
│  Workflow YAML · Domain YAML · Case JSON                    │
├─────────────────────────────────────────────────────────────┤
│                    DATA + TOOLS                             │
│  MCP Servers · Fixture DB · API Providers · Vector Search   │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Coordinator (`coordinator/`)

The coordinator owns the lifecycle of every workflow instance. It decides what runs, when it pauses, who reviews it, and when it resumes. It does not know what the workflow *does* — only that it produces results that need governance.

**Key responsibilities:**
- Instance lifecycle: `start → running → suspended → resumed → completed/failed`
- Governance tier enforcement (auto / spot_check / gate / hold)
- Demand-driven delegation: detect resource requests, dispatch providers, collect results, resume
- Work order management: create, track, complete
- Human-in-the-loop: approval queues, task assignment, SLA enforcement
- Persistence: instances, suspensions, work orders, audit ledger (SQLite)

**Key types:**
- `InstanceState` — tracks one workflow execution (status, config, result, parent chain)
- `WorkOrder` — tracks one delegation (source instance → target provider)
- `Suspension` — snapshot of paused workflow (state, resume step, pending work orders)
- `Capability` — registered provider for a specific need type

**Policy engine** (`policy.py`) is purely deterministic. Given a domain, tier config, delegation policies, and workflow output, it returns routing decisions. No LLM calls.

### Layer 2: Engine (`engine/`)

The engine compiles YAML into executable graphs and runs them step by step.

**Composition pipeline:**
```
workflow.yaml + domain.yaml → merge_workflow_domain()
                                    ↓
                              merged config
                                    ↓
                            compose_workflow()
                                    ↓
                    StateGraph (LangGraph compatible)
                                    ↓
                              .compile()
                                    ↓
                            CompiledGraph
                                    ↓
                    step_execute() / step_resume()
```

**Three-layer merge** (`composer.py`): The workflow defines structure (steps, transitions). The domain defines expertise (prompts, rules, tools). `merge_workflow_domain()` resolves `${domain.X.Y}` references, producing a single merged config where every step has concrete parameters.

**Node factories** (`nodes.py`): Each primitive has a factory that creates a LangGraph node function. The node: resolves parameters from state, builds context, calls the LLM, parses the response into a typed schema, and writes the result to state. Specialized factories exist for `retrieve` (multi-source tool calling) and `act` (action execution with authorization).

**Stepper** (`stepper.py`): Replaces single-shot `invoke()` with step-by-step `stream()`. After each node completes, a callback inspects the output. If the output contains a `ResourceRequest`, the stepper raises `StepInterrupt`, which the coordinator catches to begin the delegation cycle.

**Resume** (`resume.py`): Pure functions for mid-graph resume. `compile_subgraph()` builds a StateGraph containing only steps reachable from the resume point (forward slice + backward jumps for retry loops). `prepare_resume_state()` preserves outputs from prior steps and loop counts from the suspension snapshot.

**Providers** (`providers.py`): Three provider types plug into `ToolRegistry`:
- `APIProvider` — wraps REST endpoints as tools
- `VectorProvider` — wraps vector search as tools
- `MCPProvider` / `MCPMultiProvider` — discovers tools dynamically via MCP (stdio, http, sse transports)

### Layer 3: Domain Configuration

**Workflow YAML** defines structure:
```yaml
name: claim_adjudication
steps:
  - name: intake
    primitive: retrieve
    params:
      specification: "${domain.intake.specification}"
    transitions:
      - default: coverage_analysis
  - name: settlement_recommendation
    primitive: generate
    params:
      template: "${domain.settlement.template}"
    transitions:
      - condition: "compliance_check.passed == false AND loop_count < 3"
        target: settlement_recommendation
      - default: compliance_check
```

**Domain YAML** defines expertise:
```yaml
domain_name: claims_processing
governance: gate
tools:
  - get_claim
  - get_policy
  - calculate_settlement

intake:
  specification: |
    Pull all available claim and policy data...

settlement:
  template: |
    CRITICAL — USE THE calculate_settlement TOOL...
```

**Case JSON** provides identity:
```json
{
  "claim_id": "CLM-2025-90112",
  "policy_number": "BOP-2025-34891",
  "workflow": "claim_adjudication",
  "domain": "claims_processing",
  "_ground_truth": { "net_payable": 25300, "tolerance_pct": 0 }
}
```

### Layer 4: Data + Tools

**MCP servers** expose capabilities via Model Context Protocol:
- `data_services.py` — core banking data (backed by fixture SQLite DB)
- `claims_services.py` — claims data + `calculate_settlement` deterministic tool
- `actions_server.py` — write-side actions (email, disputes, SAR)
- `compliance_server.py` — regulation search, compliance checks

**Fixture database** (`fixtures/cognitive_core.db`): SQLite with 16 tables seeded by `fixtures/db.py`. Covers members, accounts, transactions, loans, disputes, complaints, fraud scores, devices, AML alerts, check deposits, hardship cases, spending summaries, patients, regulations. The `fixtures/api.py` module registers these as tools with the same signatures as production APIs.

**Tool resolution** follows three paths:
1. **Production**: `CLAIMS_MCP_URL` → MCPProvider → http → real services
2. **Dev with MCP**: `CLAIMS_MCP_CMD` → MCPProvider → stdio → MCP server → fixture files
3. **Dev without MCP**: `create_case_registry()` → loads `cases/fixtures/*.json` directly

---

## 3. Execution Model

### 3.1 Normal Flow (No Interrupts)

```
Coordinator.start("claim_adjudication", "claims_processing", case_input)
  │
  ├─ Create InstanceState (status=RUNNING)
  ├─ Load & merge three layers
  ├─ Build tool registry
  ├─ step_execute(config, case_input, callback)
  │     │
  │     ├─ compile_workflow() → CompiledGraph
  │     └─ compiled.stream(initial_state)
  │           │
  │           ├─ [intake]           → retrieve claim & policy data
  │           ├─ [coverage_analysis] → analyze coverage applicability
  │           ├─ [damage_assessment] → assess damage amounts
  │           ├─ [subrogation_check] → evaluate recovery potential
  │           ├─ [settlement_recommendation] → generate settlement (uses calculator tool)
  │           └─ [compliance_check]  → verify against regulations
  │
  ├─ StepResult(completed=True, final_state)
  ├─ _on_completed() → governance evaluation
  └─ InstanceState(status=COMPLETED, result={...})
```

### 3.2 Interrupted Flow (Delegation)

```
step_execute running...
  │
  ├─ [coverage_analysis] output contains:
  │     resource_requests:
  │       - need: "scheduled_equipment_verification"
  │         reason: "Policy has equipment breakdown endorsement..."
  │
  ├─ callback detects ResourceRequest → StepInterrupt
  │
  └─ _WorkflowInterrupted raised
       │
       ├─ _on_step_interrupted()
       │     ├─ policy.match_needs("scheduled_equipment_verification")
       │     │     → Capability(provider_type=workflow, workflow=equipment_schedule_lookup)
       │     ├─ Create WorkOrder (source → target)
       │     ├─ Compact state → Suspension
       │     │     (resume_step="coverage_analysis", snapshot, wo_need_map)
       │     ├─ _dispatch_provider("equipment_schedule_lookup")
       │     │     → Coordinator.start(child workflow)
       │     └─ Source instance status = SUSPENDED
       │
       │  ... child workflow completes ...
       │
       ├─ _check_delegation_completion()
       │     ├─ All work orders done? → Yes
       │     ├─ Build external_input from WO results
       │     └─ Coordinator.resume(source_instance_id, external_input)
       │
       └─ step_resume(config, snapshot, "coverage_analysis", callback)
             ├─ compile_subgraph() → only reachable steps from coverage_analysis
             ├─ prepare_resume_state() → inject external_input, preserve loop_counts
             └─ compiled.stream(prepared_state)
                   │
                   ├─ [coverage_analysis] → now has equipment schedule in context
                   ├─ [damage_assessment] → continues forward
                   └─ ... completes normally
```

### 3.3 Compliance Loop

The settlement→compliance loop is the engine's loop control mechanism in action:

```
[settlement_recommendation] → calls calculate_settlement tool → $25,300
        │
        └─ transitions:
             condition: "compliance_check.passed == false AND loop_count < 3"
                → target: settlement_recommendation (retry)
             default: → compliance_check
        │
[compliance_check] → verifies against regulations
        │
        ├─ passed=true → END (loop_count irrelevant)
        └─ passed=false, loop_count < 3 → back to settlement_recommendation
                          loop_count ≥ 3 → END (with compliance failure noted)
```

Loop counts are tracked in `state["loop_counts"]` per step name. On resume from suspension, loop counts are preserved from the snapshot (except the resume step, which resets to 0).

---

## 4. Module Reference

### 4.1 Engine Modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `composer.py` | 586 | Three-layer merge, graph compilation, condition evaluation, `compile_subgraph`, `run_workflow_from_step` |
| `nodes.py` | 826 | Node factories for all 8 primitives. Trace callbacks. JSON extraction. |
| `stepper.py` | 347 | Step-by-step execution via `.stream()`. `StepInterrupt` for resource requests. |
| `resume.py` | 166 | `collect_reachable_steps`, `clamp_transitions`, `prepare_resume_state`. Pure functions, zero deps. |
| `state.py` | 188 | `WorkflowState` TypedDict. Parameter resolution (`${step.field}`). Context building. |
| `tools.py` | ~240 | `ToolRegistry` class. `create_case_registry` for fixture-backed testing. |
| `providers.py` | 715 | `APIProvider`, `VectorProvider`, `MCPProvider`, `MCPMultiProvider`. Full MCP client. |
| `llm.py` | 452 | Multi-provider LLM client. Model alias resolution from `llm_config.yaml`. |
| `actions.py` | 450 | `ActionRegistry` for Act primitive. Simulation and MCP-backed execution. |
| `agentic.py` | 449 | ReAct loop executor for agentic mode steps. |
| `tool_dispatch.py` | 462 | Dispatch layer between primitives and tool registry. |
| `domain_contract.py` | 617 | Domain contract specification and enforcement. |
| `validate.py` | 675 | Configuration validation. |
| `audit.py` | 473 | Audit trail recording (structured JSON). |
| `db.py` | 546 | SQLite persistence layer. |
| `retry.py` | 480 | LLM call retry with exponential backoff. |
| `webhooks.py` | 416 | Webhook dispatch for external event notification. |
| `config_loader.py` | — | YAML loading with environment variable interpolation. |
| `cost.py` | — | Per-call LLM cost estimation. |
| `eval_gate.py` | — | Model versioning with eval-gated deployment (baseline comparison). |
| `guardrails.py` | — | Input/output guardrails. |
| `health.py` | — | Health/readiness/startup probes. |
| `hitl_routing.py` | — | HITL routing logic. |
| `hitl_state.py` | — | HITL state management. |
| `kill_switch.py` | — | Emergency stop (domain or global). |
| `logging.py` | — | Structured logging with correlation IDs. |
| `logic_breaker.py` | — | Circuit breakers: sliding window quality monitoring, auto-tier upgrade. |
| `manifest.py` | — | Deployment manifest generation. |
| `pii.py` | — | PII detection and masking. |
| `rate_limit.py` | — | Per-provider request rate limiting. |
| `replay.py` | — | Workflow replay from audit trail. |
| `secrets.py` | — | Secret management. |
| `semantic_cache.py` | — | Two-layer LLM response cache (exact hash + vector similarity). |
| `shadow.py` | — | Shadow mode for parallel evaluation of model changes. |
| `spec_lock.py` | — | Spec version locking for reproducibility. |
| `tier.py` | — | Governance tier utilities. |

### 4.2 Coordinator Modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `runtime.py` | 2,002 | `Coordinator` class: full lifecycle. Stepper integration. Interrupt handling. Delegation dispatch. |
| `policy.py` | 602 | `PolicyEngine`: governance tier evaluation, delegation condition matching, capability need matching, contract validation. |
| `store.py` | 493 | SQLite store: instances, work orders, suspensions, audit ledger. |
| `types.py` | 327 | Core types: `InstanceState`, `WorkOrder`, `WorkOrderResult`, `Suspension`, `Capability`, `GovernanceTier`. |
| `tasks.py` | 417 | Task queue: governance approvals, human tasks, claiming, resolution, SLA. |
| `escalation.py` | 363 | Escalation rules and handling. |
| `cli.py` | 374 | CLI interface for coordinator operations. |
| `contracts.py` | 172 | Contract assertion utilities. |
| `config.yaml` | ~300 | Governance tiers, delegation policies, capability catalog, contracts. |

### 4.3 MCP Servers

| Server | Lines | Tools |
|--------|-------|-------|
| `data_services.py` | 448 | get_member, get_accounts, get_transactions, get_loans, get_dispute, get_complaint, get_fraud_score, get_devices, get_aml_alert, get_check_deposit, get_nsf_events, get_financial_goals, get_spending_summary, get_patient, get_regulation |
| `claims_services.py` | 319 | get_claim, get_policy, get_incident_report, get_claimant_history, get_equipment_schedule, get_contractor_info, get_contractor_gl_limits, get_legal_cost_estimates, **calculate_settlement** |
| `actions_server.py` | 392 | send_email (real SMTP), create_dispute, update_account_status, submit_sar, create_task, log_member_note |
| `compliance_server.py` | 229 | search_regulations, check_compliance, get_examination_findings, search_enforcement_actions |

### 4.4 Supporting Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| API server | `api/server.py` | FastAPI: POST /v1/cases, GET /v1/cases/{id}, approvals, health |
| Async worker | `api/worker.py` | Case execution (inline or Redis/arq) |
| Foundry adapter | `api/foundry_adapter.py` | Azure AI Foundry agent registration |
| Fixture DB | `fixtures/cognitive_core.db` | SQLite, 16 tables, 775+ transaction rows |
| Fixture API | `fixtures/api.py` | Registers DB tools matching production signatures |
| Fixture seeder | `fixtures/db.py` | Schema + data for all fixture tables |
| Primitive catalog | `registry/primitives.py` | Primitive configs, required/optional params |
| Prompt templates | `registry/prompts/` | classify.txt, investigate.txt, think.txt, etc. |
| Output schemas | `registry/schemas.py` | Pydantic models for each primitive's output |
| Eval runner | `evals/runner.py` | Execute eval packs, compare against criteria |
| Eval reporter | `evals/report.py` | HTML report generation |
| Eval cases | `evals/cases/` | 25 card dispute + 25 product return cases |
| LLM config | `llm_config.yaml` | Model aliases, provider settings, reverse lookup |
| Production registry | `config/production_registry.json` | Production API endpoint definitions |
| Schema glossary | `docs/schema_glossary.yaml` | Canonical field names → production field paths |
| LangGraph shim | `langgraph/graph.py` | Minimal StateGraph/CompiledGraph for offline use |
| Tests | `tests/` | 43 files, 19,434 lines |
| Archive | `archive/` | 22 domains, 21 workflows, 13 cases (previous iterations) |

---

## 5. Active Domain Configuration

### Insurance Claims (Primary)

| Layer | File | Description |
|-------|------|-------------|
| Workflow | `workflows/claim_adjudication.yaml` | 6 steps: intake → coverage → damage → subrogation → settlement → compliance |
| Domain | `domains/claims_processing.yaml` | GL/property/BOP claims. Tool declarations. Step-specific prompts. |
| Cases | `cases/insurance_claim_{simple,medium,hard}.json` | Three complexity tiers |
| Fixtures | `cases/fixtures/{simple,medium,hard}_tools.json` | MCP mock responses |
| Human responses | `cases/fixtures/human_task_responses.json` | Scripted specialist responses |

### Back-Office Skill Agents

| Workflow | Domain | Triggered By |
|----------|--------|-------------|
| `equipment_schedule_lookup.yaml` | `underwriting_records.yaml` | need: `scheduled_equipment_verification` |
| `vendor_coi_lookup.yaml` | `vendor_management.yaml` | need: `third_party_coi_retrieval` |
| `field_scheduling_optimizer.yaml` | `field_operations.yaml` | need: `adjuster_scheduling` |
| `recovery_optimizer.yaml` | `subrogation.yaml` | need: `subrogation_recovery_analysis` |

### Capability Catalog (coordinator/config.yaml)

| Need | Provider Type | Target |
|------|--------------|--------|
| `scheduled_equipment_verification` | workflow | `equipment_schedule_lookup` |
| `forensic_accounting_review` | human_task | `specialist_forensic_accounting` |
| `third_party_coi_retrieval` | workflow | `vendor_coi_lookup` |
| `adjuster_scheduling` | workflow | `field_scheduling_optimizer` |
| `subrogation_recovery_analysis` | workflow | `recovery_optimizer` |
| `coverage_specialist_review` | human_task | `coverage_specialist` |
| `independent_appraisal` | human_task | `independent_appraisal` |

---

## 6. State Management

### WorkflowState TypedDict

```python
class WorkflowState(TypedDict, total=False):
    input: dict[str, Any]           # Original case input
    step_outputs: dict[str, Any]    # {step_name: {primitive output}}
    loop_counts: dict[str, int]     # {step_name: iteration_count}
    context: str                    # Accumulated context string
    delegation: dict[str, Any]      # External input from delegations
    messages: list                  # LangGraph message history
```

### Parameter Resolution

Domain YAML references state with `${...}` syntax:
- `${input.claim_id}` → case input field
- `${intake.data.claim_details}` → output from a prior step
- `${domain.coverage_analysis.instruction}` → domain config value (resolved at merge time)

### State Compaction

When a workflow suspends, `_compact_state_for_suspension()` reduces the state to prevent prompt growth on resume. Only the most recent step outputs and essential context are preserved.

---

## 7. Governance Model

### Tiers

| Tier | HITL | Sample Rate | Use Case |
|------|------|-------------|----------|
| `auto` | None | 0% | Low-risk, high-confidence |
| `spot_check` | Post-completion review | 10% (configurable) | Normal operations |
| `gate` | Pre-completion approval | 100% | High-value or regulated |
| `hold` | Pre-execution approval | 100% | Critical or new |

### Circuit Breakers (`logic_breaker.py`)

Sliding window monitors per-(domain, primitive) quality. When low-confidence rate exceeds thresholds, the tier auto-upgrades:
- \>50% low confidence → upgrade to `spot_check`
- \>80% low confidence → upgrade to `gate`
- Auto-recovers when rate drops

### Eval-Gated Deployment (`eval_gate.py`)

Model/provider changes gated behind eval results:
1. Run eval pack against candidate
2. Compare against absolute thresholds AND stored baseline
3. Both must pass: absolute quality gates AND no >5% regression
4. Pass → approve + save new baseline. Fail → reject.

---

## 8. Testing Strategy

### Unit Tests (`tests/`)

43 test files covering every engine and coordinator module. Key test files:

| Test | What It Covers |
|------|---------------|
| `test_coordinator.py` (1,312 lines) | Full coordinator lifecycle, governance, delegation |
| `test_core.py` (1,015 lines) | Composition, node creation, workflow execution |
| `test_contracts.py` (898 lines) | Domain contract validation |
| `test_demand_driven.py` (894 lines) | Demand-driven delegation patterns |
| `test_mid_graph_resume.py` (893 lines) | Subgraph compilation, state preparation, resume |
| `test_stepper.py` (609 lines) | Step-by-step execution, interrupt detection |

### Eval Framework (`evals/`)

Two eval packs with 25 cases each:
- **Card dispute** — normal (10), edge (10), adversarial (5)
- **Product return** — normal (10), edge (10), adversarial (5)

Each case defines acceptance criteria. The runner executes workflows and compares outputs. HTML reports show pass/fail with detailed diffs.

### Batch Test (`run_batch_test.py`)

Runs insurance claim cases N times with scripted human task responses. Measures:
- Completion rate (X/N)
- Settlement accuracy vs ground truth
- Deviation percentage
- Interrupt count and types

Target performance:

| Case | Runs | Net Payable | Deviation | Interrupts |
|------|------|-------------|-----------|------------|
| Simple | 5/5 | $25,300 | 0% | 0 |
| Medium | 5/5 | $133,159 | <5% | 2 |
| Hard | TBD | ~$168,944 | <15% | 3-5 |

---

## 9. Deployment Modes

### Development (Local)

```bash
# Without MCP (fixture-backed)
python run_batch_test.py --case simple --n 1

# With MCP server
export CLAIMS_MCP_CMD="python mcp_servers/claims_services.py"
python run_batch_test.py --case simple --n 1

# API server
CC_WORKER_MODE=inline uvicorn api.server:app --reload
```

### Production

```bash
# MCP servers as HTTP endpoints
export CLAIMS_MCP_URL="https://claims-mcp.internal/mcp"
export DATA_MCP_URL="https://data-mcp.internal/mcp"

# API server with Redis worker
uvicorn api.server:app --host 0.0.0.0 --port 8080
arq api.arq_worker.WorkerSettings
```

### Azure AI Foundry

`api/foundry_adapter.py` registers Cognitive Core workflows as Foundry agents. Each workflow becomes a callable agent in the Foundry catalog.

---

## 10. Codebase Statistics

```
Engine:              31 files    14,722 lines
Coordinator:          9 files     4,813 lines
MCP Servers:          4 files     1,388 lines
API:                  6 files     1,387 lines
Fixtures:             3 files     1,279 lines
Registry:             3 files       584 lines
Evals:                6 files     1,944 lines
Tests:               43 files    19,434 lines
────────────────────────────────────────────────
Total active code:            ~26,000+ lines
Total with tests:             ~45,000+ lines

Archive:
  Domains:           22 files
  Workflows:         21 files
  Cases:             13 files

Eval cases:          50 JSON files (card dispute + product return)
Prompt templates:    9 files (one per primitive + orchestrator)
```
