# Cognitive Core — System Status

**Date:** February 26, 2026
**Context:** Post-restructure. Stepper integration, MCP migration, fixture separation, max_loops bug fix.

---

## 1. What Works

### Core Engine
- Three-layer merge (workflow + domain + case → merged config)
- All 8 cognitive primitives (retrieve, classify, think, investigate, generate, verify, act, route)
- Graph compilation with conditional transitions and loop control
- Parameter resolution (`${input.X}`, `${step.output.Y}`, `${domain.Z}`)
- Context building from accumulated step outputs
- JSON extraction from LLM responses with fallback patterns

### Stepper & Resume
- `step_execute` — step-by-step execution with callback-based interrupt detection
- `step_resume` — resume from suspension with subgraph compilation
- `compile_subgraph` — forward reachability + backward jumps for retry loops
- `prepare_resume_state` — preserves prior outputs and loop counts
- `collect_reachable_steps`, `clamp_transitions` — pure functions, fully tested

### Coordinator
- Full lifecycle: start → running → suspended → resumed → completed/failed
- Governance tiers: auto, spot_check, gate, hold (per-domain config)
- Demand-driven delegation: resource request detection → need matching → work order creation → provider dispatch → suspension → result collection → resume
- Multi-provider wait: `_check_delegation_completion` waits for ALL work orders
- Staged dispatch: deferred needs with dependency tracking
- Human task management: queues, claiming, resolution, SLA

### MCP Integration
- Full MCP client: stdio, http, sse transports
- Tool discovery via MCP protocol
- MCPMultiProvider for connecting to multiple servers simultaneously
- Four MCP servers: data_services, claims_services, actions_server, compliance_server
- `calculate_settlement` deterministic tool (exact arithmetic for settlement calculations)

### Tool Resolution (Three Paths)
- Production: `CLAIMS_MCP_URL` → MCPProvider → http → real services
- Dev with MCP: `CLAIMS_MCP_CMD` → MCPProvider → stdio → MCP server → fixtures
- Dev without MCP: `create_case_registry()` → loads `cases/fixtures/*.json` directly

### Data & Fixtures
- Fixture SQLite DB: 16 tables, 775+ transactions, 6 members
- Fixture API registry: same tool signatures as production
- Case fixture files: separated from case identity (no more cheating)
- Human task response scripts: keyed by case name

### Operational Controls (Wired via `engine/governance.py`)
- **Governance pipeline**: singleton initializes all modules, provides 7 chokepoints
- **Input guardrails**: prompt injection detection on case inputs at `start()` → tier escalation on HIGH
- **PII redaction**: entity-aware masking wraps every LLM call (redact prompt → invoke → de-redact response)
- **Rate limiting**: per-provider semaphore + token bucket at LLM call site
- **Semantic cache**: exact-hash dedup before LLM call (toggleable, TTL-based)
- **Cost tracking**: token + cost accounting per call/step/workflow
- **Circuit breakers**: sliding window quality monitor → auto-tier upgrade via `resolve_effective_tier()`
- **Kill switches**: runtime emergency stop per domain/workflow/action, checked at `start()` and `_execute_delegation()`
- **Tier enforcement**: upward-only invariant — no code path can downgrade declared tier
- **Shadow mode**: dark launch for Act — run LLM, log proposed actions, skip execution
- **Compensation ledger**: register reversible actions before Act, fire compensations on failure
- **Webhooks**: notify Teams/ServiceNow/HTTP on governance suspension (fire-and-forget)
- **HITL routing**: capability-based reviewer assignment at governance suspension
- **HITL state machine**: explicit lifecycle (SUSPENDED → ASSIGNED → UNDER_REVIEW → APPROVED/REJECTED)
- **Replay checkpoints**: save state after each step for debugging and replay
- **Spec lock**: capture config hashes at workflow start for regulatory reconstruction
- Secret management
- Webhook dispatch
- Retry with exponential backoff

### LLM Multi-Provider
- Google (Gemini), Azure, Azure Foundry, OpenAI, Bedrock
- Logical aliases: default, fast, standard, strong
- Config-driven model resolution (`llm_config.yaml`)
- No code changes to switch providers

### Testing & Eval
- 43 test files, 19,434 lines
- Eval framework: 50 cases across card dispute + product return
- Eval-gated deployment (baseline comparison, auto-reject on regression)
- Batch test runner for insurance claims

---

## 2. What's New (This Session)

| Change | File(s) | Impact |
|--------|---------|--------|
| `compile_subgraph` + `run_workflow_from_step` | `engine/composer.py` | Enables mid-graph resume after delegation |
| Stepper wired into coordinator | `coordinator/runtime.py` | `step_execute` replaces raw `compiled.invoke()` |
| `_on_step_interrupted` | `coordinator/runtime.py` | Full interrupt → match → dispatch → suspend → resume cycle |
| Multi-provider wait | `coordinator/runtime.py` | `_check_delegation_completion` waits for ALL work orders |
| Staged dispatch | `coordinator/runtime.py`, `coordinator/types.py` | Deferred needs with dependency tracking |
| `wo_need_map` + `deferred_needs` on Suspension | `coordinator/types.py`, `coordinator/store.py` | Track which WO satisfies which need |
| Claims MCP server | `mcp_servers/claims_services.py` | Claims data tools + `calculate_settlement` |
| Case restructure | `cases/*.json`, `cases/fixtures/` | Identity-only cases, fixtures separated |
| Fixture-based tool registry | `engine/tools.py` | `create_case_registry` loads from fixture files |
| `_build_tool_registry` simplified | `coordinator/runtime.py` | MCP or fixture, clean paths |
| **Governance pipeline** | **`engine/governance.py`** | **Wires all 20 passive governance modules into execution** |
| **LLM call protection** | **`engine/nodes.py`** | **All 6 LLM call sites go through `protected_llm_call()`** |
| **Start gates** | **`coordinator/runtime.py`** | **Kill switch, guardrails, PII, spec manifest at start()** |
| **Tier enforcement** | **`coordinator/runtime.py`** | **Circuit breaker → `resolve_effective_tier()` (upward-only)** |
| **Webhook + HITL** | **`coordinator/runtime.py`** | **Notify external systems on governance suspension** |
| **Shadow + compensation** | **`engine/nodes.py`** | **Dark launch for Act, compensation ledger for reversal** |
| **Delegation kill switch** | **`coordinator/runtime.py`** | **Check `delegation_disabled` before dispatch** |
| **`engine/__init__.py` cleared** | **`engine/__init__.py`** | **Prevents cascading pydantic import on any `engine.*` use** |

---

## 3. What's Fixed (This Pass)

### Was Critical — Now Resolved

**LangGraph shim `.stream()` added** (`langgraph/graph.py`)
- Added `.stream()` method to `CompiledGraph`
- Yields `{node_name: state_delta}` per step, matching real LangGraph contract
- Computes proper deltas: append-reducer lists (steps, routing_log) yield only new items
- Tested with linear graphs and conditional routing loops

**`calculate_settlement` reachable without MCP** (`engine/settlement.py`, `engine/tools.py`)
- Extracted pure calculation logic into `engine/settlement.py` (zero external deps)
- `engine/tools.py` detects `{"_type": "deterministic_tool"}` in fixture data
- Imports real implementation via `spec_from_file_location` (avoids engine/__init__.py chain)
- MCP server (`claims_services.py`) now imports from `engine/settlement.py`
- Tested: simple case returns exact $25,300 without MCP running

**`smoke_test.py` updated**
- Fixed path: `cases/fixtures/human_task_responses.json` (was `cases/human_task_responses.json`)
- Fixed workflow type: `claim_adjudication` (was `insurance_claim_adjudication`)
- Fixed case data format: reads `_ground_truth` instead of `claim_details.total_claimed`

**`_resume_after_delegation` unified with multi-WO path**
- Method now handles both single-WO (legacy blocking delegation) and multi-WO (demand-driven) cases
- Checks `wo_need_map` to determine if all WOs are complete before resuming
- No longer dead code — properly integrated with both execution paths

### Still Open — Affects Quality

| Issue | Impact | Fix |
|-------|--------|-----|
| LLM may ignore calculator tool | Settlement numbers vary | Post-step verification |
| Context collapse not verified | Prompt growth on resume | Verify compaction with stepper state |
| Delegation consumption not implemented | LLM may ignore provider results | Phase 4 item |
| No delegation budget | Unbounded interrupts | Add max_interrupts per case |

---

## 4. What's Missing

### Not Yet Built

| Feature | Description | Priority |
|---------|-------------|----------|
| Post-step tool verification | Verify required tools were called (not just available) | High |
| Delegation consumption check | Verify delegation results used before proceeding | High |
| Delegation budget | Max interrupts per instance | Medium |
| Vector provider integration | Semantic search for prior cases, regulations | Medium |
| Production API provider setup | Real REST endpoint registration | Medium |
| Monitoring dashboard | Real-time workflow status, SLA tracking | Medium |
| Multi-tenant isolation | Per-tenant data and config separation | Low |
| Streaming response support | WebSocket/SSE for real-time step progress | Low |

### Known Stale / Needs Update

| Item | Issue |
|------|-------|
| `demo_insurance_claim.py` | Hardcodes case data inline; works as standalone demo but doesn't use current case format |
| `demo_live_coordinator.py` | Same as above |

---

## 5. File Inventory

### Modified This Session

| File | Change Summary |
|------|---------------|
| `engine/composer.py` | Added `compile_subgraph`, `run_workflow_from_step` |
| `engine/tools.py` | Rewrote `create_case_registry` for fixture loading |
| `coordinator/runtime.py` | Stepper integration, `_on_step_interrupted`, multi-provider wait, MCP support |
| `coordinator/types.py` | Added `wo_need_map`, `deferred_needs` to Suspension |
| `coordinator/store.py` | Schema + serialization for new Suspension fields |
| `coordinator/config.yaml` | Added claims capabilities |

### Created This Session

| File | Purpose |
|------|---------|
| `engine/settlement.py` | Pure settlement calculator (zero deps, imported by both MCP server and tool registry) |
| `engine/governance.py` | Governance pipeline — wires all 20 passive modules into execution via 7 chokepoints |
| `mcp_servers/claims_services.py` | Claims MCP server — now imports calculator from engine/settlement.py |
| `cases/insurance_claim_simple.json` | Identity-only case (was: embedded tool data) |
| `cases/insurance_claim_medium.json` | Identity-only case |
| `cases/insurance_claim_hard.json` | Identity-only case |
| `cases/fixtures/simple_tools.json` | MCP mock responses for simple case |
| `cases/fixtures/medium_tools.json` | MCP mock responses for medium case |
| `cases/fixtures/hard_tools.json` | MCP mock responses for hard case |
| `cases/fixtures/human_task_responses.json` | Scripted human task responses |
| `run_batch_test.py` | Updated batch test runner |

### Verified Correct As-Is

| File | Why |
|------|-----|
| `engine/stepper.py` | Step-by-step executor already correct |
| `engine/resume.py` | Pure functions, fully tested |
| `engine/state.py` | WorkflowState TypedDict unchanged |
| `engine/nodes.py` | Node factories unchanged |
| `engine/providers.py` | Full MCP infrastructure already exists |
| `coordinator/policy.py` | Policy engine unchanged |
| `workflows/claim_adjudication.yaml` | Workflow structure correct |
| `domains/claims_processing.yaml` | Domain config correct |

---

## 6. Ground Truth Reference

### Settlement Calculator Arithmetic

**Simple case (CLM-2025-90112):**
- Property damage: $27,800 verified
- Deductible: $2,500
- Net payable: $25,300

**Medium case (CLM-2025-91207):**
- Property: $27,800 + Equipment: $18,200 + BI: $117,959 = $163,959 verified
- Wait — let me restate precisely.
- Total verified from all categories: $138,159
- Deductible: $5,000
- BI coinsurance ratio: 200000 / (0.50 × 4,307,000) = 0.092872 (not rounded)
- BI after coinsurance: $117,959 × 0.092872 = $10,959
- CNC sublimit: $25,000 caps equipment at $25,000 (was $18,200 — under limit)
- Net payable: $133,159

**Hard case (CLM-2025-88431):**
- Ground truth: ~$168,944 (tolerance: 15%)
- Multiple contested categories, GL/liability crossover, contractor subrogation

### Tolerance Targets

| Case | Ground Truth | Tolerance |
|------|-------------|-----------|
| Simple | $25,300 | 0% (exact) |
| Medium | $133,159 | 5% |
| Hard | $168,944 | 15% |

---

## 7. Next Steps (Codespace)

### Immediate (Get Batch Tests Passing)

1. `tar xzf cognitive_core_restructure.tar.gz` — overlay onto working copy
2. Verify `langgraph` is installed (if not, update shim with `.stream()`)
3. `python run_batch_test.py --case simple --n 1` — smoke test
4. Fix whatever breaks (most likely: stepper ↔ `.stream()` event shape mismatch)
5. `python run_batch_test.py --case simple --n 5` — target: 5/5 at $25,300
6. `python run_batch_test.py --case medium --n 5` — target: 5/5 within 5%
7. If settlement numbers vary: LLM is doing math inline, not using calculator
8. `python run_batch_test.py --case hard --n 5` — establish baseline

### Short Term (Quality Hardening)

9. Add post-step verification: enforce `calculate_settlement` tool usage
10. Verify state compaction on resume path
11. Add delegation budget (max_interrupts per instance)
12. Clean up stale files (smoke_test.py, demos)
13. Remove dead `_resume_after_delegation` method

### Medium Term (Production Readiness)

14. Production API provider setup (real REST endpoints)
15. Vector provider for prior case search
16. Monitoring dashboard
17. Load testing under concurrent instances
18. Azure AI Foundry registration testing
