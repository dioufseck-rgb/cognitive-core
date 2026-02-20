# Cognitive Core — Reliability Audit

## Status: 13/15 Fixed · 1,097 Unit Tests + 20 Live Eval Cases · 2 Deferred (design-level)

Systematic review of 8,345 lines across coordinator/ and engine/,
focused on failure modes that compound at scale (100s of delegations).

---

## CRITICAL

### C1. Delegation Depth Limit — ✅ FIXED

`MAX_DELEGATION_DEPTH = 20` in `start()`. Raises `DelegationDepthExceeded`
with last 5 chain entries. Checked before any state is created.

### C2. Synchronous Blocking Delegation Stack — ✅ MITIGATED

Depth limit (C1) caps at 20 levels — safe for Python stack + LangGraph
context. Full async dispatch deferred to Phase 4.

### C3. State Snapshot Bloat — ✅ FIXED

`_compact_state_for_suspension()` strips `raw_response` (5-15KB/step),
`prompt_used` (2-10KB/step), truncates large retrieve payloads and
artifacts. `MAX_SNAPSHOT_BYTES = 512KB` warning. 80KB → 15KB typical.

### C4. Transaction Boundaries — ✅ FIXED

`store.transaction()` context manager. `_commit()` is a no-op inside
transaction block. Atomic COMMIT/ROLLBACK prevents orphaned suspensions.

---

## HIGH

### H1. SQLite Busy Timeout — ✅ FIXED

`PRAGMA busy_timeout=5000` — 5s retry on lock contention.

### H2. Silent F&F Handler Failures — ✅ FIXED

Both F&F and blocking failures log `delegation_handler_failed` to ledger
with policy, target, mode, error.

### H3. Idempotency Key Scoping — ✅ ALREADY DONE

Keys include `created_at` or `resume_nonce`, scoping to instance lifecycle.

### H4. Resume Nonce Validation — ✅ ALREADY DONE

`_resume_after_delegation` passes `suspension.resume_nonce` to `resume()`.

### H5. Governance Re-eval on Resume — ✅ ALREADY DONE

`_on_completed(instance, final_state, is_resume=True)` skips governance.

### H6. LLM Parse Failure Circuit Breaker — ✅ FIXED

**Nodes:** `_parse_failed: True` flag in error output.

**Router:** `_evaluate_condition` handles `_parse_failed` as built-in
condition. Composite router circuit breaker: parse failed + max_loops
exhausted → END instead of continuing with garbage.

Workflow authors can write:
```yaml
transitions:
  - when: "_parse_failed"
    goto: retry_step
```

---

## MEDIUM

### M1. Correlation Chain Limit — ✅ ALREADY DONE

`list_instances(limit=500)` with `LIMIT ?` in SQL.

### M2. Strict JSON Serialization — ✅ ALREADY DONE

`COGNITIVE_CORE_STRICT=1` triggers strict check on state snapshots.
Raises `TypeError` instead of silently converting objects to strings.

### M3. Stuck Instance Detection — ✅ FIXED

`store.find_stuck_instances(max_running_seconds)` — RUNNING beyond timeout.
`store.find_orphaned_suspensions()` — suspended with all work orders done.

### M4. Tool Registry Heuristic — 📋 DEFERRED

Low risk with `get_*` convention. Fix: explicit `_tool_sources` key.

### M5. LLM Rate Limiting — 📋 DEFERRED

Needs threading infrastructure. Fix: semaphore in `create_llm()`.

---

## Test Coverage

1,097 unit tests covering all enterprise modules. 20 live eval cases
validating end-to-end execution with real LLM providers across 3 workflows
(claim_intake, fraud_screening, damage_assessment). Key reliability areas:
- Contract validation (TypedDict boundaries, CLI integration)
- Delegation depth limit (at/under max)
- Transaction atomicity (commit, rollback, multi-save)
- Parse failure detection and routing
- Handler failure logging
- Stuck/orphaned instance detection
- Full 3-workflow coordination lifecycle
- Escalation brief generation on governance suspension
