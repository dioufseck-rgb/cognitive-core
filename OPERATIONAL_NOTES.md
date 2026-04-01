# Cognitive Core — Operational Notes

> **Version:** v0.1.0-technical-preview  
> **Date:** March 2026  
> **Audience:** Architects and engineers evaluating or deploying Cognitive Core

This document states clearly what works, what is experimental, what assumptions exist, and what not to rely on yet. It builds more trust with a serious architect than any amount of aspirational documentation.

---

## What is production-ready

These components have been exercised across multiple domain packs and are suitable for production use at appropriate scale:

**Execution engine**
- Eight cognitive primitives (`retrieve`, `classify`, `investigate`, `deliberate`, `verify`, `generate`, `govern`, `orchestrate`) with validated prompt templates and Pydantic output schemas
- Configuration-driven workflow execution — workflow YAML, domain YAML, case JSON — with no per-use-case application code
- Domain reference resolution (`${domain.section.field}`) at merge time; runtime reference resolution at execution time
- Conditional transitions, loop controls, and step-level temperature configuration

**Governance pipeline**
- Four-tier governance model: `auto` / `spot_check` / `gate` / `hold`
- Governance gate suspension and resume — human decision injects into the coordinator and workflow continues from the correct step
- HITL task queue with configurable SLA, priority, and queue routing
- Governance tier is upward-only — a domain may escalate but never de-escalate within a run

**Coordinator**
- Workflow instance lifecycle: `start` / `resume` / `checkpoint` / `terminate`
- SQLite-backed state persistence across process restarts
- Action ledger with append semantics and SHA-256 hash chain for tamper detection
- Ledger verification: `store.verify_ledger(instance_id)` → `{valid, first_invalid_entry, entry_count}`
- Work order publication to task queue on governance suspension

**API layer**
- FastAPI server with HITL endpoints: `GET /workorder`, `POST /decision`, `POST /evidence`
- SSE action ledger stream: `GET /instances/{id}/stream` — all six event types
- HTML trace page: `GET /instances/{id}/trace` — Watch / Input / Result modes with live SSE, HITL form, and trace export
- Ledger verification endpoint: `GET /instances/{id}/verify`

**Domain library**
- Seven production-grade domain packs: consumer-lending, fraud-investigation, content-moderation, ecommerce-returns, eligibility-check, compliance-review, clinical-triage
- Seven coordinator templates: simple, sequential-lifecycle, hub-and-spoke, two-stage-review, fire-and-forget, wait-for-result, parallel-handlers
- Five workflow patterns: eligibility-determination, adversarial-review, investigation-and-reporting, triage-and-escalation, compliance-and-conformance

**Retry and resilience**
- Configurable retry with exponential backoff on LLM calls (max 3 attempts, 1s/2s/4s)
- Retry on: timeout, 429, 500, 503
- Circuit breaker: N consecutive failures → open circuit → fail fast
- Same-provider fallback model (e.g., gemini-2.0-flash → gemini-2.0-pro)

**Input validation**
- Pydantic case input validation at `coordinator.start()` — clean structured errors before any LLM work begins
- Size limits enforced: 512 KB total payload, 64 KB per field, 32 K chars per string
- Domain scaffold reference validation: step names caught at load time, not mid-workflow

---

## What is experimental

These components exist and work in the happy path but have not been hardened for production edge cases. Use with care.

**`coordinator/optimizer.py` — dispatch optimization**  
Experimental dispatch optimizer that ranks handler candidates by cost/latency/quality. The ranking model is heuristic, not calibrated. In production, use simple round-robin or fixed assignment until this is validated on real traffic.

**`coordinator/federation.py` — multi-coordinator federation**  
Coordinator-to-coordinator delegation across process boundaries. The protocol is defined and the happy path works; error handling for network partitions and coordinator unavailability is not complete. Do not use in production without your own circuit breaking.

**`coordinator/hardening.py` — partial**  
- Hash chain: shipped and production-ready (see above)
- DDR (Demand-Driven Resumption) audit: not yet shipped. The interface exists but the audit trail for demand-driven delegation is not complete.

**`coordinator/resilience.py` — partial**  
- Retry: shipped and production-ready (see above)
- Staleness detection: not yet shipped. Instances that go stale (e.g., a delegated handler never returns) are not automatically detected and surfaced. Implement your own SLA monitoring against the task queue until this is complete.
- Oscillation detection: interface defined, not yet exercised in production.

**`analytics/` — causal DAG and SDA policy**  
The causal DAG registry and SDA policy models are experimental. They work in the fraud demo but have not been validated against real policy data. Treat the analytics layer as a proof-of-concept until domain experts have reviewed the models for your use case.

**Multi-provider LLM routing**  
The framework supports Anthropic, OpenAI, and Google providers. Cross-provider routing (e.g., classify on Claude, generate on GPT-4o) works but has not been tested under mixed-provider load. Stick to a single provider per deployment until tested.

---

## What is not yet implemented

**Installer**  
`pip install cognitive-core` is not yet available on PyPI. Install from source: `pip install -e ".[runtime,api]"`.

The build backend is `hatchling`. If you encounter build errors in a restricted network environment (hatchling requires network access during build), switch to setuptools by editing `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"
```

**Role-based access control**  
The REST API has no authentication or RBAC. Version 0.1 assumes you deploy behind your own API gateway with auth. Do not expose the API directly to the internet.

**Multi-tenant deployment**  
All instances share a single SQLite database and a single coordinator process. Tenant isolation requires either separate deployments or a database-level tenancy scheme that is not yet built.

**PostgreSQL performance tuning**  
The framework supports PostgreSQL via the `CC_DB_BACKEND=postgres` environment variable, but the schema has not been tuned for high-volume production use. Missing: connection pooling configuration, query plan analysis, index coverage for high-throughput queries. For high-volume production use, engage a DBA before cutting over.

**Real-time fraud scoring**  
The fraud-investigation domain pack demonstrates multi-workflow delegation and governance. It does not include real-time scoring (velocity checks, device fingerprinting, graph-based fraud rings). The workflow is correct; the scoring signals are stubs. See the domain pack README for scope.

**Async workflow execution**  
The coordinator executes workflows synchronously in a thread pool. Long-running workflows (>5 minutes) will hold a thread for their duration. True async execution (suspend to DB, resume from queue) is designed but not yet implemented.

**`pip install -e ".[runtime,api,mcp]"` extras**  
The `[api]` and `[mcp]` extras exist in `pyproject.toml` but have not been verified to produce clean installs in all environments. If you hit dependency issues, install manually from `requirements.txt`.

---

## Assumptions

**Single-process deployment**  
The coordinator is a singleton within a process. Running multiple processes against the same database without coordination will produce race conditions. For multi-process deployments, put a queue in front and route all instances to a single coordinator process.

**SQLite for low-to-medium volume**  
SQLite is appropriate for development and low-volume production (< 100 concurrent workflows). Switch to PostgreSQL for anything above that. The coordinator uses `CC_DB_BACKEND=postgres` and a standard connection string via `CC_DB_URL`.

**LLM provider availability**  
There is no fallback provider. If your LLM provider is unavailable, workflows will fail after retry exhaustion. Plan for provider outages in your deployment architecture.

**Case inputs fit in memory**  
Case inputs are loaded into memory and passed through the workflow as Python dicts. Very large inputs (> 512 KB) are rejected at validation. Binary data (PDFs, images) should be stored externally and referenced by URL or ID in the case input.

**Governance decisions are synchronous**  
The HITL decision endpoints (`POST /instances/{id}/decision`) trigger synchronous workflow resumption in a thread pool. If your LLM provider is slow, the HTTP response may return before the workflow completes. Poll `GET /instances/{id}` for final status.

---

## Known issues

**HTML trace page HITL re-render**  
After a human decision is submitted via the trace page, the SSE stream reconnects. In rare cases, completed step entries that arrived while the SSE was disconnected may not be replayed if `since_id` is not advanced correctly. Workaround: reload the trace page — it replays the full ledger.

**Idempotency key collisions on retry**  
If a workflow step is retried after a transient failure but the original attempt partially committed (e.g., the ledger write succeeded but the state update did not), the idempotency key may block the retry. This surfaces as a silent skip, not an error. Mitigation: the circuit breaker will catch persistent failures within a few attempts.

**SQLite WAL mode not enabled by default**  
Concurrent reads during a long-running workflow may experience brief locks. Enable WAL mode for your production SQLite database: `PRAGMA journal_mode=WAL`. This is not set automatically because it is a persistent database setting.

**Large delegation chains**  
The `MAX_DELEGATION_DEPTH` guard prevents infinite delegation loops but does not prevent wide fan-out (many handlers in parallel). Very wide fan-out (> 20 handlers) may exhaust the thread pool. The thread pool size defaults to 4 workers; adjust `CC_WORKERS` for your workload.

---

## Change log for v0.1.0-technical-preview

- Added: HITL endpoints (`GET /workorder`, `POST /decision`, `POST /evidence`)
- Added: SSE action ledger stream (`GET /instances/{id}/stream`)
- Added: HTML trace page (`GET /instances/{id}/trace`) with Watch / Input / Result modes
- Added: Ledger verification endpoint (`GET /instances/{id}/verify`)
- Added: Case input Pydantic validation at `coordinator.start()` boundary
- Added: Six-path smoke test suite (`tests/smoke/`)
- Added: Integration example (`examples/integration_example.py`)
- Added: MCP server template (`examples/mcp_template.py`)
- Added: General-purpose API server (`cognitive_core/api/server.py`)
- Existing: Hash chain on action ledger (SHA-256, genesis constant, tamper detection)
- Existing: Retry with exponential backoff and circuit breaker (`engine/retry.py`)
- Existing: Seven domain packs, seven coordinator templates, five workflow patterns
