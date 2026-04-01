# Changelog

## [0.1.0-technical-preview] — 2026-03-31 — Open Technical Preview

> Target standard: A skeptical, technically strong architect can discover Cognitive Core,
> install it, understand it, run it, adapt it, and conclude it is one of the most serious
> governed AI frameworks available.

### Added — Governance Loop (Sprint 1)
- `GET /api/instances/{id}/workorder` — structured work order for suspended instances:
  brief, reasoning trace, decision options, governance tier
- `POST /api/instances/{id}/decision` — inject human decision, resume workflow;
  body: `{decision, rationale, reviewer_id}`
- `POST /api/instances/{id}/evidence` — supply missing evidence to suspended retrieve
  step, resume from correct step; body: `{step_name, content, content_type}`
- `GET /api/instances/{id}/stream` — SSE action ledger stream; events:
  `step_started`, `step_completed`, `governance_decision`, `hitl_requested`,
  `hitl_resolved`, `workflow_completed`, `workflow_failed`
- `GET /instances/{id}/trace` — self-contained HTML trace page with three modes:
  Watch (live SSE), Input (HITL form), Result (complete audit trace + export)
- `cognitive_core/api/server.py` — general-purpose FastAPI server; configurable
  via `CC_COORD_CONFIG` / `CC_COORD_BASE` env vars; works with any domain pack
- `GET /api/instances/{id}/verify` — ledger hash chain verification endpoint

### Added — Hardening (Sprint 2)
- `cognitive_core/engine/input_validation.py` — Pydantic case input validation at
  `coordinator.start()` boundary; structured errors with field name, reason,
  expected/received; size limits (512 KB total, 64 KB/field, 32 K chars/string)
- Input validation wired into `coordinator.start()` before any LLM work begins
- `tests/smoke/` — six-path smoke test suite, 23 tests, <60s, no LLM required:
  happy path, HITL path, evidence path, invalid input (4 cases), retry path (4 cases),
  multi-workflow delegation (4 cases), hash chain verification (3 cases)

### Added — Documentation (Sprint 3)
- `examples/integration_example.py` — full case lifecycle in both direct coordinator
  and HTTP API modes; HITL round-trip shown end-to-end in code
- `examples/mcp_template.py` — 60-line MCP server template with inline comments;
  stub-to-real pattern for connecting retrieve primitive to live data sources

### Added — Honesty Layer (Sprint 5)
- `OPERATIONAL_NOTES.md` — explicit statement of what is production-ready,
  experimental, not yet implemented, assumed, and known-broken

### Existing — Internal Credibility (Sprint 4, already present)
- SHA-256 hash chain on action ledger: `entry_hash = sha256(prior_hash + content)`
- `store.verify_ledger(instance_id)` → `{valid, first_invalid_entry, entry_count}`
- Genesis constant: fixed string as chain root for every instance
- Tamper detection: modification of any record causes verify to fail at that entry
- `cognitive_core/coordinator/ledger_chain.py` — standalone verify utility module

### Changed
- `QUICKSTART.md` rebuilt around HTML trace page as primary artifact; server-first
  flow with trace URL; HITL interaction shown with real output; smoke test run shown

---

## [0.1.0] — 2026-03-xx — Initial open-source release

### Added
- Eight typed cognitive primitives: `classify`, `retrieve`, `investigate`,
  `challenge`, `verify`, `deliberate`, `generate`, `govern`
- Three-layer configuration architecture: workflow YAML + domain YAML + case JSON
- Coordinator state machine with governance tiers (auto, spot_check, gate, hold)
- Delegation policies: fire-and-forget and wait-for-result between workflows
- Demand-driven delegation via `resource_requests` in `deliberate` output
- 14-module governance pipeline: PII redaction, guardrails, kill switch, shadow mode,
  eval gate, spec lock, cost tracking, semantic cache, compensation ledger,
  HITL routing, rate limiting, webhooks, replay, analytics artifacts
- Analytics artifact layer: causal DAGs, SDA policy models, constraint checkers
- FastAPI REST server with coordinator API
- Fraud operations demo with 9 cases and 4 workflows
- 38-assertion mechanism test suite (no LLM required)

### Architecture decisions
- `think` renamed to `deliberate` — aligns with epistemic function: meta-cognitive
  synthesis toward a warranted conclusion, not freeform thinking
- `act` replaced by `govern` — execution is not an epistemic operation; the primitive
  layer is now purely epistemic. `ActOutput` retained in `primitives/artifacts.py` as
  `ExecutionArtifact` for downstream systems
- `GovernanceWrapper` dissolved — governance is a first-class primitive, not a
  wrapper applied to cognitive outputs
- Coordinator is now provably non-cognitive: schedules and routes, does not decide

### Experimental (functional, unstable API)
- `coordinator/optimizer.py` — dispatch optimization (assignment, VRP archetypes)
- `coordinator/federation.py` — multi-coordinator federation
- `coordinator/hardening.py` — DDR audit trail, partial failure handling
- `coordinator/resilience.py` — revalidation guards, staleness detection

### Known gaps
- Service Bus task queue adapter — SQLite is current backend
- Eval gate not enforced at coordinator startup
- Single-worker deployment only
- No OpenTelemetry export
