# Changelog

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
