# Cognitive Core — REST API Reference

> **Version:** v0.1.0-technical-preview  
> **Base URL:** `http://localhost:8000` (default)  
> **Auth:** None in v0.1. Deploy behind your own API gateway with authentication.

All request and response bodies are JSON (`Content-Type: application/json`).  
All timestamps are Unix epoch floats.  
All errors return `{"detail": "message"}` with an appropriate 4xx/5xx status.

---

## Authentication

Version 0.1 has no built-in authentication or RBAC. See `OPERATIONAL_NOTES.md`.

---

## Case submission

### POST /api/start

Submit a new case for processing. Returns immediately with an instance ID.
The workflow runs asynchronously in the thread pool.

**Request**

```json
{
  "workflow_type": "loan_application_review",
  "domain": "consumer_lending",
  "case_input": {
    "applicant_name": "Diane Whitfield",
    "loan_amount": 8500,
    "get_credit": { "score": 614, "..." : "..." }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `workflow_type` | string | yes | Name of the workflow YAML (without `.yaml`) |
| `domain` | string | yes | Name of the domain YAML (without `.yaml`) |
| `case_input` | object | yes | Case data. Must be a non-empty dict, max 512 KB |

**Response 200**

```json
{
  "instance_id": "wf_a3f2c1b8",
  "workflow_type": "loan_application_review",
  "domain": "consumer_lending",
  "status": "running",
  "trace_url": "/instances/wf_a3f2c1b8/trace",
  "stream_url": "/api/instances/wf_a3f2c1b8/stream"
}
```

**Errors**

| Status | Condition |
|---|---|
| 422 | `workflow_type`, `domain`, or `case_input` missing or empty |
| 422 | `case_input` fails schema validation (see `engine/input_validation.py`) |
| 500 | Coordinator could not start the workflow |

---

## Instance state

### GET /api/instances

List workflow instances.

**Query params**

| Param | Default | Description |
|---|---|---|
| `limit` | 50 | Max instances to return |
| `status` | — | Filter by status: `running`, `suspended`, `completed`, `failed` |

**Response 200** — array of instance summaries:

```json
[
  {
    "instance_id": "wf_a3f2c1b8",
    "workflow_type": "loan_application_review",
    "domain": "consumer_lending",
    "status": "suspended",
    "governance_tier": "gate",
    "created_at": 1743400000.0,
    "updated_at": 1743400287.4,
    "elapsed_seconds": 287.4,
    "step_count": 5,
    "current_step": "__governance_gate__",
    "case_meta": { "applicant_name": "Diane Whitfield", "loan_amount": 8500 },
    "error": null
  }
]
```

---

### GET /api/instances/{instance_id}

Get full instance state including result and action ledger.

**Response 200**

```json
{
  "instance_id": "wf_a3f2c1b8",
  "workflow_type": "loan_application_review",
  "domain": "consumer_lending",
  "status": "completed",
  "governance_tier": "gate",
  "correlation_id": "corr_abc123",
  "created_at": 1743400000.0,
  "updated_at": 1743400620.1,
  "elapsed_seconds": 620.1,
  "step_count": 7,
  "current_step": "generate_output",
  "result": { "disposition": "approve_modified", "..." : "..." },
  "error": null,
  "case_meta": {},
  "ledger": [
    {
      "id": 1,
      "instance_id": "wf_a3f2c1b8",
      "correlation_id": "corr_abc123",
      "action_type": "workflow_started",
      "details": {},
      "idempotency_key": null,
      "created_at": 1743400000.0
    }
  ]
}
```

**Errors**

| Status | Condition |
|---|---|
| 404 | Instance not found |

---

### GET /api/instances/{instance_id}/chain

Return the full delegation chain for a case, ordered oldest-first. Each entry
is an enriched instance with work orders, pending tasks, and case input.

**Response 200** — array of enriched instances. See `/api/instances` for field shapes.

---

### POST /api/instances/{instance_id}/terminate

Terminate a workflow instance immediately.

**Request**

```json
{ "reason": "Cancelled by analyst" }
```

**Response 200**

```json
{ "instance_id": "wf_a3f2c1b8", "status": "terminated" }
```

---

### POST /api/instances/{instance_id}/resume

Resume a suspended instance with optional external input.

**Request**

```json
{
  "external_input": { "any": "data" },
  "resume_nonce": ""
}
```

**Response 200** — fires and forgets; poll `GET /api/instances/{id}` for status:

```json
{ "instance_id": "wf_a3f2c1b8", "status": "processing" }
```

---

## HITL governance (Sprint 1.1)

### GET /api/instances/{instance_id}/workorder

Return the structured work order for a suspended instance. Use this to render
the reviewer UI — it includes brief, reasoning trace, and decision options.

**Response 200**

```json
{
  "instance_id": "wf_a3f2c1b8",
  "task_id": "task_xyz789",
  "workflow_type": "loan_application_review",
  "domain": "consumer_lending",
  "governance_tier": "gate",
  "status": "suspended",
  "suspended_at_step": "__governance_gate__",
  "created_at": 1743400000.0,
  "updated_at": 1743400287.4,
  "brief": "Applicant Diane Whitfield, $8,500 loan request. High-risk classification with unverified income. Recommend approve_modified at $6,000 with income verification condition.",
  "escalation_reason": "high_risk + approve_modified + unverified income",
  "escalation_brief": {
    "summary": "...",
    "key_facts": []
  },
  "reasoning_trace": [
    {
      "action": "step_completed",
      "step": "classify_risk",
      "primitive": "classify",
      "output": { "category": "high_risk", "confidence": 0.81, "reasoning": "..." },
      "elapsed_ms": 880
    }
  ],
  "decision_options": ["approve", "approve_modified", "deny", "refer"],
  "result_summary": { "disposition": "approve_modified" },
  "case_meta": {},
  "resume_nonce": "nonce_abc123"
}
```

**Errors**

| Status | Condition |
|---|---|
| 400 | Instance is not suspended |
| 404 | Instance or suspension record not found |

---

### POST /api/instances/{instance_id}/decision

Inject a human decision into a suspended instance and resume the workflow.
Returns immediately; workflow continues asynchronously.

**Request**

```json
{
  "decision": "approve_modified",
  "rationale": "Medical derogatory marks non-predictive. Approve at $6K with income verification.",
  "reviewer_id": "jsmith"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `decision` | string | yes | One of the `decision_options` from the work order |
| `rationale` | string | no | Free-text rationale, logged to audit ledger |
| `reviewer_id` | string | yes | Reviewer identifier, logged to audit ledger |

**Decision routing**

| Decision | Coordinator method |
|---|---|
| `approve`, `approve_modified`, `release`, `refer` | `coordinator.approve()` |
| `deny`, `reject`, `escalate` | `coordinator.reject()` |

**Response 200**

```json
{
  "instance_id": "wf_a3f2c1b8",
  "decision": "approve_modified",
  "reviewer_id": "jsmith",
  "status": "processing"
}
```

**Errors**

| Status | Condition |
|---|---|
| 400 | Instance is not suspended |
| 404 | Instance not found |
| 422 | `decision` or `reviewer_id` is empty |

---

### POST /api/instances/{instance_id}/evidence

Supply missing evidence to a suspended retrieve step and resume the workflow.

**Request**

```json
{
  "step_name": "gather_application",
  "content": { "get_credit": { "score": 750, "utilisation_pct": 22 } },
  "content_type": "json"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `step_name` | string | yes | Name of the retrieve step that requested evidence |
| `content` | any | yes | The evidence payload (any JSON-serialisable value) |
| `content_type` | string | no | `text`, `json`, or `document` (default: `text`) |

**Response 200**

```json
{
  "instance_id": "wf_a3f2c1b8",
  "step_name": "gather_application",
  "content_type": "json",
  "status": "processing"
}
```

**Errors**

| Status | Condition |
|---|---|
| 400 | Instance is not suspended |
| 404 | Instance not found |
| 422 | `step_name` empty or `content` null |

---

## SSE action ledger stream (Sprint 1.2)

### GET /api/instances/{instance_id}/stream

Server-Sent Events stream of the action ledger for a workflow instance.

Opens immediately and stays open until the workflow completes or fails.
For completed instances, replays from the ledger.

**Query params**

| Param | Default | Description |
|---|---|---|
| `since_id` | 0 | Only return ledger entries with `id > since_id` |

**Response** — `Content-Type: text/event-stream`

Each SSE message has an `event:` line and a `data:` JSON payload:

```
event: step_completed
data: {"event":"step_completed","instance_id":"wf_a3f2c1b8","ledger_id":4,"step_name":"classify_risk","primitive":"classify","output":{"category":"high_risk","confidence":0.81},"elapsed_ms":880,"timestamp":1743400145.3}

event: hitl_requested
data: {"event":"hitl_requested","instance_id":"wf_a3f2c1b8","governance_tier":"gate","reason":"high_risk + approve_modified + unverified income","timestamp":1743400287.4}

event: workflow_completed
data: {"event":"workflow_completed","instance_id":"wf_a3f2c1b8","status":"completed","elapsed_seconds":620.1,"step_count":7,"result":{"disposition":"approve_modified"},"timestamp":1743400620.1}
```

**Event types**

| Event | Fired when |
|---|---|
| `step_started` | A workflow step begins execution |
| `step_completed` | A step finishes successfully |
| `governance_decision` | `govern` primitive evaluates tier |
| `hitl_requested` | Workflow suspends at a governance gate |
| `hitl_resolved` | Human decision submitted, workflow resuming |
| `workflow_completed` | Workflow finishes in `completed` state |
| `workflow_failed` | Workflow finishes in `failed` or `terminated` state |

**Common event payload fields**

| Field | Type | Description |
|---|---|---|
| `event` | string | Event type (one of the above) |
| `instance_id` | string | Instance this event belongs to |
| `ledger_id` | int | Action ledger row id (use for `since_id` resumption) |
| `step_name` | string | Workflow step name |
| `primitive` | string | Primitive type (`classify`, `retrieve`, etc.) |
| `output` | object | Step output (primitive-specific) |
| `elapsed_ms` | int | Step execution time in milliseconds |
| `timestamp` | float | Unix epoch timestamp |

**JavaScript example**

```javascript
const es = new EventSource(`/api/instances/${instanceId}/stream`);

es.addEventListener('step_completed', e => {
  const data = JSON.parse(e.data);
  console.log(`${data.step_name} (${data.primitive}) — ${data.elapsed_ms}ms`);
});

es.addEventListener('hitl_requested', e => {
  const data = JSON.parse(e.data);
  // Fetch /api/instances/{id}/workorder and render the review form
  showReviewForm(data.instance_id);
  es.close();
});

es.addEventListener('workflow_completed', e => {
  const data = JSON.parse(e.data);
  console.log(`Done: ${data.result?.disposition}`);
  es.close();
});
```

---

## Ledger integrity (Sprint 4.1)

### GET /api/instances/{instance_id}/verify

Verify the SHA-256 hash chain integrity of an instance's action ledger.

Each ledger entry stores `entry_hash = sha256(prior_hash + canonical_content)`.
The genesis entry hashes a fixed constant. Modification of any record causes
verification to fail at that entry.

**Response 200**

```json
{
  "valid": true,
  "first_invalid_entry": null,
  "entries_checked": 14,
  "instance_id": "wf_a3f2c1b8"
}
```

If tampered:

```json
{
  "valid": false,
  "first_invalid_entry": 7,
  "entries_checked": 7,
  "instance_id": "wf_a3f2c1b8"
}
```

| Field | Type | Description |
|---|---|---|
| `valid` | bool | `true` if chain is intact |
| `first_invalid_entry` | int \| null | Ledger row id of first invalid entry, or null |
| `entries_checked` | int | Number of entries verified |
| `instance_id` | string | Instance verified |

---

## Task queue

### GET /api/tasks/pending

List all pending governance approval tasks.

**Response 200** — array of tasks:

```json
[
  {
    "task_id": "task_xyz789",
    "instance_id": "wf_a3f2c1b8",
    "workflow_type": "loan_application_review",
    "domain": "consumer_lending",
    "governance_tier": "gate",
    "correlation_id": "corr_abc123",
    "queue": "underwriter_review",
    "created_at": 1743400000.0,
    "sla_seconds": 86400,
    "expires_at": 1743486400.0,
    "priority": 0,
    "callback_method": "approve",
    "resume_nonce": "nonce_abc123"
  }
]
```

---

### POST /api/tasks/{task_id}/approve

Approve a governance gate task.

**Request**

```json
{ "approver": "jsmith", "notes": "Approve as submitted." }
```

**Response 200** — fires and forgets:

```json
{ "instance_id": "wf_a3f2c1b8", "status": "processing" }
```

---

### POST /api/tasks/{task_id}/reject

Reject a governance gate task. Terminates the workflow.

**Request**

```json
{ "rejector": "jsmith", "reason": "Outside policy." }
```

**Response 200**

```json
{ "instance_id": "wf_a3f2c1b8", "status": "processing" }
```

---

## HTML trace page

### GET /instances/{instance_id}/trace

Serves a self-contained HTML page for the workflow instance. No `Content-Type: application/json`.

Three modes, selected automatically:

| Mode | Condition | Shows |
|---|---|---|
| **Watch** | Instance is running | Live step stream via SSE |
| **Input** | Instance is suspended | HITL review form with work order |
| **Result** | Instance is completed or failed | Full audit trace + ledger verification + export |

The page connects to the SSE stream automatically and transitions between modes
without page reload. The **Download Trace** button exports a self-contained
offline HTML file with the full audit trail embedded.

---

## Rate limiting

v0.1 has no rate limiting. The thread pool (default: 4 workers) is the only
concurrency control. Configure `CC_WORKERS` to adjust pool size.

---

## Error response shape

All endpoints return errors in FastAPI's standard shape:

```json
{ "detail": "Instance not found: wf_a3f2c1b8" }
```

Validation errors from Pydantic return a structured list:

```json
{
  "detail": [
    {
      "loc": ["body", "workflow_type"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
