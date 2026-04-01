# Quickstart

Run a governed institutional AI workflow and watch it execute in real time.

---

## Prerequisites

- Python 3.11+
- An API key from: **Anthropic**, OpenAI, or Google

---

## 1. Clone and install

```bash
git clone https://github.com/dioufseck-rgb/cognitive-core.git
cd cognitive-core
pip install -e ".[runtime,api]"
```

> **Minimal install** — only your provider:
> ```bash
> pip install -e ".[api]" langchain-anthropic langgraph   # Anthropic
> pip install -e ".[api]" langchain-openai langgraph      # OpenAI
> pip install -e ".[api]" langchain-google-genai langgraph # Google
> ```

---

## 2. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here   # Claude
# export OPENAI_API_KEY=your_key_here    # GPT-4o
# export GOOGLE_API_KEY=your_key_here    # Gemini
```

The framework auto-detects which provider to use from the key you set.

---

## 3. Start the server

```bash
cd library/domain-packs/consumer-lending

CC_COORD_CONFIG=coordinator_config.yaml \
CC_COORD_BASE=. \
uvicorn cognitive_core.api.server:app --port 8000
```

You should see:

```
INFO:     Started server process [12345]
[startup] Coordinator ready (config: coordinator_config.yaml)
[startup] Action ledger hash chain enabled
[startup] Thread pool ready (4 workers)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 4. Submit a case

In a second terminal, from the repo root:

```python
# submit_case.py
import urllib.request, json

case = {
    "applicant_name": "Diane Whitfield", "applicant_age": 42,
    "loan_amount": 8500, "loan_purpose": "Medical expenses",
    "investigation_findings": "",
    "get_credit":     {"score": 614, "utilisation_pct": 68, "derogatory_marks_24mo": 3,
                       "oldest_account_years": 7, "payment_history": "2 lates in 18 months"},
    "get_financials": {"annual_income_verified": 42000, "dti_ratio": 0.48,
                       "monthly_obligations": 1680, "requested_monthly_payment": 320},
    "get_employment": {"status": "part_time", "employer": "Various", "tenure_years": 0.8,
                       "income_source": "hourly", "verification_status": "unverified"},
    "get_banking":    {"avg_monthly_balance": 1200, "nsf_events_12mo": 0, "account_age_years": 12},
    "get_identity":   {"verification_status": "verified", "fraud_flag": False},
}

payload = json.dumps({
    "workflow_type": "loan_application_review",
    "domain": "consumer_lending",
    "case_input": case,
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/api/start",
    data=payload,
    headers={"Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print(f"Instance: {resp['instance_id']}")
print(f"Trace:    http://localhost:8000{resp['trace_url']}")
```

```
$ python submit_case.py
Instance: wf_a3f2c1b8
Trace:    http://localhost:8000/instances/wf_a3f2c1b8/trace
```

---

## 5. Watch the workflow execute

Open the trace URL in your browser. Steps appear as they complete:

```
Case: APP-003 — Diane Whitfield, $8,500 loan
────────────────────────────────────────────────────────────────────
▶ gather_application        retrieve      312ms
    Sources: get_credit, get_financials, get_employment, get_banking, get_identity

▶ classify_risk             classify      880ms
    ████████████░░░░░░░░  high_risk   confidence 0.81
    Score 614, DTI 48%, part-time employment. Multiple derogatory marks 18mo.

▶ investigate_risk_factors  investigate   2.1s
    Primary driver: DTI at 48% with unverified income.
    Derogatory marks are medical — not behavioral. Employment gap 4 months.
    Mitigating: long banking relationship, no NSF events, prior good standing.

▶ deliberate_recommendation deliberate    1.8s
    Recommendation: approve_modified
    Warrant: Approve at $6,000 with income verification condition.
    Medical derogatory marks non-predictive. DTI at $6K is 38%.

▶ verify_compliance         verify        640ms
    ✓ ECOA conforms
    ✓ FCRA conforms
    ✗ Ability-to-repay: income unverified — condition required
    ✓ Amount limits conforms

▶ govern_decision           govern        490ms
    GATE ⏸
    high_risk + approve_modified + unverified income
```

---

## 6. Approve or deny at the governance gate

When the workflow hits a GATE, the page switches to Input mode:

```
┌───────────────────────────────────────────────────────────────────┐
│  ⏸  UNDERWRITER REVIEW REQUIRED                        GATE       │
│                                                                   │
│  Recommend: approve_modified at $6,000                            │
│  Warrant: Medical derogatory marks non-predictive.                │
│           DTI improves to 38% at modified amount.                 │
│                                                                   │
│  Compliance: income unverified — condition required               │
│                                                                   │
│  Reviewer ID: [ reviewer              ]                           │
│  Rationale:   [ __________________________ ]                      │
│                                                                   │
│  [ Approve ]  [ Approve Modified ]  [ Deny ]  [ Refer ]           │
└───────────────────────────────────────────────────────────────────┘
```

Enter a rationale and click a decision. The workflow resumes immediately.

---

## 7. Download the audit trace

After completion, the page switches to Result mode with a **Download Trace** button.
The downloaded file is a self-contained HTML audit trail that renders offline.

A copy of the reference trace is included in the repo:
[`examples/loan-trace-example.html`](examples/loan-trace-example.html)

---

## 8. Verify ledger integrity

```bash
curl http://localhost:8000/api/instances/wf_a3f2c1b8/verify
```

```json
{
  "valid": true,
  "first_invalid_entry": null,
  "entries_checked": 14,
  "instance_id": "wf_a3f2c1b8"
}
```

Every ledger entry includes `entry_hash = sha256(prior_hash + content)`.
Modification of any record causes verification to fail at that entry.

---

## 9. Run the smoke tests

```bash
pytest tests/smoke/ -v
```

All 23 tests pass in under 60 seconds. No LLM calls required — mock responses only.

---

## CLI demo (no server required)

```bash
python demos/loan-application-review/run.py
```

Runs all four applications through the full chain and prints structured output.

---

## Architecture in one paragraph

Eight typed epistemic primitives compose into workflows via YAML. A domain YAML injects expertise into those primitives at runtime. A coordinator manages workflow lifecycle, governance tiers, and multi-workflow delegation. Every step produces a typed output. Every governance decision produces an accountability chain recorded in a tamper-evident SHA-256 hash chain ledger. The audit trail is the computation — there is nothing to reconstruct after the fact.

Full design: [docs/institutional-intelligence.md](docs/institutional-intelligence.md)
Maturity and assumptions: [OPERATIONAL_NOTES.md](OPERATIONAL_NOTES.md)
