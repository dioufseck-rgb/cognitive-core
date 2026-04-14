# Quickstart

Run a governed institutional AI workflow in five minutes.

---

## Prerequisites

- Python 3.11+
- An API key: **Anthropic**, Google, or OpenAI

---

## 1. Clone and install

```bash
git clone https://github.com/dioufseck-rgb/cognitive-core.git
cd cognitive-core
pip install -e .
```

---

## 2. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key   # Claude
# export GOOGLE_API_KEY=your_key    # Gemini
# export OPENAI_API_KEY=your_key    # OpenAI
```

The framework auto-detects the provider from the key present in the environment.

---

## 3. Start the server

Point the server at any domain pack:

```bash
CC_COORD_CONFIG=library/domain-packs/consumer-lending/coordinator_config.yaml \
CC_COORD_BASE=library/domain-packs/consumer-lending \
uvicorn cognitive_core.api.server:app --port 8000
```

You should see:

```
[startup] Coordinator ready (config: coordinator_config.yaml)
[startup] Action ledger hash chain enabled
[startup] Thread pool ready (4 workers)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open `http://localhost:8000` — the landing page shows all instances and auto-refreshes every 5 seconds.

---

## 4. Submit a case

In a second terminal:

```python
# submit_case.py
import urllib.request, json

case = {
    "applicant_name": "Diane Whitfield", "applicant_age": 42,
    "loan_amount": 8500, "loan_purpose": "Medical expenses",
    "get_credit":     {"score": 614, "utilisation_pct": 68,
                       "derogatory_marks_24mo": 3, "oldest_account_years": 7,
                       "payment_history": "2 lates in 18 months"},
    "get_financials": {"annual_income_verified": 42000, "dti_ratio": 0.48,
                       "monthly_obligations": 1680, "requested_monthly_payment": 320},
    "get_employment": {"status": "part_time", "employer": "Various",
                       "tenure_years": 0.8, "income_source": "hourly",
                       "verification_status": "unverified"},
    "get_banking":    {"avg_monthly_balance": 1200, "nsf_events_12mo": 0,
                       "account_age_years": 12},
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
Instance: wf_a3f2c1b8
Trace:    http://localhost:8000/instances/wf_a3f2c1b8/trace
```

---

## 5. Watch it execute

Open the trace URL. Steps appear as they complete, with live epistemic state:

```
▶ gather_application        retrieve      312ms
▶ classify_risk             classify      880ms   high_risk (0.81)
    epistemic: SUPPORTED · overall=0.44 · warranted=✓ · rq=0.90 · oc=0.82 · sep=0.51
▶ investigate_risk_factors  investigate   2.1s
    epistemic: SUPPORTED · overall=0.43 · warranted=✓ · rq=0.92 · oc=0.85
▶ deliberate_recommendation deliberate    1.8s    approve_modified
    epistemic: SUPPORTED · overall=0.48 · warranted=✓ · rq=0.95 · oc=0.88
▶ verify_compliance         verify        640ms   conforms=False (1 violation)
▶ govern_decision           govern        490ms
    ⏸  GATE — high_risk + approve_modified + unverified income
```

---

## 6. Approve or deny at the governance gate

When the workflow hits a GATE the page switches to Input mode. Enter a reviewer ID, optional rationale, and click a decision. The workflow resumes immediately.

---

## 7. Verify ledger integrity

```bash
curl http://localhost:8000/api/instances/wf_a3f2c1b8/verify
```

```json
{"valid": true, "first_invalid_entry": null, "entries_checked": 14}
```

Every ledger entry is `sha256(prior_hash + content)`. Modification of any record is detectable.

---

## Run a domain pack directly (no server)

Each domain pack includes a `run.py`:

```bash
python library/domain-packs/consumer-lending/run.py
python library/domain-packs/content-moderation/run.py
python library/domain-packs/clinical-triage/run.py
```

---

## Run the agentic demonstration

Two hardship cases, no declared sequence, autonomous trajectory differentiation:

```bash
python demos/loan-hardship-agentic/run.py
python demos/loan-hardship-agentic/run.py --case reeves
python demos/loan-hardship-agentic/run.py --case webb
```

See [demos/loan-hardship-agentic/README.md](demos/loan-hardship-agentic/README.md) for full instructions and what to look for.

---

## Run the smoke tests

```bash
pytest tests/smoke/ tests/test_devs_kernel.py
# 50 tests, ~2 minutes, no LLM calls required
```

---

## Architecture in one paragraph

Eight typed epistemic primitives compose into workflows via YAML. A domain YAML injects expertise into those primitives at runtime. Two execution modes: workflow (declared sequence) and agentic (orchestrator reasons the path from goal and evidence). A coordinator manages workflow lifecycle, governance tiers, and human-in-the-loop suspension. Every step produces a three-layer epistemic state. Every governance decision is recorded in a tamper-evident SHA-256 hash chain ledger. The audit trail is endogenous to the computation.

Full design: [docs/institutional-intelligence.md](docs/institutional-intelligence.md)  
Maturity and assumptions: [OPERATIONAL_NOTES.md](OPERATIONAL_NOTES.md)
