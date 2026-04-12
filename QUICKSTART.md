# Quickstart

Run a governed institutional AI workflow in five minutes.

---

## Prerequisites

- Python 3.11+
- An API key: Google Gemini, Anthropic, or OpenAI

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
export GOOGLE_API_KEY=your_key      # Gemini (primary development provider)
# export ANTHROPIC_API_KEY=your_key # Claude
# export OPENAI_API_KEY=your_key    # OpenAI
```

The framework auto-detects the provider from the key present in the environment.

---

## 3. Run a prior authorization appeal case

```bash
cd demos/prior-auth-appeal
python run.py --case pa_2024_a001
```

You will see steps completing in real time with epistemic state at each step:

```
▶ retrieve_clinical_record       retrieve      1.2s   confidence=1.00
▶ retrieve_plan_criteria         retrieve      0.9s   confidence=1.00
▶ retrieve_regulatory_framework  retrieve      1.1s   confidence=1.00
▶ retrieve_clinical_evidence     retrieve      0.8s   confidence=1.00
▶ classify_clinical_presentation classify      1.4s   myelopathy_primary (1.00)
    epistemic: SUPPORTED · warranted=✓ · rq=0.95 · oc=0.92
▶ investigate_plan_criteria_met  investigate   3.1s
    epistemic: SUPPORTED · warranted=✓ · rq=0.92 · oc=0.88
▶ investigate_regulatory         investigate   2.8s
    epistemic: SUPPORTED · warranted=✓ · rq=0.96 · oc=0.94
▶ investigate_clinical_standard  investigate   2.6s
    epistemic: SUPPORTED · warranted=✓ · rq=0.94 · oc=0.90
▶ verify_cross_source            verify        1.3s   conforms=False
▶ deliberate_disposition         deliberate    2.1s   OVERTURN (1.00)
▶ generate_determination         generate      3.4s
▶ challenge_determination        challenge     2.9s   survives=True
▶ govern_appeal_outcome          govern        1.1s
    ✓  SPOT_CHECK — determination survives challenge, high confidence
```

The orchestrator reasoned this path autonomously. No step sequence was declared.

---

## 4. Run the full benchmark

Replicates the 11-case benchmark from the paper:

```bash
cd demos/prior-auth-appeal
python run_benchmark.py
```

This runs all 11 benchmark cases against CC. Results are saved to `output/benchmark/`.

To score and compare against the ReAct and Plan-and-Solve baselines:

```bash
python score_benchmark.py
python compare_benchmark.py
```

Ground truth for each case is in the `ground_truth_complexity` block of each case JSON file under `cases/`.

---

## 5. Run the loan modification demo

```bash
cd demos/loan-modification
python run.py
```

This demonstrates the configuration economics claim: same framework, different domain YAML, different output vocabulary and failure mode profile.

---

## 6. Verify ledger integrity

Every run produces a hash-chained audit ledger. To verify a completed instance:

```bash
curl http://localhost:8000/api/instances/wf_<id>/verify
```

```json
{"valid": true, "first_invalid_entry": null, "entries_checked": 13}
```

Or inspect the ledger directly from the SQLite database:

```python
import sqlite3, json
conn = sqlite3.connect('cognitive_core.db')
cur = conn.cursor()
cur.execute('SELECT action_type, details FROM action_ledger WHERE instance_id=? ORDER BY id', ('wf_<id>',))
for action_type, details in cur.fetchall():
    print(action_type, json.loads(details).get('primitive',''))
```

---

## 7. Run the tests

No LLM calls required:

```bash
pytest tests/unit/ tests/smoke/ tests/test_devs_kernel.py
# 203 tests
```

---

## Architecture in one paragraph

Nine typed epistemic primitives compose into workflows via YAML. A domain YAML injects expertise into those primitives at runtime. Two execution modes: workflow (declared sequence) and agentic (orchestrator reasons the path from goal and evidence, with hard constraints enforced by the substrate). A coordinator manages workflow lifecycle, governance tiers, and human-in-the-loop suspension. Every step produces a three-layer epistemic state (mechanical signals, judgment signals, coherence flags). Every governance decision is recorded in a tamper-evident SHA-256 hash chain ledger. The audit trail is endogenous to the computation.

Full design: [arXiv paper](https://arxiv.org/abs/PLACEHOLDER)
