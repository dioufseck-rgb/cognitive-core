# Quickstart

Run a governed agentic workflow on one of the two demonstrated domains.

---

## Prerequisites

- Python 3.11+
- An API key. Google Gemini is the primary tested provider; Anthropic, OpenAI, Azure and Bedrock are implemented but not yet tested end-to-end (see the main README).

---

## 1. Clone and install

```bash
git clone https://github.com/dioufseck-rgb/cognitive-core.git
cd cognitive-core
pip install -e .
```

## 2. Set your API key

```bash
export GOOGLE_API_KEY=your_key      # primary tested provider
# export ANTHROPIC_API_KEY=your_key
# export OPENAI_API_KEY=your_key
```

Provider and model are set in `llm_config.yaml`.

## 3. Run a domain from the command line

```bash
# Prior authorization appeal (the domain evaluated in the IEEE Computer paper)
python demos/prior-auth-appeal/run.py --case pa_2024_a001.json

# Same case, also run the ReAct baseline for comparison, and save determinations
python demos/prior-auth-appeal/run.py --case pa_2024_a001.json --compare --save

# Loan modification
python demos/loan-modification/run.py --case lm_2024_a001.json --compare
```

Both domains run in agentic mode. The orchestrator chooses the sequence from nine available primitives; the workflow YAML constrains it (required primitives, step limits, must end with `govern`). Each step prints its epistemic state, and the run ends with a governance tier.

## 4. Run the server and trace UI

```bash
CC_COORD_CONFIG=demos/prior-auth-appeal/coordinator_config.yaml \
CC_COORD_BASE=demos/prior-auth-appeal \
uvicorn cognitive_core.api.server:app --port 8000
```

Open `http://localhost:8000`. Submit a case with `POST /api/start`, then open the returned trace URL to follow execution. When a run reaches a `gate` or `hold` tier it suspends until a reviewer decision is posted.

## 5. Verify ledger integrity

```bash
curl http://localhost:8000/api/instances/<instance_id>/verify
```

Every ledger entry is `sha256(prior_hash + canonical_content)`, so modification of any record is detectable.

## 6. Run the tests

```bash
pytest tests/smoke/ tests/unit/ tests/test_devs_kernel.py
# no LLM calls required
```

## 7. Reproduce the published results

See [Reproducing the published results](README.md#reproducing-the-published-results) in the main README.

---

## Architecture in one paragraph

Nine typed epistemic primitives compose into workflows declared in YAML. A domain YAML supplies expertise and governance configuration at runtime. There are two execution modes: workflow mode, with a declared sequence, and agentic mode, where an orchestrator chooses the path from the goal and the evidence. A coordinator manages the workflow lifecycle, governance tiers, and human-in-the-loop suspension. Every step produces a three-layer epistemic state, and every governance decision is recorded in a tamper-evident SHA-256 hash-chain ledger.
