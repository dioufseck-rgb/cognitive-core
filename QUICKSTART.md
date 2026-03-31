# Quickstart

Run a governed institutional AI workflow in five minutes.

---

## Prerequisites

- Python 3.10+
- an LLM API key


---

## 1. Clone and install

```bash
git clone https://github.com/dioufseck-rgb/cognitive-core.git
cd cognitive-core
pip install -e ".[dev]"
```

## 2. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## 3. Run a demo

Pick one:

**Content moderation** — classifies posts, tests borderline decisions adversarially, governs disposition across four tiers (auto / spot-check / gate / hold):

```bash
python demos/content-moderation/run.py
```

**Loan application review** — six primitives, four applicants, four governance outcomes:

```bash
python demos/loan-application-review/run.py
```

**Fraud operations** — multi-workflow investigation chain with parallel specialist delegation:

```bash
python demos/fraud-operations/run.py
```

---

## What you'll see

Each demo shows the same structure: a workflow running through typed epistemic steps, each producing a structured output, the govern primitive determining the institutional disposition.

Example output from the content moderation demo:

```
─────────────────────────────────────────────────────────────────
  POST-002  |  social_post
  "These people flooding into our country are not like us..."
─────────────────────────────────────────────────────────────────
  Classification:  borderline  (confidence 0.85)
  Policy:          ✗ 1 violation(s)
  Challenge:       fails  (2 vulnerabilities)
  Governance:      ⏸  HUMAN REVIEW  —  suspend_for_approval
  Work order:      → human_review queue
```

The govern primitive determined that a borderline classification with a failed challenge requires human review before any action. That decision is structural — not a prompt instruction, not a post-hoc filter.

---

## Next: use a domain pack

The domain library contains five ready-to-run packs for institutional decision domains. Each pack ships with example cases and a domain scaffold you fill in with your own knowledge.

```bash
# Run the content moderation pack with your own posts
python library/domain-packs/content-moderation/run.py

# Run with a specific case
python library/domain-packs/consumer-lending/run.py --case-id APP-002
```

See [library/README.md](library/README.md) to pick the right pack for your domain.

---

## Next: build your own workflow

Two files define a complete governed workflow:

**`workflow.yaml`** — the epistemic sequence:

```yaml
name: my_review
steps:
  - name: classify_item
    primitive: classify
    params:
      categories: "${domain.classify_item.categories}"
      criteria: "${domain.classify_item.criteria}"

  - name: govern_outcome
    primitive: govern
    params:
      workflow_state: |
        classification: ${classify_item.category}
        confidence: ${classify_item.confidence}
      governance_context: "${domain.govern_outcome.governance_context}"
```

**`domain.yaml`** — the domain expertise:

```yaml
domain_name: my_review
workflow: my_review
governance: gate

classify_item:
  categories: |
    - approved: ...
    - declined: ...
    - review: ...
  criteria: |
    ...

govern_outcome:
  governance_context: |
    AUTO when: classification is approved, confidence >= 0.85
    GATE when: classification is review or confidence < 0.85
    HOLD when: ...
```

Run it with a minimal coordinator config pointing at your files. The primitive layer, execution engine, governance pipeline, and HITL state machine are infrastructure — you configure, not code.

See [library/patterns/](library/patterns/) for production-ready patterns covering the full range of institutional decision workflows.

---

## Architecture in one paragraph

Eight typed epistemic primitives compose into workflows. A domain YAML injects expertise into those primitives at runtime. A coordinator manages workflow lifecycle, governance tiers, and multi-workflow delegation. Every step produces a typed output. Every governance decision produces an accountability chain. The audit trail is the computation — there is nothing to reconstruct after the fact.

Full architecture: [docs/institutional-intelligence.md](docs/institutional-intelligence.md)
