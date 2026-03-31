# Content Moderation

**Pattern:** P04 — Adversarial Review  
**Overlays:** governance  
**Coordinator:** simple  

Classifies content against policy categories, verifies against explicit policy rules, adversarially challenges borderline decisions from multiple perspectives, and governs disposition.

## What it does

Every case runs through four steps:

1. **Classify** — assigns a category (safe, borderline, harassment, misinformation, explicit, spam)
2. **Verify** — checks against explicit policy rules
3. **Challenge** — adversarially tests the classification when confidence is below threshold or policy fails (skipped for high-confidence conforming content)
4. **Govern** — determines tier (AUTO / SPOT_CHECK / GATE / HOLD) and disposition

Four governance outcomes are possible:
- `AUTO proceed` — safe, high-confidence content publishes immediately
- `SPOT_CHECK proceed` — publishes, sampled for quality review
- `GATE suspend` — held for human reviewer with a structured work order
- `HOLD` — compliance hold for legal or policy exposure

## Files

```
content-moderation/
  content_moderation.yaml              ← fill in your categories, rules, and thresholds
  workflow.yaml            ← the workflow (do not edit)
  coordinator_config.yaml  ← governance queues and SLAs (edit queue names)
  run.py                   ← runner script
  cases/
    example_posts.json     ← 4 example cases showing expected paths
```

## Setup

From the repository root:

```bash
# Install dependencies
pip install -e ".[dev]"

# Set your LLM API key
export ANTHROPIC_API_KEY=your_key_here

# Run the example cases
python -m library.domain-packs.content-moderation.run
```

## Customising for your platform

Open `content_moderation.yaml` and fill in:

| Section | What to change |
|---------|---------------|
| `classify_content.categories` | Your content category taxonomy |
| `classify_content.criteria` | How the classifier should weigh context, author history, platform type |
| `verify_policy.rules` | Your specific policy rules, numbered |
| `challenge_decision.perspective` | Who is challenging (trust & safety, legal, free speech, etc.) |
| `challenge_decision.threat_model` | What failure modes to probe for |
| `govern_disposition.governance_context` | When each tier fires on your platform |

In `coordinator_config.yaml`, update the queue names to match your infrastructure:
- `qa_review` → your quality sampling queue
- `human_review` → your content reviewer queue  
- `compliance_review` → your legal/compliance hold queue

## Scaffold boundary

The scaffold covers the majority of platform variation — category definitions, policy rules, adversarial perspectives, and governance thresholds.

If you need structural changes — a second challenge step from an independent team, a retrieve step pulling author history from an external system, a different challenge trigger threshold — those require changes to the workflow YAML. See `library/patterns/P04-adversarial-review/` for the base pattern.

## Governance defaults

| Tier | Default condition |
|------|-------------------|
| AUTO | Safe, confidence ≥ 0.85; or clear violation, confidence ≥ 0.80 |
| SPOT_CHECK | Safe, confidence 0.65–0.84 |
| GATE | Borderline; any confidence < 0.65; challenge found vulnerabilities |
| HOLD | Named public figure; legal exposure; open escalation on author |

Adjust thresholds in `govern_disposition.governance_context` to match your risk tolerance.
