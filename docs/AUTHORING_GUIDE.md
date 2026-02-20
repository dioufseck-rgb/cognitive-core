# Cognitive Core — Authoring Guide

How to add a new use case without touching the framework.

---

## Overview

Every use case is three files:

| Layer | File | Who Authors | Changes How Often |
|-------|------|-------------|-------------------|
| **Workflow** | `workflows/*.yaml` | AI Engineers | Rarely — reuse existing |
| **Domain** | `domains/*.yaml` | SME + AI Engineer | Per use case |
| **Case** | `cases/*.json` | Systems / APIs | Per execution |

The framework code (`engine/`, `registry/`) does not change.

---

## Step 1: Choose a Workflow

Before writing anything, check if an existing workflow fits your use case.

### Available Workflows

| Workflow | Pattern | Good For |
|----------|---------|----------|
| `dispute_resolution.yaml` | Retrieve → Classify → Verify → Investigate → Classify → Generate → Challenge | Any case with intake data, a decision to make, and a member-facing response |
| `loan_hardship.yaml` | Classify → Investigate (branching) → Classify → Generate → Challenge | Assessment with branching paths based on classification |
| `sar_investigation.yaml` | Retrieve → Classify → Investigate → Classify → Generate → Challenge → Verify | Investigative workflows ending in a compliance check |
| `spending_advisor.yaml` | Retrieve → Classify → Investigate → Generate → Challenge | Conversational Q&A over data |
| `nurse_triage.yaml` | Retrieve → Classify → Classify → Investigate → Verify → Generate → Challenge | Safety-critical with mandatory verification |
| `regulatory_impact.yaml` | Retrieve → Classify → Investigate → Classify → Generate → Challenge | Analysis with a second classification for action priority |

### Agentic Workflows

| Workflow | Good For |
|----------|----------|
| `spending_advisor_agentic.yaml` | Exploratory analysis where the path depends on what's discovered |
| `loan_hardship_agentic.yaml` | Complex assessment with optional verification and escalation paths |

**When to create a new workflow:** Only if the cognitive pattern is genuinely different — different primitives, different loop structure, different routing logic. If you just need different categories, rules, or criteria, that's a domain change, not a workflow change.

---

## Step 2: Author the Domain Config

The domain config is where subject matter expertise lives. It provides the content that gets injected into the workflow's primitive steps.

### Structure (Sequential Mode)

```yaml
domain_name: my_domain          # Snake case, unique
workflow: dispute_resolution    # Which workflow this pairs with
description: >
  One-line description of what this domain covers.

# One section per workflow step, named to match the step name.
# Each section provides the params that step needs.

classify_dispute_type:          # Must match step name in workflow
  categories: |                 # Multi-line string with category definitions
    - category_name: Description of when to use this category.
    - another_category: Description.
  criteria: |
    How to decide between categories.
    Reference case data: ${input.field_name}
  confidence_threshold: "0.8"

investigate_dispute:
  scope: |
    What to investigate and how deep.
  effort_level: deep

generate_response:
  requirements: |
    What the output should contain.
    TONE: How to write it.
  format: member_letter
  constraints: |
    - Rule 1
    - Rule 2

challenge_response:
  perspective: |
    Who is challenging and from what angle.
  threat_model: |
    CRITICAL: What absolutely must not happen.
    HIGH: What should be caught.
    MEDIUM: Nice to catch.
```

### Structure (Agentic Mode)

```yaml
domain_name: my_domain_agentic
workflow: my_workflow_agentic
description: >
  Description.

goal: |
  What the workflow should accomplish.
  What the final deliverable is.

orchestrator_strategy: |
  Guidance for the orchestrator on how to sequence primitives.

primitive_configs:
  config_name_1:
    primitive: classify
    params:
      categories: |
        ...
      criteria: |
        ...
      confidence_threshold: "0.7"

  config_name_2:
    primitive: investigate
    params:
      question: |
        ...
      scope: |
        ...
```

### Reference Syntax

**Sequential mode** — reference specific step names:

```yaml
${input.field_name}              # Case data
${step_name.field}               # Output from a named step
${step_name.data.source_name}    # Retrieved data source
```

**Agentic mode** — use `_last_` to avoid hardcoding step names:

```yaml
${input.field_name}              # Case data (same as sequential)
${_last_retrieve.data.source}    # Most recent retrieve step
${_last_classify.category}       # Most recent classify step
${_last_investigate.finding}     # Most recent investigate step
${_last_think.thought}           # Most recent think step (freeform reasoning)
${_last_think.conclusions}       # Think step key takeaways
${_last_think.decision}          # Think step recommendation
${_last_generate.artifact}       # Most recent generate step
${_last_challenge.survives}      # Most recent challenge step
${_last_verify.conforms}         # Most recent verify step
${previous.field}                # Whatever step ran most recently
```

### Primitive Param Reference

| Primitive | Required Params | What They Do |
|-----------|----------------|--------------|
| **retrieve** | `specification` | Describes what data to fetch and from where |
| **classify** | `categories`, `criteria` | Category definitions and decision criteria |
| **investigate** | `question`, `scope` | What to investigate and how deep |
| **think** | `instruction` | Freeform reasoning prompt — never terminal, feeds a structured step |
| **verify** | `rules` | Rules/criteria to check against |
| **generate** | `requirements`, `format`, `constraints` | What to produce, in what format, with what rules |
| **challenge** | `perspective`, `threat_model` | Adversarial perspective and vulnerability definitions |

### Tips for Good Domain Configs

1. **Write like you're training a new hire.** If a competent analyst could follow your description and get it right 80-90% of the time, that's sufficient. The framework escalates the rest.

2. **Include specific thresholds where they exist in policy.** "$500,000 limit" and "policy must be active" are real business rules — include them. But don't invent scoring formulas.

3. **Describe what to look for, not how to score it.** "Look for rapid filing, high frequency, and amount escalation" is better than "Pattern P1: days_active < 90, score +25 points."

4. **Say what the output should contain.** List the JSON fields you need. The framework ensures it's valid JSON with the right structure.

5. **Describe when to escalate.** "If the risk is high and patterns are found, always recommend SIU referral" is a domain rule. Confidence-based escalation is automatic.

6. **Don't write coherence tables or enum definitions.** Tell the LLM that the risk level should inform its judgment. It will be directionally consistent. If it's not, that's a prompt problem, not a schema problem.

7. **Domain specs could be RAG'd from policy documents.** If your domain spec reads like something you could extract from an existing policy manual, you're at the right level of abstraction. If it reads like code, you've gone too far.

### LLM Parameters

Model and temperature can be set at multiple levels. Each level overrides the one above it:

```
CLI default (--model, --temperature)
  └→ Workflow orchestrator config (agentic only)
      └→ Domain primitive_config (per invocation)
          └→ Step override (sequential only)
```

**Sequential mode** — set `model` and `temperature` on any step:

```yaml
steps:
  - name: classify_hardship
    primitive: classify
    model: gemini-2.5-pro          # Override for this step only
    temperature: 0.3                # Higher temp for classification
    params:
      categories: "..."
```

**Agentic mode** — set `model` and `temperature` on any primitive_config:

```yaml
primitive_configs:
  challenge_guidance:
    primitive: challenge
    model: gemini-2.5-pro          # Use Pro for the critical review
    temperature: 0.1                # Keep it precise
    params:
      perspective: "..."

  generate_guidance:
    primitive: generate
    temperature: 0.4                # More creative for member-facing content
    params:
      requirements: "..."
```

**Orchestrator model** — set in the workflow YAML:

```yaml
orchestrator:
  model: gemini-2.5-pro            # Override orchestrator model
  strategy: "..."
```

**Default:** If nothing is set, all steps use the CLI `--model` (default: `gemini-2.0-flash`) and `--temperature` (default: `0.1`).

**When to override:**
- **Higher temperature (0.3-0.5):** Generate steps producing creative or conversational content
- **Lower temperature (0.0-0.1):** Verify, challenge, and classify steps where precision matters
- **Different model:** Challenge steps that need stronger reasoning, or generate steps that need better writing

---

## Step 3: Build the Case Data

The case JSON provides runtime data for a specific execution. In dev/test, it's a static file. In production, it's assembled from API calls.

### Structure

```json
{
  "source_name_1": {
    "field_a": "value",
    "field_b": 123
  },
  "source_name_2": {
    "field_c": ["item1", "item2"]
  }
}
```

Each top-level key becomes a tool in the registry. The retrieve step fetches them by name.

### Key Rules

1. **Top-level keys = data sources.** The retrieve specification in the domain config references these by name.

2. **Field names must match your references.** If the domain config says `${input.member_profile.military_status}`, the case JSON must have `member_profile.military_status` at that exact path.

3. **Use the Schema Glossary.** Production systems use different field names. The glossary maps production field names to framework field names. See `docs/schema_glossary.yaml`.

---

## Production Patterns

Two kinds of patterns: **workflow patterns** (for AI engineers building the
plumbing) and **domain authoring guidance** (for SMEs describing the process).

### Domain Authoring: Write Policy, Not Code

The domain spec should read like a policy document — the kind you'd hand to a
smart new hire on their first day. If a reasonable analyst could follow your
description and get it right 80-90% of the time, that's sufficient. The other
10-20% will escalate to humans automatically via the framework's confidence
thresholds and governance tiers.

**Do:**
- Write categories as plain descriptions with concrete examples
- Write eligibility rules as numbered checklists a human could follow
- Include specific thresholds where they exist ("under $500K", "active status")
- Describe what patterns to look for, not how to score them
- Say what the output should contain, not how to compute it
- Describe when a human should take over

**Don't:**
- Write scoring formulas with point values
- Define state machines or coherence matrices
- Specify exact confidence thresholds per step
- Write enum definitions and validation rules
- Try to make the LLM's reasoning fully deterministic

**Example — good fraud screening spec:**
```yaml
investigate_patterns:
  question: |
    Look for these fraud patterns in the claim data:
    1. "Rapid filing" — Did they file very soon after getting the policy?
    2. "High frequency" — Multiple claims recently?
    3. "Amount escalation" — This claim much larger than prior ones?

    If nothing suspicious, recommend clearing the claim.
    If concerns, recommend SIU referral.
    High-risk claims with even one pattern are more concerning
    than low-risk claims with the same pattern.
```

**Example — over-engineered (don't do this):**
```yaml
investigate_patterns:
  question: |
    Pattern P1: days_active < 90
    Pattern P2: prior_claims >= 2 in last 12 months
    Pattern P3: amount > 2x average(prior_claims.amount)
    Finding: 0 patterns = "no_suspicious_patterns", 1-2 = "minor", 3+ = "significant"
    COHERENCE TABLE:
      low_risk → no_suspicious_patterns or minor
      high_risk → significant only
```

The first version lets the LLM apply judgment. The second version is a rules
engine written in YAML — at that point, just write Python.

### Workflow Patterns (For Engineers)

These patterns apply to the workflow YAML, not the domain spec. They prevent
structural failures that the LLM can't recover from.

#### Pattern 1: Always Branch on Verify Failures

**Never** let a workflow continue linearly past a verify step. If verification
fails, the downstream steps receive garbage context and produce contradictory
artifacts (e.g., `conforms=False` but `decision=auto_approve`).

```yaml
# ✅ CORRECT: conditional routing
- name: check_eligibility
  primitive: verify
  transitions:
    - when: "output.conforms == false"
      goto: generate_denial       # skip risk assessment entirely
    - default: assess_risk

# ❌ WRONG: linear flow continues regardless
- name: check_eligibility
  primitive: verify
  transitions:
    - default: assess_risk        # runs even when ineligible
```

This applies to every verify step: eligibility checks, documentation checks,
compliance checks. If verification fails, route to a denial/incomplete artifact
or terminate. Never let the "happy path" steps execute with failed verification.

### Pattern 2: Curate Verify Inputs with `subject`

The verify primitive sees the full workflow context by default — every field
from every prior step. When extraneous data is present (e.g., fraud flags in
an eligibility check), the LLM will use it to invent violations.

**Always** provide a curated `subject` param listing only the fields relevant
to the rules being checked:

```yaml
- name: check_eligibility
  primitive: verify
  params:
    rules: "${domain.check_eligibility.rules}"
    subject: |
      Policy status: ${gather_data.data.get_policy.status}
      Coverage type: ${gather_data.data.get_policy.coverage_type}
      Claim amount: ${gather_data.data.get_claim.amount}
      Incident date: ${gather_data.data.get_claim.incident_date}
    additional_instructions: |
      Check ONLY rules E1-E4. IGNORE flags, risk factors, or other
      data that may appear in context.
```

Without `subject`, the LLM sees the full case JSON including fields like
`flags: ['fraud_indicator', 'repeat_claimant']` and cannot resist using them.

### Workflow Patterns (For Engineers)

These patterns apply to the workflow YAML, not the domain spec. They prevent
structural failures that the LLM can't recover from.

#### Pattern 1: Always Branch on Verify Failures

**Never** let a workflow continue linearly past a verify step. If verification
fails, the downstream steps receive garbage context and produce contradictory
artifacts (e.g., `conforms=False` but `decision=auto_approve`).

```yaml
# ✅ CORRECT: conditional routing
- name: check_eligibility
  primitive: verify
  transitions:
    - when: "output.conforms == false"
      goto: generate_denial       # skip risk assessment entirely
    - default: assess_risk

# ❌ WRONG: linear flow continues regardless
- name: check_eligibility
  primitive: verify
  transitions:
    - default: assess_risk        # runs even when ineligible
```

#### Pattern 2: Curate Verify Inputs

Verify steps see the full context by default. The LLM will invent violations
from extraneous data (e.g., seeing fraud flags and failing an eligibility
check because the flags "look suspicious"). Use the `subject` param to
limit what the verify step sees.

```yaml
- name: check_eligibility
  primitive: verify
  model: standard
  params:
    rules: "${domain.check_eligibility.rules}"
    subject: |
      Policy status: ${gather_claim_data.data.get_policy.status}
      Coverage: ${gather_claim_data.data.get_policy.coverage_type}
      Amount: ${gather_claim_data.data.get_claim.amount}
```

#### Pattern 3: Artifacts Must Be Objects, Never Strings

The generate prompt must produce artifacts as JSON objects, not JSON strings.
The generate prompt template (`registry/prompts/generate.txt`) enforces this
with explicit CORRECT/WRONG examples. JSON-inside-a-string is the #1 cause
of parse failures.

#### Pattern 4: Use Model Tiers for Hard Steps

Fast/cheap models (Flash, mini) are fine for classification and retrieval.
Use `model: standard` for verify steps with complex rules and think steps
that need multi-factor reasoning.

#### Pattern 5: Gate Delegations on Eligibility

Delegation policies fire on data conditions (e.g., `amount >= 5000`).
Without an eligibility precondition, delegations fire on denied claims.

```yaml
delegation_policies:
  - name: fraud_screening
    conditions:
      - selector: last_verify
        field: conforms
        operator: eq
        value: true            # only delegate if eligible
```

#### Pattern 6: Two Gate Types, Two Policies

**Governance gates** (mandatory review tiers) can be auto-approved in CI/eval.
**Quality gates** (parse errors, schema violations) must block in CI.
Use `--auto-approve --strict-gates` for honest CI results.

---

## Invariant Reference

The eval harness enforces these structural invariants on every case. They
verify the framework is routing correctly — not that the LLM made the right
domain decision.

| ID | Name | What It Checks |
|----|------|----------------|
| I1 | Deny path routing | `conforms=False` → assessment steps must NOT run, denial step MUST run |
| I2 | Parse integrity | No generate step may have `PARSE ERROR` in artifact |
| I3 | Artifact schema | Generate artifacts must be dicts with required keys present |
| I4 | Confidence floor | No generate step may have `confidence=0.0` |
| I5 | Quality gates | Quality gate fired → fail in `--strict-gates` mode |
| I6 | No denied delegations | `conforms=False` → zero delegations must fire |
| I7a | Fraud coherence | Risk category and finding should be directionally consistent |
| I7b | Damage routing | Doc conformance routes to correct generate step |

**Philosophy:** Invariants check structure, not semantics. I1 checks that a
denied claim didn't proceed to assessment — not whether the denial was correct.
I3 checks that the artifact has the right keys — not whether the values are
right. The LLM handles semantics; the framework handles plumbing.

---

## Checklist for a New Use Case

```
□ Workflow selected (or new one created if pattern is genuinely new)
□ Domain config written with SME input
□ All ${domain.*} references in workflow resolve to domain config keys
□ All ${input.*} references in domain config match case JSON structure
□ All transition targets in workflow reference valid step names
□ Case JSON has all sources listed in retrieve specification
□ Run with --validate-only to check structure
□ Run with synthetic case data to verify end-to-end
□ Review trace output with SME for correctness
□ Add to schema glossary if new data sources are introduced
```

---

## Running

```bash
# Sequential (workflow + domain + case)
python -m engine.runner \
  -w workflows/WORKFLOW.yaml \
  -d domains/DOMAIN.yaml \
  -c cases/CASE.json

# Agentic (workflow + domain + case, same command)
python -m engine.runner \
  -w workflows/WORKFLOW_agentic.yaml \
  -d domains/DOMAIN_agentic.yaml \
  -c cases/CASE.json

# Validate only (no LLM calls)
python -m engine.runner \
  -w workflows/WORKFLOW.yaml \
  -d domains/DOMAIN.yaml \
  -c cases/CASE.json \
  --validate-only

# Save full state for debugging
python -m engine.runner \
  -w workflows/WORKFLOW.yaml \
  -d domains/DOMAIN.yaml \
  -c cases/CASE.json \
  --output state.json --verbose
```
