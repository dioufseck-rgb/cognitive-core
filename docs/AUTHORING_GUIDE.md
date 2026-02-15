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

### Sequential Workflows

| Workflow | Pattern | Good For |
|----------|---------|----------|
| `dispute_resolution` | Retrieve → Classify → Verify → Investigate → Classify → Generate → Challenge | Case with intake data, a decision, and member-facing response |
| `loan_hardship` | Classify → Investigate (branching) → Generate → Challenge | Assessment with branching paths based on classification |
| `complaint_resolution` | Classify ×2 → Investigate → Generate → Challenge | Member complaint processing with severity routing |
| `spending_advisor` | Retrieve → Classify → Investigate → Generate → Challenge | Conversational Q&A over data |
| `nurse_triage` | Classify ×2 → Investigate → Verify → Generate → Challenge | Safety-critical with mandatory verification |
| `regulatory_impact` | Classify → Investigate (adaptive) → Classify → Generate → Challenge | Analysis with adaptive investigation depth |
| `sar_investigation` | Classify → Investigate → Classify → Generate → Challenge → Verify | Investigative workflow ending in compliance check |

### Agentic Workflows

| Workflow | Good For |
|----------|----------|
| `loan_hardship_agentic` | Complex cases where Think synthesis may change the routing decision |
| `spending_advisor_agentic` | Exploratory analysis where the path depends on what's discovered |
| `complaint_resolution_agentic` | Complaints needing adaptive depth — simple vs escalation |
| `nurse_triage_agentic` | Ambiguous symptoms requiring differential reasoning |
| `regulatory_impact_agentic` | Multi-business-line regulations with cross-impact analysis |
| `sar_investigation_agentic` | Complex suspicious activity with multiple patterns |

### Decision: Sequential or Agentic?

Use **sequential** when:
- The path is well-understood
- Auditability and predictability matter
- Production deployment

Use **agentic** when:
- The optimal path depends on what intermediate steps discover
- Think synthesis might change routing (escalate vs resolve directly)
- Prototyping a new use case before crystallizing the pattern

**Lifecycle**: prototype in agentic → observe which paths the orchestrator takes → crystallize the most common paths into a sequential workflow.

---

## Step 2: Write the Domain Config

The domain file is where SME expertise lives. It defines:

- **Categories** for classification steps
- **Investigation scope** and effort level
- **Rules** for verification steps
- **Requirements and constraints** for generation
- **Threat models** for challenge
- **Orchestrator strategy** (agentic only)

### Sequential Domain Example

```yaml
# domains/member_complaint.yaml
domain_name: member_complaint
workflow: complaint_resolution

classify_complaint_type:
  categories: |
    - service_inconsistency: Conflicting info from different agents.
    - fee_dispute: Disputes a fee or charge.
    - process_failure: Process didn't work as designed.
    - multi_issue: Spans multiple categories.
  criteria: |
    Classify the ROOT problem, not surface frustration.
    If fees resulted from a process failure, classify as process_failure.

investigate_complaint:
  effort_level: deep
  scope: |
    1. ROOT CAUSE: Trace timeline from member's perspective
    2. FINANCIAL IMPACT: Quantify all harm
    3. REMEDIATION: Fee reversal? Apology level? Process fix?

generate_response:
  format: member_response_letter
  constraints: |
    - Never blame the member
    - Include specific dollar amounts
    - Under 400 words

challenge_response:
  perspective: |
    1. THE MEMBER: Do I feel heard?
    2. COMPLIANCE: Any UDAAP concerns?
    3. RETENTION: Is this enough to keep a $45K member?
  threat_model: |
    CRITICAL: Blaming member, factual errors, missing remediation
    HIGH: Generic apology, no fee reversal, no follow-up contact
```

### Agentic Domain Example

Agentic domains add `goal`, `orchestrator_strategy`, and `primitive_configs`:

```yaml
# domains/member_complaint_agentic.yaml
domain_name: member_complaint_agentic
workflow: complaint_resolution_agentic

goal: |
  Process complaint and produce response or escalation brief.
  UDAAP applies. Must be challenged before delivery.

orchestrator_strategy: |
  - Classify type and severity first
  - If critical → Think about escalation vs direct resolution
  - If high-priority → deep investigation, strong response
  - Generate escalation brief if Think recommends it

primitive_configs:
  classify_complaint:
    primitive: classify
    temperature: 0.05
    params:
      categories: |
        ...
      criteria: |
        Complaint: ${input.complaint_text}
        ...
```

### Reference Syntax

| Reference | Resolves | When |
|-----------|----------|------|
| `${domain.section.field}` | Domain YAML value | Merge time |
| `${input.field}` | Case JSON value | Runtime |
| `${step_name.field}` | Prior step output | Runtime |
| `${_last_classify.field}` | Most recent classify output | Runtime (agentic) |
| `${previous.field}` | Immediately prior step | Runtime (sequential) |

---

## Step 3: Create a Case File

The case JSON contains the runtime data — member profile, transactions, loan details, symptoms, regulations, etc.

```json
{
  "member_id": "MBR-2019-55023",
  "member_name": "Michael Torres",
  "member_profile": {
    "name": "Michael Torres",
    "tenure_years": 6,
    "products": ["checking", "savings", "auto_loan"],
    "total_relationship_value": 45000
  },
  "complaint_text": "I am extremely frustrated...",
  "account_context": {
    "total_fees": 50.00,
    "bounced_payments": [...]
  }
}
```

In production, this JSON comes from APIs. In dev/test, it's a file.

---

## Step 4: Test

```bash
# Validate config (no LLM calls)
python -m engine.runner \
  -w workflows/complaint_resolution.yaml \
  -d domains/member_complaint.yaml \
  --validate-only

# Run with tracing
python -m engine.runner \
  -w workflows/complaint_resolution.yaml \
  -d domains/member_complaint.yaml \
  -c cases/complaint_torres.json

# Save output for inspection
python -m engine.runner \
  -w workflows/complaint_resolution.yaml \
  -d domains/member_complaint.yaml \
  -c cases/complaint_torres.json \
  --output results/complaint_torres_output.json
```

---

## Step 5: Iterate

Tuning happens at three levels:

1. **Domain config** — adjust categories, criteria, scope, constraints, threat models. This is the primary tuning surface. Changes affect one use case.

2. **Prompt templates** — adjust `registry/prompts/*.txt` for cross-cutting improvements. Changes affect ALL workflows using that primitive.

3. **Model/temperature** — use the parameter cascade. Challenge at `gemini-2.5-pro` for adversarial diversity. Generate at higher temp for conversational warmth.

---

## Seven Primitives Reference

| Primitive | Required Params | Schema | When to Use |
|-----------|----------------|--------|-------------|
| **Retrieve** | `specification` | `RetrieveOutput` | First step — gather data before analysis |
| **Classify** | `categories`, `criteria` | `ClassifyOutput` | Route by type, severity, or decision category |
| **Investigate** | `question`, `scope` | `InvestigateOutput` | Analyze data, test hypotheses, find root cause |
| **Think** | `instruction` | `ThinkOutput` | Synthesize multiple findings, weigh tradeoffs |
| **Verify** | `rules` | `VerifyOutput` | Rule-by-rule compliance check |
| **Generate** | `requirements`, `format`, `constraints` | `GenerateOutput` | Produce member-facing or internal documents |
| **Challenge** | `perspective`, `threat_model` | `ChallengeOutput` | Adversarial review — always last before END |

---

## Agentic Constraints Reference

| Constraint | Purpose |
|------------|---------|
| `max_steps` | Hard ceiling on total primitive invocations |
| `max_repeat` | Max times same primitive can be invoked with same step_name |
| `must_include` | Primitives required in every run |
| `must_end_with` | Terminal primitive (typically `challenge`) |
| `challenge_must_pass` | Cannot end if last challenge failed |
