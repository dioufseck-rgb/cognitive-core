# Consumer Lending

**Pattern:** P02 — Eligibility Determination  
**Overlays:** exception, governance  
**Coordinator:** simple  

Full-chain loan application review with ECOA/FCRA compliance enforcement. Retrieves applicant data, classifies risk tier, investigates high-risk profiles, deliberates on a warranted recommendation, verifies regulatory compliance, and governs disposition.

## What it does

Six steps, two conditional branches:

1. **Retrieve** — pulls credit, financials, employment, banking, identity data
2. **Classify** — assigns risk tier (prime / near_prime / high_risk / ineligible)
3. **Investigate** — fires for high_risk only; examines mitigating factors and default probability
4. **Deliberate** — produces warranted recommendation (approve / approve_modified / approve_with_conditions / decline / refer_specialist)
5. **Verify** — checks ECOA, FCRA, DTI cap, adverse action documentation
6. **Govern** — determines tier (AUTO / SPOT_CHECK / GATE / HOLD) and disposition

**Ineligible** cases skip deliberation and go directly to compliance check — the compliance step records the disqualifier.

## Files

```
consumer-lending/
  consumer_lending.yaml              ← fill in your risk tiers, rules, and thresholds
  workflow.yaml            ← the workflow (do not edit)
  coordinator_config.yaml  ← governance queues and SLAs
  run.py                   ← runner script
  cases/
    applications.json      ← 4 example applications (prime through ineligible)
```

## Setup

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your_key_here
python -m library.domain-packs.consumer-lending.run
```

## Customising

Open `consumer_lending.yaml` and fill in:

| Section | What to change |
|---------|---------------|
| `gather_application.specification` | Your data sources and what each returns |
| `classify_risk.categories` | Your risk tier definitions and score/DTI thresholds |
| `classify_risk.criteria` | Tie-breaking logic between tiers |
| `investigate_risk_factors.scope` | What to analyse for high-risk profiles |
| `deliberate_recommendation.instruction` | Your outcome codes and warrant requirements |
| `deliberate_recommendation.focus` | Your underwriting criteria |
| `verify_compliance.rules` | Your regulatory and policy rules (numbered) |
| `govern_decision.governance_context` | When each tier fires |

## Regulatory rules

The `verify_compliance.rules` section ships with ECOA, FCRA, DTI cap, ability-to-repay, and adverse action requirements. These reflect US consumer lending law. Do not remove or soften these rules without legal review. Replace them with your jurisdiction's equivalent requirements if operating outside the US.

## The four example applications

| ID | Profile | Expected path |
|----|---------|---------------|
| APP-001 | Prime — score 748, DTI 28%, stable employment | approve → AUTO |
| APP-002 | Near-prime — score 681, DTI 41%, 2 late payments | approve_modified or refer_specialist → GATE |
| APP-003 | High-risk — score 594, DTI 48%, unstable employment | investigate → decline → HOLD |
| APP-004 | Ineligible — active bankruptcy, DTI 58% | ineligible → compliance check → HOLD |

## Scaffold boundary

The scaffold covers risk tier definitions, underwriting criteria, compliance rules, and governance thresholds — the large majority of lending variation across institutions.

Structural changes requiring workflow edits:
- Second underwriter review after deliberation (two-stage-review coordinator template)
- Specialist underwriting for high-risk cases in a separate workflow (wait-for-result coordinator template)
- Different workflow for secured vs unsecured products (two separate domain packs)
