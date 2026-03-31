# Fraud Investigation

**Pattern:** P05 — Investigation and Reporting  
**Overlays:** governance  
**Coordinator:** parallel-handlers  

Structured multi-workflow fraud investigation. A triage workflow classifies fraud type and routes to a specialty investigation. The specialty investigation retrieves evidence, investigates the fraud pattern, and fires two parallel handlers — regulatory review (SAR determination) and case resolution (adversarial challenge, member notification, case summary). Resumes at report generation when both handlers return.

## Workflow structure

```
fraud_triage                      classify type → fire-and-forget → specialty
  └─► fraud_specialty_investigation   retrieve → investigate → deliberate
        ├─► fraud_regulatory_review     [parallel] verify → generate → govern
        └─► fraud_case_resolution       [parallel] challenge → generate → govern
              └─► generate_final_report  [primary resumes] synthesize all results
```

## Files

```
fraud-investigation/
  fraud_investigation.yaml              ← pack overview (type-specific knowledge in domains/)
  coordinator_config.yaml  ← delegation policies, governance tiers, queues
  run.py                   ← runner script
  workflows/
    fraud_triage.yaml
    fraud_specialty_investigation.yaml
    fraud_regulatory_review.yaml
    fraud_case_resolution.yaml
  domains/
    fraud_triage.yaml
    card_fraud.yaml          ← card-not-present, skimmer, friendly fraud
    check_fraud.yaml         ← counterfeit, altered amount, duplicate
    app_scam_fraud.yaml      ← romance, investment, impersonation, grandparent
    fraud_regulatory.yaml    ← SAR determination, BSA compliance
    fraud_case_resolution.yaml ← adversarial challenge, member notification
  cases/
    card_fraud_cnp.json
    check_fraud_altered_amount.json
    app_scam_romance.json
```

## Setup

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your_key_here

# Run card fraud case (default)
python library/domain-packs/fraud-investigation/run.py

# Run specific fraud type
python library/domain-packs/fraud-investigation/run.py --fraud-type check_fraud
python library/domain-packs/fraud-investigation/run.py --fraud-type app_scam

# Run custom case
python library/domain-packs/fraud-investigation/run.py --case path/to/case.json
```

## Customising

This pack ships with domain knowledge for a financial institution. To adapt:

**For a different institution:**
- `domains/card_fraud.yaml` — update investigate scope and determination framework for your card product
- `domains/check_fraud.yaml` — update for your check processing environment
- `domains/app_scam_fraud.yaml` — update vulnerability categories and recovery channels
- `domains/fraud_regulatory.yaml` — update SAR thresholds, regulatory references, filing deadlines
- `coordinator_config.yaml` — update queue names, SLAs, and governance tier queues

**For a different fraud domain (insurance, healthcare billing, etc.):**
The pack structure applies directly — the specialty investigation, regulatory review, and case resolution workflows are domain-agnostic. Replace the domain YAMLs with your fraud typology and regulatory framework.

## Coordinator failure modes

This pack uses `parallel-handlers` — both regulatory review and case resolution must complete before the specialty investigation resumes. Understand these failure modes:

| Failure | Behaviour |
|---------|-----------|
| Handler timeout | Coordinator escalates to HOLD queue. Primary stays suspended. |
| Handler error | Error injected as `${input.delegation.<name>.error}` in primary context. Report step handles gracefully. |
| Partial completion | Primary waits for ALL handlers. One fast handler waits for a slow one. |

SLAs in `coordinator_config.yaml`:
- Regulatory review: 7200s (2 hours)  
- Case resolution: 14400s (4 hours)

Adjust to match your operational requirements.

## Governance defaults

| Tier | Condition |
|------|-----------|
| AUTO | Triage only (no specialist review) |
| GATE | Specialist investigation complete — analyst reviews full package before action |
| HOLD | BSA officer review for SAR filings or high-value cases |

The specialty investigation always routes to GATE — the analyst receives the complete integrated package (investigation finding + regulatory determination + adversarial challenge) before any action executes.

## Scaffold boundary

Type-specific variation (fraud typology, regulatory thresholds, recovery procedures) lives in the domain YAMLs and is fully customisable via scaffold. 

Structural changes requiring workflow edits:
- Adding a third parallel handler (e.g. asset recovery workflow)
- Different delegation trigger conditions (e.g. fire regulatory only on confirmed fraud)
- Adding a post-resolution follow-up workflow (sequential-lifecycle coordinator)
