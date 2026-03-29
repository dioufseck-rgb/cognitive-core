# Loan Application Review

A regulated decision workflow using six of the eight cognitive primitives.
This is what Cognitive Core looks like at production depth.

## What it shows

- All six epistemic primitives in one workflow: `retrieve`, `classify`,
  `investigate`, `deliberate`, `verify`, `govern`
- The `deliberate` primitive producing a warranted recommendation with
  explicit logical connection between evidence and action
- Regulatory compliance as a `verify` step — not ad hoc checks
- Four governance outcomes across four realistic applicant profiles
- How the `govern` primitive reads accumulated state to determine tier

## Run it

```bash
export GOOGLE_API_KEY=your-key
python demos/loan-application-review/run.py
```

## The workflow

```
gather_application (retrieve)
    ↓
classify_risk (classify)
    ├── high_risk → investigate_risk_factors (investigate)
    │                   ↓
    ├── ineligible ─────┤
    └── prime/near_prime ┘
                        ↓
         deliberate_recommendation (deliberate)
                        ↓
            verify_compliance (verify)
                        ↓
              govern_decision (govern)
                        ├── auto → proceed
                        ├── spot_check → proceed + sample
                        ├── gate → suspend → underwriter queue
                        └── hold → hold → compliance queue
```

## The four governance outcomes

| Application | Risk tier | Recommendation | Governance |
|-------------|-----------|----------------|------------|
| APP-001 Priya Sharma | prime | approve | auto |
| APP-002 Tomás Rivera | near_prime | approve_modified | gate |
| APP-003 Diane Whitfield | high_risk | refer_specialist | gate |
| APP-004 Marcus Webb | ineligible | decline | hold |

## Why this matters

A traditional approach would hard-code decision trees and score thresholds.
This approach expresses the **reasoning logic** in domain YAML. The LLM
applies that logic to each case, producing a warranted conclusion.

The `deliberate` step's `warrant` field is the logical connection between
the applicant's financial picture and the recommendation — required for
adverse action compliance. It is generated, not templated.

The `govern` step's `AccountabilityChain` records every governance decision
made during the instance's lifetime. Built-in audit trail.

## Customise it

- Change `classify_risk.criteria` to use different risk thresholds
- Add a `challenge` step after `deliberate_recommendation` for adversarial review
- Add a delegation policy to trigger identity verification for high-risk cases
- Add a second domain (`domain_auto_loan.yaml`) with different parameters
