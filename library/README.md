# Cognitive Core — Domain Library

The domain library makes the framework's core claim operational: **a new institutional decision domain requires configuration, not construction.**

The library supplies the workflow side of that equation — canonical patterns, composable overlays, and coordinator templates so well-designed that domain knowledge is the only variable a new adopter needs to supply.

---

## Start here: Domain Packs

Domain packs are the front door. Each pack is a complete, ready-to-run governed decision workflow for a specific institutional domain.

| Pack | Pattern | What it decides | Coordinator |
|------|---------|-----------------|-------------|
| [Fraud Investigation](domain-packs/fraud-investigation/) | P05 + parallel-handlers | Fraud determination, SAR routing, case resolution | parallel-handlers |
| [Consumer Lending](domain-packs/consumer-lending/) | P02 | Loan approval with ECOA/FCRA compliance | simple |
| [Content Moderation](domain-packs/content-moderation/) | P04 | Content classification and disposition | simple |
| [Clinical Triage](domain-packs/clinical-triage/) | P06 | Patient contact severity and escalation | simple |
| [Compliance Review](domain-packs/compliance-review/) | P07 | Conformance assessment and compliance memo | simple |

**To use a pack:**

1. Copy the pack directory to your project (or run it from the repo)
2. Open `domain.yaml` and fill in the `[FILL IN]` sections with your domain knowledge
3. Update queue names in `coordinator_config.yaml` to match your infrastructure
4. Run it: `python library/domain-packs/<pack-name>/run.py`

Each pack's README explains exactly what to fill in, what not to change, and where the scaffold boundary sits.

---

## How it works: Four Layers

Under each domain pack, four layers compose to produce the complete workflow.

```
Pattern           The epistemic sequence — which primitives fire in what order
Overlay           Composable modifiers: adversarial, exception, human-review,
                  delegation, governance
Coordinator       Structural relationship between workflows: single workflow,
template          or multi-workflow with delegation chains and suspend/resume
Domain            The only layer domain owners edit — declares pattern +
scaffold          overlays + coordinator, fills in all domain-specific parameters
```

Most adopters never need to touch the pattern, overlay, or coordinator layers. The domain scaffold is the entire surface of customisation for standard domains.

```
library/
  domain-packs/            ← start here
  patterns/                ← 5 base workflow definitions (v1)
  overlays/                ← 5 composable modifiers
  coordinator-templates/   ← 4 structural templates (v1)
```

---

## When you need more than a pack

**Your domain fits an existing pack structurally but your knowledge is different** → fill in the domain scaffold. That is exactly what it is for.

**Your domain is close to an existing pack but needs one extra step** → check whether the step can be expressed as an overlay declaration. The adversarial and exception overlays handle the most common structural additions without workflow changes.

**Your domain needs a fundamentally different step sequence** → you are working with a pattern directly. Read the pattern's `workflow.yaml` and the [pattern design guide](#patterns) below. This is framework work, not scaffold work.

**Your domain needs multiple workflows** → select a coordinator template. The coordinator templates handle the multi-workflow structural shapes. The domain scaffold's `coordinator` section wires the templates to your specific workflows.

---

## Patterns

Five patterns are available in v1. Each ships with a documented `workflow.yaml` that can be used directly as the base for a new domain pack.

| # | Pattern | When to use |
|---|---------|-------------|
| P02 | [Eligibility Determination](patterns/P02-eligibility-determination/) | Entity claims eligibility; rules are explicit; warrant required |
| P04 | [Adversarial Review](patterns/P04-adversarial-review/) | Primary objective is testing a conclusion, not producing one |
| P05 | [Investigation and Reporting](patterns/P05-investigation-and-reporting/) | Open question requiring structured inquiry and formal report |
| P06 | [Triage and Escalation](patterns/P06-triage-and-escalation/) | Severity assessment with response artifact in every case |
| P07 | [Compliance and Conformance](patterns/P07-compliance-and-conformance/) | Conformance check against explicit rules; verify-first design |

The full 13-pattern taxonomy is documented in [`docs/library-architecture.md`](../docs/library-architecture.md). Patterns P01, P03, P08–P13 are on the roadmap for v2.

---

## Overlays

Five overlays are available. Declare them in your domain scaffold header.

| Overlay | What it adds | Declare when |
|---------|-------------|--------------|
| `adversarial` | Challenge step after investigate or classify | Fraud, abuse risk, formal counter-position required |
| `delegation` | Typed work-order delegation to handler workflows | Specialist input needed from another workflow |
| `exception` | Exception-handling branch from verify or classify | Standard rule doesn't fit; deviation may be justified |
| `human-review` | Structured brief + mandatory human checkpoint | Policy requires human approval unconditionally |
| `governance` | Govern step with tier resolution and work order | Always — declared automatically |

In v1, overlays are pre-composed into domain pack variants. Full overlay injection by the composer is a v2 feature.

---

## Coordinator Templates

Four coordinator templates are available. Declare one in your domain scaffold's `coordinator_template` field.

| Template | Structure | Use when |
|----------|-----------|----------|
| `simple` | Single workflow, governance tiers | Decision is self-contained |
| `fire-and-forget` | Primary completes; handler runs independently | Handler result not needed by primary |
| `wait-for-result` | Primary suspends until one handler returns | One specialist workflow required |
| `parallel-handlers` | Primary suspends until N handlers all return | Multiple parallel specialists required |

Sequential and hub-and-spoke coordinator templates are on the roadmap for v2.

**Understand failure modes before using multi-workflow templates.** Handler timeouts, partial completions, and handler errors behave differently in each template. Each template's YAML documents these failure modes.

---

## Adding a new domain pack

The test for whether a new domain belongs in the library:

> Can the workflow be written as a fixed sequence of epistemic steps, and can the domain expertise be written as explicit parameters?

If yes — create a new domain pack:

1. Identify the matching pattern from the table above
2. Copy the pattern's `workflow.yaml` to your new pack directory
3. Copy the appropriate coordinator template as `coordinator_config.yaml`
4. Write `domain.yaml` — declare the pattern, overlays, and coordinator, then fill in each step's parameters
5. Write 3–5 example cases covering the main paths through the workflow
6. Write `run.py` following the pattern in existing packs
7. Write `README.md` documenting what the pack does, what to fill in, and the scaffold boundary

If the answer is "it depends on the case in ways I can't enumerate" — the framework handles the structured portion and hands off at the point of irreducible judgment. Document that boundary in the README.

---

## What this library is not

The library is evidence that the framework works and an onboarding path for teams adopting it. It is not the product. Domain packs are starter kits — the right onboarding unit, not the right long-term packaging unit. A production deployment will assemble composable workflow configurations that the packs anticipated but did not fully define.

The library does not lower the quality bar. It raises the floor. Every workflow built from a library pattern inherits the same governance pipeline, the same audit trail, and the same typed output contracts. The domain knowledge is yours. The institutional structure is built in.
