# Cognitive Core — Demo Talking Points

## Setup (before the audience arrives)

- Terminal open, font size 16+, dark theme
- `export GOOGLE_API_KEY=...` already set
- Run `python -m engine.runner -w workflows/dispute_resolution.yaml -d domains/card_dispute.yaml -c cases/card_clear_fraud.json --validate-only` once to confirm everything works
- Have the presentation deck open as backup (but the terminal IS the demo)

## Opening (2 minutes)

**Say:** "I want to show you a working system, not a slide deck. What you're about to see is a framework that decomposes AI workflows into composable cognitive operations — seven building blocks that compose into any use case we need. I built this to answer the question: what does an AI factory actually look like when you run it?"

**Say:** "Everything runs on Gemini Flash at about $0.002 per workflow execution. The challenge step uses Pro for adversarial review — different model, different blind spots. Total cost for what you'll see today is under a nickel."

## Demo 1: Card Dispute — Sequential (3 minutes)

**Run it.** While it's running:

**Say:** "Watch the trace output. Each line is a cognitive primitive executing — classify, verify, investigate, generate, challenge. These are typed operations with structured inputs and outputs. Not prompt chaining. Not 'here's some context, figure it out.'"

**Point out:** "See that deterministic route? Classification came back at 0.95 confidence — unauthorized transaction, clear fraud. The workflow skipped investigation entirely and went straight to resolution. That's a conditional branch evaluated without calling the LLM."

**After it finishes, say:** "That's five primitives, sequential, about 20 seconds. Now I'm going to show you the same framework on a completely different use case without changing any code."

## Demo 2: Loan Hardship — Sequential (3 minutes)

**Run it.** While it's running:

**Say:** "Same engine, same primitives, different YAML files. The workflow file defines the cognitive pattern — classify, investigate, verify, generate, challenge. The domain file supplies the expertise — SCRA rules, military transition categories, regulatory constraints. The case file has Angela Reeves' data."

**Point out:** "This is the three-layer separation. When compliance updates SCRA rules, they edit the domain YAML. When we improve the workflow logic, every domain that uses this pattern inherits the improvement. When a new member calls, their data flows into a new case file. Each layer changes independently."

**Transition:** "Now let me show you what happens when we let the AI decide its own path."

## Demo 3: Spending Advisor — Agentic (5 minutes)

**This is the money demo. Take your time.**

**Run it.** While it's running:

**Say:** "This is agentic mode. Instead of a predetermined path, an LLM orchestrator reads the accumulated state after each step and decides what to do next. Same seven primitives, same typed outputs, same audit trail — but the sequence emerges at runtime."

**Watch for the generate → challenge loop.** When challenge fails:

**Say:** "There it is. The generator produced advice with a factual error — [describe what happened]. The challenge step, running on a *different model*, caught it. Now watch what the orchestrator does."

**If orchestrator reinvestigates instead of retrying:**

**Say:** "It went back to investigate instead of just retrying generation. It recognized the problem wasn't a wording issue — it was a data gap. That's adaptive behavior within constraints."

**After it finishes:** "Three attempts to get from factually wrong to accurate, comprehensive, personalized. Every iteration improved. The adversarial loop is the quality mechanism."

## Demo 4: Military Hardship — Agentic + Think (5 minutes)

**This is the insight demo.**

**Say:** "Same Angela Reeves case as Demo 2. Same data. But now the orchestrator has the Think primitive available and full autonomy within constraints. Watch what decision it makes."

**Run it.** While it's running:

**Point out the dual investigation:** "Two investigations — military protections and financial situation separately. The orchestrator decided it needed both before it could reason about the case."

**Point out Think:** "Now it's invoking Think — freeform synthesis. It's connecting military transition, income uncertainty, SCRA complexity, two different loans, perfect payment history. It's reasoning about what the *combination* means."

**Point out the routing decision:** "Think concluded: too many unknowns for an automated letter. And look — the orchestrator routed to specialist escalation instead of generating a member letter. Same input data as Demo 2, but a fundamentally different and safer decision."

**After it finishes:** "Demo 2 produced a member letter with conditional language. Demo 4 produced a specialist escalation brief that passed challenge with zero vulnerabilities. Forty extra seconds to make the right decision for a regulated environment."

**This is the key point:** "The Think primitive didn't just produce better output. It changed the routing. It gave the orchestrator permission to conclude that the right action was *not* to generate."

## Demo 5: Validate-Only (2 minutes)

**Say:** "Let me show you what it takes to add a new use case. No code. Just configuration."

**Run validate-only on both sequential and agentic configs.**

**Say:** "That's the configuration layer. Step names, primitive types, transition modes, loop limits. A new use case means writing a domain YAML file — categories, rules, constraints. A compliance officer can review it because it's written in their language."

**Point out:** "We have 8 workflows and 9 domain configs today. Each combination is a production use case. The marginal cost of use case thirty-one is writing a YAML file."

## Closing (2 minutes)

**Say:** "Five demos. Same seven primitives. Same engine. Sequential mode for production — deterministic, auditable, fast. Agentic mode for discovery — finds optimal paths that we then crystallize into sequential."

**For leadership audience:** "This is the reference architecture. When we evaluate AI Factory vendors, we ask them to demonstrate these same capabilities live. If they can't show adversarial validation catching a real error, they haven't solved the production quality problem."

**For tech lead audience:** "The framework is yours to extend. Adding a primitive is: prompt template, output schema, register it. Adding a use case is: domain YAML. The engine handles structured output parsing, state management, loop detection, error recovery, and tracing — once, for every use case."

**Close with:** "Seven primitives. Three layers. Two modes. That's the factory."

## Q&A Prep — Likely Questions

**"How does this compare to LangChain/LangGraph/CrewAI?"**
This is a layer above those frameworks. They describe how information flows (prompt chaining, routing). We describe what thinking is performed (classify, investigate, challenge). We run on LangGraph today. The cognitive layer is portable.

**"What about hallucination?"**
Three mechanisms: (1) Investigate forces data extraction before hypothesis formation — no hallucinated trends. (2) Challenge with a different model catches factual errors the generator misses. (3) Every output field is traceable to evidence_used and evidence_missing.

**"How does this work with our existing APIs?"**
The Retrieve primitive uses a tool registry with pluggable providers — API calls, vector DB, MCP servers. In production, you register your endpoints. The framework handles assembly.

**"What about latency?"**
Sequential workflows: 15–30 seconds. Agentic with self-correction: 60–140 seconds. The question is whether you can afford *not* to have the correction loop — one wrong number in regulated finance is a compliance event.

**"Can we use Azure OpenAI / Anthropic instead of Gemini?"**
The engine abstracts model calls. Change the model string and provider config. The primitives and workflows are model-agnostic.

**"What happens when the LLM produces garbage JSON?"**
Four-level parser: standard JSON → strip markdown fences → repair common syntax errors → regex extraction. If all fail, the engine detects the failure and either retries or gracefully degrades (Retrieve continues with data even if LLM assessment fails to parse).

**"Why seven primitives and not five or ten?"**
Started as five. Retrieve and Think emerged from implementation. The framework is designed so adding a primitive is: prompt template + output schema + registration. If an eighth emerges, it slots in.

**"How do we get this to production?"**
Phase 1: Harden prompts and schemas on real NFCU data. Phase 2: Production Retrieve providers for our APIs. Phase 3: Sequential workflows for the first 3 use cases. Phase 4: AIOps (monitoring, A/B testing, drift detection).
