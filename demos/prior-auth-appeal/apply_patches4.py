from pathlib import Path
import sys
f = Path("domains/prior_auth_appeal.yaml")
t = f.read_text(encoding="utf-8")
OLD = """  5. CHALLENGE BEFORE DISPOSING.
     The challenge step examines the proposed disposition from the
     perspective of the party that lost: if overturning, challenge as
     the plan's medical director would. If upholding, challenge as the
     member's attorney would.

  6. REFLECT IF CHALLENGE FAILS OR SOURCES CONFLICT.
     Use reflect to diagnose which assumption is driving the conflict
     and whether it has been verified across all three sources.

  7. GOVERN ROUTES TO APPROPRIATE REVIEWER.
     IMR-level cases go to GATE. Clear approvals/denials may go to AUTO
     if all criteria are unambiguously met. Never AUTO when sources conflict."""
NEW = """  5. CHALLENGE THE DISPOSITION EXACTLY ONCE.
     After deliberate and generate complete, run challenge once, from
     the perspective of the party that lost: if overturning, challenge
     as the plan's medical director would; if upholding, as the
     member's attorney would. Do not challenge again later in the
     workflow under any circumstances.

  6. REFLECT ADJUDICATES THE CHALLENGE - THEN THE LOOP ENDS.
     After challenge, run reflect once to adjudicate:
     - Challenge identified a genuine flaw in the epistemic basis
       (missing evidence, misapplied rule, unexamined hypothesis):
       re-deliberate ONCE, regenerate, then proceed DIRECTLY to govern.
       Never challenge the revised determination.
     - Challenge identified no genuine flaw: preserve the determination
       and proceed DIRECTLY to govern. Unresolved tension is input for
       govern's tier escalation, not grounds for another cycle.

  7. GOVERN ROUTES TO APPROPRIATE REVIEWER.
     IMR-level cases go to GATE. Clear approvals/denials may go to AUTO
     if all criteria are unambiguously met. Never AUTO when sources
     conflict. Unresolved challenge tension raises the tier; it never
     reopens reasoning."""
if NEW.splitlines()[0] in t:
    print("[ALREADY APPLIED]")
elif OLD in t:
    f.write_text(t.replace(OLD, NEW), encoding="utf-8")
    print("[OK] orchestrator strategy: single-challenge termination semantics")
else:
    print("[FAILED] anchor not found - send me lines 43-81 of domains/prior_auth_appeal.yaml")
    sys.exit(1)
