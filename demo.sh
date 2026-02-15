#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Cognitive Core — Live Demo Script
# Each scenario presents the case in detail before running.
# ═══════════════════════════════════════════════════════════════════

set -e
BLUE='\033[1;34m'
GOLD='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
CYAN='\033[1;36m'
DIM='\033[2m'
WHITE='\033[1;37m'
PURPLE='\033[1;35m'
RESET='\033[0m'

MODEL="${MODEL:-gemini-2.0-flash}"

banner() {
  echo ""
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${GOLD}  DEMO $1: $2${RESET}"
  echo -e "${DIM}  $3${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo ""
}

case_card() {
  echo -e "${WHITE}  ┌─ THE CASE ──────────────────────────────────────────────┐${RESET}"
  echo -e "${WHITE}  │${RESET}"
}

case_field() {
  # $1 = label, $2 = value
  printf "  ${WHITE}│${RESET}  ${CYAN}%-14s${RESET} %s\n" "$1" "$2"
}

case_bullet() {
  echo -e "  ${WHITE}│${RESET}    • $1"
}

case_end() {
  echo -e "${WHITE}  │${RESET}"
  echo -e "${WHITE}  └────────────────────────────────────────────────────────┘${RESET}"
  echo ""
}

pause() {
  echo ""
  echo -e "${DIM}  Press ENTER to run (or Ctrl+C to stop)...${RESET}"
  read -r
}

# ───────────────────────────────────────────────────────────────────
# Pre-flight
# ───────────────────────────────────────────────────────────────────
if [ -z "$GOOGLE_API_KEY" ]; then
  echo -e "${RED}ERROR: Set GOOGLE_API_KEY before running.${RESET}"
  echo "  export GOOGLE_API_KEY=your_key"
  exit 1
fi

echo -e "${GOLD}"
echo "   ╔══════════════════════════════════════════════════════════╗"
echo "   ║             COGNITIVE CORE — LIVE DEMO                  ║"
echo "   ║   Seven Primitives · Three Layers · Two Modes           ║"
echo "   ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  Model: ${BLUE}$MODEL${RESET}"
echo -e "  Override: ${DIM}MODEL=gemini-2.5-pro ./demo.sh${RESET}"
echo ""

# ═══════════════════════════════════════════════════════════════════
# DEMO 1: Card Dispute — Sequential
# ═══════════════════════════════════════════════════════════════════
banner "1/7" "Card Dispute — Sequential" \
  "Retrieve → Classify → Verify → Investigate → Generate → Challenge"

case_card
case_field "Headline:" "Clear Fraud — Unauthorized Amazon Purchase"
case_field "Member:" "Sarah Chen, 8-year member, excellent standing"
case_field "Transaction:" "\$847.23 at Amazon.com on 2026-01-15"
case_field "Fraud score:" "0.92 — strong unauthorized signal"
case_field "Card status:" "In member's possession (not lost/stolen)"
case_field "Device:" "Fingerprint doesn't match any known device"
echo -e "  ${WHITE}│${RESET}"
case_field "Regulatory:" "Reg E (EFT), Reg Z (credit cards)"
case_field "Expected:" "Fast-path → provisional credit → member letter"
case_end

echo -e "${DIM}  WHAT TO WATCH:${RESET}"
echo "  • High confidence classify → skips full investigation (fast path)"
echo "  • Generate produces the actual member response letter"
echo "  • Challenge checks Reg E/Z compliance on the letter"
echo ""

pause

python -m engine.runner \
  -w workflows/dispute_resolution.yaml \
  -d domains/card_dispute.yaml \
  -c cases/card_clear_fraud.json \
  -m "$MODEL"

echo ""
echo -e "${DIM}  Press ENTER to continue...${RESET}"
read -r

# ═══════════════════════════════════════════════════════════════════
# DEMO 2: Loan Hardship — Sequential
# ═══════════════════════════════════════════════════════════════════
banner "2/7" "Loan Hardship — Sequential" \
  "Classify → Investigate (branch) → Classify → Generate → Challenge"

case_card
case_field "Headline:" "Military Spouse — Medical Retirement Hardship"
case_field "Member:" "Angela Reeves, member since 2018"
case_field "Situation:" "Husband medically retiring from Navy after 12 years"
case_field "Income:" "Dual \$142K → single ~\$65K (55% drop)"
case_field "Mortgage:" "\$287K balance, \$2,847/mo — 38% of current income"
case_field "Auto loan:" "\$18.2K balance, \$487/mo"
case_field "Payments:" "No missed payments — proactively seeking help"
echo -e "  ${WHITE}│${RESET}"
case_field "Regulatory:" "SCRA, MLA, ECOA, UDAAP"
case_field "Expected:" "Military classify → SCRA investigation → guidance"
case_end

echo -e "${DIM}  WHAT TO WATCH:${RESET}"
echo "  • Classify routes to military-specific investigation (not financial)"
echo "  • Domain injects SCRA/MLA expertise via \${domain.*} references"
echo "  • Challenge reviews for regulatory compliance gaps"
echo ""

pause

python -m engine.runner \
  -w workflows/loan_hardship.yaml \
  -d domains/military_hardship.yaml \
  -c cases/military_hardship_reeves.json \
  -m "$MODEL"

echo ""
echo -e "${DIM}  Press ENTER to continue...${RESET}"
read -r

# ═══════════════════════════════════════════════════════════════════
# DEMO 3: Member Complaint — Sequential
# ═══════════════════════════════════════════════════════════════════
banner "3/7" "Member Complaint — Sequential" \
  "Classify ×2 → Investigate → Generate → Challenge"

case_card
case_field "Headline:" "Service Failure — Inconsistent Hold Information"
case_field "Member:" "Michael Torres, 6yr member, \$45K relationship"
case_field "Complaint:" ""
echo -e "  ${WHITE}│${RESET}    ${RED}\"I called three times last week about a hold on my paycheck"
echo -e "  ${WHITE}│${RESET}    deposit and each time I was told something different. The first"
echo -e "  ${WHITE}│${RESET}    person said 24 hours, the second said 3-5 business days, and"
echo -e "  ${WHITE}│${RESET}    the third said they couldn't see any hold at all. Meanwhile I"
echo -e "  ${WHITE}│${RESET}    bounced two payments and got charged \$50 in fees.\"${RESET}"
echo -e "  ${WHITE}│${RESET}"
case_field "Harm:" "2 bounced payments (\$325.43), \$50 NSF fees"
case_field "Risk:" "Attrition score 0.6, says 'considering closing accounts'"
case_field "Value:" "\$45K across checking, savings, auto + direct deposit"
echo -e "  ${WHITE}│${RESET}"
case_field "Regulatory:" "UDAAP (unfair/deceptive acts)"
case_field "Expected:" "Classify type+severity → root cause → response + fee reversal"
case_end

echo -e "${DIM}  WHAT TO WATCH:${RESET}"
echo "  • Two classifications: complaint TYPE then SEVERITY"
echo "  • Investigation quantifies financial harm and traces the timeline"
echo "  • Generate must include fee reversal amount and direct contact"
echo "  • Challenge from member, compliance, AND retention perspectives"
echo ""

pause

python -m engine.runner \
  -w workflows/complaint_resolution.yaml \
  -d domains/member_complaint.yaml \
  -c cases/complaint_torres.json \
  -m "$MODEL"

echo ""
echo -e "${DIM}  Press ENTER to continue...${RESET}"
read -r

# ═══════════════════════════════════════════════════════════════════
# DEMO 4: Spending Advisor — Agentic
# ═══════════════════════════════════════════════════════════════════
banner "4/7" "Spending Advisor — Agentic" \
  "LLM orchestrator decides path at runtime"

case_card
case_field "Headline:" "Personal Finance Check-In"
case_field "Member:" "David Williams, 34, \$72K income"
case_field "Question:" "\"How's my spending been?\""
case_field "Data:" "12 months of debit card transactions"
case_field "Goals:" "Emergency fund (\$15K target), vacation fund (\$3K)"
echo -e "  ${WHITE}│${RESET}"
case_field "Mode:" "${PURPLE}AGENTIC${RESET} — orchestrator decides path"
case_field "Expected:" "Self-correction if Challenge catches math errors"
case_end

echo -e "${DIM}  WHAT TO WATCH:${RESET}"
echo "  • Orchestrator chooses primitives dynamically"
echo "  • generate → challenge loop: Pro model catches errors Flash makes"
echo "  • If challenge fails, watch whether orchestrator reinvestigates or retries"
echo "  • Per-step model: Flash generates, Pro challenges"
echo ""

pause

python -m engine.runner \
  -w workflows/spending_advisor_agentic.yaml \
  -d domains/debit_spending_agentic.yaml \
  -c cases/spending_advisor_williams.json \
  -m "$MODEL"

echo ""
echo -e "${DIM}  Press ENTER to continue...${RESET}"
read -r

# ═══════════════════════════════════════════════════════════════════
# DEMO 5: Military Hardship — Agentic with Think
# ═══════════════════════════════════════════════════════════════════
banner "5/7" "Military Hardship — Agentic + Think" \
  "Think changes the routing: member letter → specialist escalation"

case_card
case_field "Headline:" "Same Case as Demo 2 — But Agentic + Think"
case_field "Member:" "Angela Reeves (same military spouse)"
case_field "Difference:" "Orchestrator has Think primitive available"
echo -e "  ${WHITE}│${RESET}"
case_field "Mode:" "${PURPLE}AGENTIC${RESET} — same 7 primitives, different path"
case_field "Expected:" "Dual investigation → Think synthesis → ESCALATION"
echo -e "  ${WHITE}│${RESET}"
echo -e "  ${WHITE}│${RESET}  ${GOLD}COMPARE TO DEMO 2:${RESET}"
echo -e "  ${WHITE}│${RESET}  Sequential → member letter (may have gaps)"
echo -e "  ${WHITE}│${RESET}  Agentic+Think → escalation brief (catches unknowns)"
case_end

echo -e "${DIM}  WHAT TO WATCH:${RESET}"
echo "  • TWO investigations: military protections AND financial situation"
echo "  • Think synthesizes: 'too many unknowns for automated letter'"
echo "  • Routes to escalation brief INSTEAD of member letter"
echo "  • Challenge passes with zero vulnerabilities (safer outcome)"
echo ""

pause

python -m engine.runner \
  -w workflows/loan_hardship_agentic.yaml \
  -d domains/military_hardship_agentic.yaml \
  -c cases/military_hardship_reeves.json \
  -m "$MODEL"

echo ""
echo -e "${DIM}  Press ENTER to continue...${RESET}"
read -r

# ═══════════════════════════════════════════════════════════════════
# DEMO 6: Member Complaint — Agentic
# ═══════════════════════════════════════════════════════════════════
banner "6/7" "Member Complaint — Agentic" \
  "Orchestrator weighs escalation vs direct resolution"

case_card
case_field "Headline:" "Same Complaint as Demo 3 — Agentic Mode"
case_field "Member:" "Michael Torres (same \$45K member)"
case_field "Difference:" "Think reasons about retention math"
echo -e "  ${WHITE}│${RESET}"
case_field "Mode:" "${PURPLE}AGENTIC${RESET} — may Think before deciding"
case_field "Question:" "Is \$50 in fees worth losing a \$45K relationship?"
echo -e "  ${WHITE}│${RESET}"
echo -e "  ${WHITE}│${RESET}  ${GOLD}COMPARE TO DEMO 3:${RESET}"
echo -e "  ${WHITE}│${RESET}  Sequential → fixed path (classify → investigate → generate)"
echo -e "  ${WHITE}│${RESET}  Agentic → orchestrator adapts based on findings"
case_end

echo -e "${DIM}  WHAT TO WATCH:${RESET}"
echo "  • May use Think to weigh escalation vs direct resolution"
echo "  • Retention calculation: \$45K relationship, direct deposit, 6 years"
echo "  • Watch whether it produces a member response or escalation brief"
echo ""

pause

python -m engine.runner \
  -w workflows/complaint_resolution_agentic.yaml \
  -d domains/member_complaint_agentic.yaml \
  -c cases/complaint_torres.json \
  -m "$MODEL"

echo ""
echo -e "${DIM}  Press ENTER to continue...${RESET}"
read -r

# ═══════════════════════════════════════════════════════════════════
# DEMO 7: Configuration Validation
# ═══════════════════════════════════════════════════════════════════
banner "7/7" "Configuration Validation" \
  "New use case = YAML config, not code"

echo -e "${DIM}  The remaining workflows — all pure YAML, no Python:${RESET}"
echo ""

echo -e "${BLUE}  Nurse Triage (sequential) — cardiac chest pain at 2am:${RESET}"
python -m engine.runner \
  -w workflows/nurse_triage.yaml \
  -d domains/cardiac_triage.yaml \
  -c cases/cardiac_chest_pain.json \
  --validate-only

echo ""
echo -e "${BLUE}  SAR Investigation (sequential) — structuring alert:${RESET}"
python -m engine.runner \
  -w workflows/sar_investigation.yaml \
  -d domains/structuring_sar.yaml \
  -c cases/sar_structuring.json \
  --validate-only

echo ""
echo -e "${BLUE}  Regulatory Impact (agentic) — AVM rule:${RESET}"
python -m engine.runner \
  -w workflows/regulatory_impact_agentic.yaml \
  -d domains/avm_regulation_agentic.yaml \
  -c cases/avm_regulation.json \
  --validate-only

echo ""
echo -e "${GOLD}"
echo "   ╔══════════════════════════════════════════════════════════╗"
echo "   ║                    DEMO COMPLETE                        ║"
echo "   ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  7 demos. Same 7 primitives. 3 layers. 2 modes."
echo ""
echo "  13 workflows (7 sequential + 6 agentic)"
echo "  14 domains (8 sequential + 6 agentic)"
echo "  9 test cases — 0 lines of Python changed to add a new use case"
echo ""
echo -e "  ${DIM}To capture detailed traces for a presentation deck:${RESET}"
echo -e "  ${CYAN}python capture_traces.py -o traces.json${RESET}"
echo ""
