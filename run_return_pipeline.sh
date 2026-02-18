#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Cognitive Core — 3-Workflow Product Return Pipeline
# ═══════════════════════════════════════════════════════════════════
#
# Runs a suspicious laptop return through the coordinator.
#
# What happens:
#   1. product_return/electronics_return executes (Retrieve→Classify→
#      Investigate→Think→Generate)
#   2. Investigation finds fraud flags → coordinator triggers:
#      a. fraud_review/return_fraud (BLOCKING — return waits)
#      b. vendor_notification/vendor_ops (FIRE-AND-FORGET — runs independently)
#   3. fraud_review completes (auto tier) → result cascades back →
#      product_return resumes at decide_resolution with fraud data
#   4. vendor_notification completes but gets governance-held (hold tier) →
#      task published to compliance_review queue → requires approval
#   5. Run `python -m coordinator.cli pending` to see the approval queue
#      then `python -m coordinator.cli approve <instance_id>` to release
#
# Prerequisites:
#   - LLM_PROVIDER set (google, azure, openai, or bedrock)
#   - Provider credentials set (GOOGLE_API_KEY, AZURE_OPENAI_API_KEY, etc.)
#   - pip install langchain-google-genai langgraph pydantic pyyaml  (or your provider's package)
#
# Usage:
#   LLM_PROVIDER=google ./run_return_pipeline.sh [--model default] [--verbose]
#
# ═══════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-default}"
DB="coordinator.db"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  PRODUCT RETURN PIPELINE — 3-Workflow Coordination Demo"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Provider: LLM_PROVIDER=${LLM_PROVIDER:-not set (defaults to azure)}"
echo "  Case:     Suspicious laptop return (Marcus Webb, \$1,249.99)"
echo "  Root:     product_return / electronics_return"
echo "  Blocking: fraud_review / return_fraud (if fraud flags found)"
echo "  F&F:      vendor_notification / vendor_ops (if item > \$500)"
echo "  DB:       $DB"
echo ""

# Clean previous run
rm -f "$DB"

echo "─── Phase 1: Run product_return/electronics_return ───────────────"
echo ""

python -m coordinator.cli \
  --db "$DB" \
  run \
  --workflow product_return \
  --domain electronics_return \
  --case cases/laptop_return_suspicious.json \
  --model "$MODEL" \
  --verbose

echo ""
echo "─── Phase 2: Check for pending approvals ─────────────────────────"
echo ""

python -m coordinator.cli --db "$DB" pending

echo ""
echo "─── Phase 3: Check stats ─────────────────────────────────────────"
echo ""

python -m coordinator.cli --db "$DB" stats

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  If vendor_notification is pending approval, run:"
echo ""
echo "    LLM_PROVIDER=google python -m coordinator.cli --db $DB pending"
echo "    LLM_PROVIDER=google python -m coordinator.cli --db $DB approve <instance_id> --approver 'Ops Manager'"
echo ""
echo "  To see the full correlation chain:"
echo ""
echo "    python -m coordinator.cli --db $DB ledger --verbose"
echo ""
