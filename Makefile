# ═══════════════════════════════════════════════════════════════════════
# Cognitive Core — Development Commands
# ═══════════════════════════════════════════════════════════════════════

.PHONY: install test demo smoke batch lint clean docker

# ── Setup ───────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff

# ── Run ─────────────────────────────────────────────────────────────

demo:
	python demo_insurance_claim.py

demo-live:
	python demo_live_coordinator.py

smoke:
	python smoke_test.py

# ── Test ────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

batch-simple:
	python run_batch_test.py --case simple --n 5

batch-medium:
	python run_batch_test.py --case medium --n 5

batch-hard:
	python run_batch_test.py --case hard --n 5

batch-all:
	python run_batch_test.py --case simple medium hard --n 5

# ── Validation ──────────────────────────────────────────────────────

lint:
	python -c "import py_compile, os; \
		[py_compile.compile(os.path.join(r,f), doraise=True) \
		 for r,d,fs in os.walk('.') if '__pycache__' not in r \
		 for f in fs if f.endswith('.py')]"
	@echo "All Python files compile OK"

validate-yaml:
	python -c "import yaml, json, os; \
		[yaml.safe_load(open(os.path.join(r,f))) \
		 for r,d,fs in os.walk('.') if '__pycache__' not in r \
		 for f in fs if f.endswith(('.yaml','.yml'))]; \
		[json.load(open(os.path.join(r,f))) \
		 for r,d,fs in os.walk('.') if '__pycache__' not in r \
		 for f in fs if f.endswith('.json')]; \
		print('All YAML/JSON files valid')"

check: lint validate-yaml
	@echo "All checks passed"

# ── MCP ─────────────────────────────────────────────────────────────

mcp-claims:
	python mcp_servers/claims_services.py

# ── Docker ──────────────────────────────────────────────────────────

docker:
	docker build -t cognitive-core .

docker-test:
	docker run --env-file .env cognitive-core python run_batch_test.py --case simple --n 3

docker-demo:
	docker run cognitive-core python demo_insurance_claim.py

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f coordinator.db
	rm -rf test_results/
