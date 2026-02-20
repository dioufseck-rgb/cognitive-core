# Cognitive Core — Makefile
#
# Usage:
#   make setup       — install deps, copy .env
#   make test        — run full test suite
#   make run         — start both API (8000) + Foundry adapter (8088)
#   make run-api     — start API only
#   make run-foundry — start Foundry adapter only
#   make claim       — submit + approve a test claim E2E
#   make docker      — build and run in Docker
#   make deploy      — deploy to Azure Container Apps
#   make register    — register agents in Foundry

.PHONY: setup test run run-api run-foundry claim docker deploy register clean

# ── Setup ────────────────────────────────────────────────

setup:
	pip install -r requirements.txt
	pip install langchain-openai azure-identity azure-ai-projects fastapi "uvicorn[standard]"
	@test -f .env || (cp .env.example .env && echo "Created .env — edit with your Azure credentials")

# ── Live LLM Evaluation ──────────────────────────────────

eval:
	python scripts/eval_live.py --auto-approve

eval-all:
	python scripts/eval_live.py --all --auto-approve -v

eval-claim:
	python scripts/eval_live.py --workflow claim_intake --auto-approve -v

eval-case:
	python scripts/eval_live.py --case $(CASE) --auto-approve -v

# ── Test ─────────────────────────────────────────────────

test:
	python -m unittest discover tests -v

test-multi-agent:
	python -m unittest tests.test_multi_agent tests.test_foundry_adapter -v

test-fast:
	python -m unittest tests.test_multi_agent tests.test_foundry_adapter tests.test_integration -v

# ── Run (Codespace) ──────────────────────────────────────

run:
	@echo "Starting API on :8000 and Foundry adapter on :8088"
	CC_WORKER_MODE=inline uvicorn api.server:app --host 0.0.0.0 --port 8000 & \
	uvicorn api.foundry_adapter:app --host 0.0.0.0 --port 8088

run-api:
	CC_WORKER_MODE=inline uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

run-foundry:
	uvicorn api.foundry_adapter:app --host 0.0.0.0 --port 8088 --reload

# ── E2E Claim Test ───────────────────────────────────────

claim:
	@echo "═══ Submitting claim via Foundry adapter ═══"
	@curl -s -X POST http://localhost:8088/responses \
	  -H "Content-Type: application/json" \
	  -d '{"input":{"messages":[{"role":"user","content":"{\"claim_id\":\"CLM-TEST\",\"amount\":12500,\"claim_type_hint\":\"physical_damage\",\"flags\":[\"high_value\"],\"get_policy\":{\"status\":\"active\",\"coverage_type\":\"comprehensive,collision\",\"tenure_months\":36,\"prior_claims\":1}}"}]}}' \
	  | python -m json.tool

claim-cli:
	python -m coordinator.cli run \
	  --workflow claim_intake \
	  --domain synthetic_claim \
	  --case cases/synthetic/sc_004_both_delegations.json

# ── Docker ───────────────────────────────────────────────

docker:
	docker build -t cognitive-core .
	docker run -p 8000:8000 -p 8088:8088 --env-file .env cognitive-core

docker-foundry:
	docker build -t cognitive-core .
	docker run -p 8088:8088 --env-file .env -e CC_ENTRY=foundry cognitive-core

# ── Azure Container Apps ─────────────────────────────────

RESOURCE_GROUP ?= rg-cognitive-core
LOCATION ?= eastus
ACR_NAME ?= crcognitivecore
APP_NAME ?= cognitive-core

deploy-infra:
	az group create --name $(RESOURCE_GROUP) --location $(LOCATION)
	az acr create --resource-group $(RESOURCE_GROUP) --name $(ACR_NAME) --sku Basic --admin-enabled true
	az containerapp env create --name cae-cognitive-core --resource-group $(RESOURCE_GROUP) --location $(LOCATION)

deploy-build:
	az acr build --registry $(ACR_NAME) --image cognitive-core:v1 --file Dockerfile .

deploy-app:
	$(eval ACR_PASSWORD := $(shell az acr credential show --name $(ACR_NAME) --query "passwords[0].value" -o tsv))
	az containerapp create \
	  --name $(APP_NAME) \
	  --resource-group $(RESOURCE_GROUP) \
	  --environment cae-cognitive-core \
	  --image $(ACR_NAME).azurecr.io/cognitive-core:v1 \
	  --registry-server $(ACR_NAME).azurecr.io \
	  --registry-username $(ACR_NAME) \
	  --registry-password $(ACR_PASSWORD) \
	  --target-port 8088 \
	  --ingress external \
	  --min-replicas 1 --max-replicas 3 \
	  --cpu 1 --memory 2Gi \
	  --env-vars \
	    CC_ENTRY=foundry \
	    LLM_PROVIDER=azure_foundry \
	    AZURE_AI_PROJECT_ENDPOINT=$(AZURE_AI_PROJECT_ENDPOINT) \
	    MODEL_DEPLOYMENT_NAME=$(MODEL_DEPLOYMENT_NAME) \
	    WORKFLOW=claim_intake \
	    DOMAIN=synthetic_claim

deploy: deploy-infra deploy-build deploy-app
	@echo "Deployed. URL:"
	@az containerapp show --name $(APP_NAME) --resource-group $(RESOURCE_GROUP) --query "properties.configuration.ingress.fqdn" -o tsv

# ── Foundry Registration ─────────────────────────────────

register-dry:
	python scripts/register_foundry_agents.py --dry-run --external-only

register:
	python scripts/register_foundry_agents.py \
	  --image $(ACR_NAME).azurecr.io/cognitive-core:v1 \
	  --endpoint $(AZURE_AI_PROJECT_ENDPOINT) \
	  --external-only

register-all:
	python scripts/register_foundry_agents.py \
	  --image $(ACR_NAME).azurecr.io/cognitive-core:v1 \
	  --endpoint $(AZURE_AI_PROJECT_ENDPOINT)

# ── Cleanup ──────────────────────────────────────────────

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.db" -delete 2>/dev/null; true

teardown:
	az group delete --name $(RESOURCE_GROUP) --yes --no-wait
