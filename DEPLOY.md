# Cognitive Core — Deployment Guide

## Quick Reference

| Environment | Command | What it proves |
|---|---|---|
| Codespace (direct) | `python -m coordinator.cli run ...` | Engine + coordinator + LLM |
| Codespace (API) | `uvicorn api.server:app --port 8000` | Full API stack |
| Codespace (Foundry) | `uvicorn api.foundry_adapter:app --port 8088` | Foundry protocol |
| Docker (local) | `docker compose up` | Container packaging |
| Azure Container Apps | `az containerapp up` | Cloud deployment |
| Foundry Agent Service | `python scripts/register_foundry_agents.py` | Foundry catalog |

---

## 1. Codespace (Direct — No Docker)

### Prerequisites
```bash
# Python 3.12+
python --version

# Install deps
pip install -r requirements.txt
pip install langchain-openai azure-identity fastapi uvicorn
```

### Configure Azure AI Foundry
```bash
# Login to Azure
az login --use-device-code

# Create .env file
cat > .env << 'EOF'
LLM_PROVIDER=azure_foundry
AZURE_AI_PROJECT_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com/api/projects/YOUR-PROJECT
MODEL_DEPLOYMENT_NAME=gpt-4o-mini
EOF

# Source it
export $(cat .env | xargs)
```

### Update llm_config.yaml
Your `llm_config.yaml` already has `azure_foundry` as default provider. Verify:
```yaml
default_provider: azure_foundry
```

### Run the multi-agent test
```bash
# Single workflow
python -m coordinator.cli run \
  --workflow claim_intake \
  --domain synthetic_claim \
  --case cases/synthetic/sc_004_both_delegations.json

# It will suspend at gate tier. Approve:
python -m coordinator.cli approve <instance_id> --approver mamadou

# Check the full chain:
python -m coordinator.cli chain <correlation_id>
```

### Run the API
```bash
# Start API server
CC_WORKER_MODE=inline uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload &

# Submit a case
curl -X POST http://localhost:8000/v1/cases \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "claim_intake",
    "domain": "synthetic_claim",
    "case_input": {
      "case_id": "CLM-2026-00847",
      "get_claim": {"claim_id": "CLM-2026-00847", "amount": 12500,
                    "claim_type_hint": "physical_damage", "flags": ["high_value"]},
      "get_policy": {"policy_id": "POL-001", "status": "active",
                     "coverage_type": "comprehensive,collision",
                     "tenure_months": 36, "prior_claims": 1}
    }
  }'

# Check status
curl http://localhost:8000/v1/cases/<instance_id>

# Check audit trail
curl http://localhost:8000/v1/cases/<instance_id>/trail

# List pending approvals
curl http://localhost:8000/v1/approvals

# Approve
curl -X POST http://localhost:8000/v1/approvals/<instance_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"approver": "mamadou"}'
```

### Run the Foundry adapter
```bash
# Start Foundry adapter
WORKFLOW=claim_intake DOMAIN=synthetic_claim \
  uvicorn api.foundry_adapter:app --host 0.0.0.0 --port 8088 &

# Call it using Foundry Responses protocol
curl -X POST http://localhost:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{
        "role": "user",
        "content": "{\"claim_id\":\"CLM-001\",\"amount\":12500,\"claim_type_hint\":\"physical_damage\",\"flags\":[\"high_value\"],\"get_policy\":{\"status\":\"active\",\"coverage_type\":\"comprehensive,collision\",\"tenure_months\":36,\"prior_claims\":1}}"
      }]
    }
  }'

# Or with routing metadata
curl -X POST http://localhost:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Process this claim",
    "metadata": {
      "workflow": "damage_assessment",
      "domain": "synthetic_damage"
    }
  }'
```

### Run the tests
```bash
# Full suite
python -m unittest discover tests -v

# Just multi-agent + Foundry
python -m unittest tests.test_multi_agent tests.test_foundry_adapter -v
```

---

## 2. Docker (Local)

### Build
```bash
docker build -t cognitive-core .
```

### Run both ports
```bash
docker run -p 8000:8000 -p 8088:8088 \
  -e LLM_PROVIDER=azure_foundry \
  -e AZURE_AI_PROJECT_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com/api/projects/YOUR-PROJECT \
  -e MODEL_DEPLOYMENT_NAME=gpt-4o-mini \
  -e WORKFLOW=claim_intake \
  -e DOMAIN=synthetic_claim \
  cognitive-core
```

### Run Foundry adapter only (for hosted agent deployment)
```bash
docker run -p 8088:8088 \
  -e CC_ENTRY=foundry \
  -e LLM_PROVIDER=azure_foundry \
  -e AZURE_AI_PROJECT_ENDPOINT=https://... \
  -e WORKFLOW=claim_intake \
  -e DOMAIN=synthetic_claim \
  cognitive-core
```

### Docker Compose (with Redis worker)
```bash
# Create .env with your credentials
cp .env.example .env
# Edit .env

docker compose up --build
```

---

## 3. Azure Container Apps

### Prerequisites
```bash
az login
az extension add --name containerapp --upgrade
```

### Create resources
```bash
RESOURCE_GROUP=rg-cognitive-core
LOCATION=eastus
ACR_NAME=crcognitivecore
APP_NAME=cognitive-core

# Resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Container registry
az acr create --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME --sku Basic --admin-enabled true

# Build and push
az acr build --registry $ACR_NAME \
  --image cognitive-core:v1 \
  --file Dockerfile .

# Container Apps environment
az containerapp env create \
  --name cae-cognitive-core \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

### Deploy
```bash
# Get ACR credentials
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

# Deploy container app
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment cae-cognitive-core \
  --image $ACR_NAME.azurecr.io/cognitive-core:v1 \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8088 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1 --memory 2Gi \
  --env-vars \
    CC_ENTRY=foundry \
    LLM_PROVIDER=azure_foundry \
    AZURE_AI_PROJECT_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com/api/projects/YOUR-PROJECT \
    MODEL_DEPLOYMENT_NAME=gpt-4o-mini \
    WORKFLOW=claim_intake \
    DOMAIN=synthetic_claim

# Get the URL
az containerapp show --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" -o tsv
```

### Test
```bash
APP_URL=$(az containerapp show --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" -o tsv)

curl -X POST https://$APP_URL/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Process claim CLM-001"}'
```

### Deploy multiple agents (one container app per workflow)
```bash
# Damage assessment agent (auto tier, lighter resources)
az containerapp create \
  --name damage-assessment \
  --resource-group $RESOURCE_GROUP \
  --environment cae-cognitive-core \
  --image $ACR_NAME.azurecr.io/cognitive-core:v1 \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8088 \
  --ingress internal \
  --cpu 0.5 --memory 1Gi \
  --env-vars \
    CC_ENTRY=foundry \
    WORKFLOW=damage_assessment \
    DOMAIN=synthetic_damage \
    LLM_PROVIDER=azure_foundry \
    AZURE_AI_PROJECT_ENDPOINT=https://...
```

---

## 4. Foundry Agent Service (Hosted Agent)

### Prerequisites
```bash
pip install azure-ai-projects azure-identity

# Your image must be in ACR
az acr build --registry $ACR_NAME \
  --image cognitive-core:v1 \
  --file Dockerfile .
```

### Register agents
```bash
# Dry run first
python scripts/register_foundry_agents.py \
  --dry-run \
  --external-only

# Register orchestrator agents only
python scripts/register_foundry_agents.py \
  --image $ACR_NAME.azurecr.io/cognitive-core:v1 \
  --endpoint $AZURE_AI_PROJECT_ENDPOINT \
  --external-only

# Register all agents (including internal)
python scripts/register_foundry_agents.py \
  --image $ACR_NAME.azurecr.io/cognitive-core:v1 \
  --endpoint $AZURE_AI_PROJECT_ENDPOINT
```

### Grant ACR pull access
```bash
# Get project managed identity
PROJECT_IDENTITY=$(az cognitiveservices account show \
  --name YOUR-FOUNDRY-RESOURCE \
  --resource-group $RESOURCE_GROUP \
  --query "identity.principalId" -o tsv)

# Grant pull access
az role assignment create \
  --assignee $PROJECT_IDENTITY \
  --role "AcrPull" \
  --scope $(az acr show --name $ACR_NAME --query id -o tsv)
```

### Verify in Foundry portal
1. Go to ai.azure.com
2. Open your project
3. Select **Agents** in left nav
4. Your registered agents should appear
5. Open agent playground → test with a claim

---

## 5. Managed Identity (Production)

For production, replace API keys with managed identity:

```bash
# Enable system-assigned identity on Container App
az containerapp identity assign \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --system-assigned

# Grant Cognitive Services User role
APP_IDENTITY=$(az containerapp identity show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "principalId" -o tsv)

az role assignment create \
  --assignee $APP_IDENTITY \
  --role "Cognitive Services User" \
  --scope /subscriptions/YOUR-SUB/resourceGroups/$RESOURCE_GROUP
```

Then in your code, `DefaultAzureCredential()` picks up the managed identity automatically. No API keys needed.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `CC_ENTRY` | No | `both` | Entry point: `api`, `foundry`, or `both` |
| `CC_PROJECT_ROOT` | No | `.` | Path to project root |
| `CC_WORKER_MODE` | No | `thread` | Worker mode: `thread`, `inline`, `arq` |
| `LLM_PROVIDER` | Yes | `azure_foundry` | LLM provider |
| `AZURE_AI_PROJECT_ENDPOINT` | If Azure | — | Foundry project endpoint |
| `MODEL_DEPLOYMENT_NAME` | If Azure | — | Model deployment name |
| `WORKFLOW` | For Foundry | — | Default workflow for Foundry adapter |
| `DOMAIN` | For Foundry | — | Default domain for Foundry adapter |
| `REDIS_URL` | If arq mode | — | Redis connection for async worker |

---

## Validation Checklist

After each deployment, verify:

```bash
# Health
curl $URL/health

# Foundry protocol
curl -X POST $URL/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"health check"}'

# Full E2E (submit claim, approve, check chain)
# 1. Submit
RESP=$(curl -s -X POST $URL/responses \
  -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"user","content":"{\"claim_id\":\"CLM-TEST\",\"amount\":12500,\"claim_type_hint\":\"physical_damage\"}"}]}}')
echo $RESP | python -m json.tool

# 2. Check for requires_action (gate tier)
# 3. Approve via API port
# 4. Verify delegation chain
```

## Live Eval Harness

The eval harness (`scripts/eval_live.py`) runs workflows end-to-end against
a live LLM provider. It validates structural invariants and checks domain-specific
outcomes.

```bash
# Run all 20 synthetic cases
python scripts/eval_live.py --all --auto-approve

# Run one workflow (e.g., fraud screening — 4 cases)
python scripts/eval_live.py --workflow fraud_screening --auto-approve

# Run a single case with verbose output
python scripts/eval_live.py --case sc_001_simple_approve --auto-approve -v

# Strict mode: quality gates are hard failures
python scripts/eval_live.py --all --auto-approve --strict-gates
```

### What It Checks

| Check | Type | Gating? |
|-------|------|---------|
| I1: Denied claim never proceeds | Structural | Yes |
| I3: Artifact has required JSON keys | Structural | Yes |
| I4: Parse reliability | Structural | Yes |
| I7a: Risk ↔ finding coherence | Semantic | Advisory |
| I7b: Doc conformance ↔ artifact type | Structural | Yes |
| Domain contract validation | Semantic | Advisory |
| Escalation brief generation | Structural | Displayed |
