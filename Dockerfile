# Cognitive Core — Unified Container
#
# Serves both entry points from a single image:
#   Port 8000 — Cognitive Core API (direct access, CLI, internal tools)
#   Port 8088 — Foundry Responses API (Foundry catalog, Teams, Copilot)
#
# Build:
#   docker build -t cognitive-core .
#
# Run locally:
#   docker run -p 8000:8000 -p 8088:8088 \
#     -e LLM_PROVIDER=azure_foundry \
#     -e AZURE_AI_PROJECT_ENDPOINT=https://... \
#     cognitive-core
#
# Run single entry point:
#   docker run -p 8088:8088 -e CC_ENTRY=foundry cognitive-core
#   docker run -p 8000:8000 -e CC_ENTRY=api cognitive-core

FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — core
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Python deps — API + Foundry + Azure
RUN pip install --no-cache-dir \
    fastapi==0.115.* \
    uvicorn[standard]==0.32.* \
    arq==0.26.* \
    redis==5.* \
    azure-identity>=1.17.0 \
    azure-ai-projects>=1.0.0 \
    langchain-openai>=0.3.0

# Application code
COPY engine/ engine/
COPY coordinator/ coordinator/
COPY registry/ registry/
COPY fixtures/ fixtures/
COPY mcp_servers/ mcp_servers/
COPY workflows/ workflows/
COPY domains/ domains/
COPY cases/ cases/
COPY evals/ evals/
COPY api/ api/
COPY scripts/ scripts/
COPY config/ config/
COPY llm_config.yaml .
COPY docker-entrypoint.sh .
COPY README.md .

# Non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check (API port)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8088/health || exit 1

EXPOSE 8000 8088

ENV CC_PROJECT_ROOT=/app
ENV CC_WORKER_MODE=thread
ENV PYTHONUNBUFFERED=1

# CC_ENTRY: "both" (default), "api", "foundry"
ENV CC_ENTRY=both

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
