# ═══════════════════════════════════════════════════════════════════════
# Cognitive Core — Container Image
# ═══════════════════════════════════════════════════════════════════════
#
# Build:  docker build -t cognitive-core .
# Run:    docker run --env-file .env cognitive-core
# Test:   docker run cognitive-core python run_batch_test.py --case simple --n 1
# Demo:   docker run cognitive-core python demo_insurance_claim.py
#
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS base

# System deps for psycopg2 (Postgres) if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Default: run the batch test smoke check
CMD ["python", "run_batch_test.py", "--case", "simple", "--n", "1"]
