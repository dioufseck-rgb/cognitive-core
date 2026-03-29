FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY . .

# Runtime environment
ENV PORT=8080
ENV LLM_PROVIDER=google
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Single worker — coordinator is thread-safe via SQLiteBackend RLock,
# but multi-worker requires validated Cloud SQL connection pooling.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
