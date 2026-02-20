# Cognitive Core — Scaling & PostgreSQL Migration Guide

## Current Architecture: SQLite (Year 1)

SQLite in WAL mode is the default backend. It is:
- Zero-configuration (no external services)
- Single-file (easy backup, deployment)
- Sufficient for **single-replica** deployment

### Operating Limits

| Parameter | Safe Limit | Alarm Threshold |
|-----------|-----------|-----------------|
| API Replicas | 1 | N/A (startup fails at >1) |
| Worker Concurrency | 4 threads | N/A (startup fails at >4) |
| Cases/Day | 1,000 | 750 |
| Audit DB Size | 5 GB | 4 GB |
| Query p99 Latency | 500ms | 300ms |
| WAL File Size | 100 MB | 75 MB |

These limits are enforced at startup (H-019) and monitored at runtime (H-018).

---

## Migration Trigger Conditions

**Migrate to PostgreSQL when ANY of these conditions is sustained for 7+ days:**

1. **Horizontal scaling required**: Need >1 API replica for throughput
2. **Volume**: >1,000 cases/day sustained
3. **Storage**: Audit DB exceeds 5 GB
4. **Latency**: Query p99 > 500ms sustained 24 hours

The SQLite health monitor (H-018) will emit structured warning events when approaching these thresholds.

---

## Migration Path

### 1. Provision Azure Database for PostgreSQL Flexible Server

```bash
az postgres flexible-server create \
  --name cognitive-core-db \
  --resource-group rg-ai-engineering \
  --location eastus \
  --sku-name Standard_B1ms \
  --storage-size 32 \
  --version 16 \
  --admin-user ccadmin \
  --admin-password <secure-password>
```

### 2. Run Schema DDL

The Postgres schema is maintained in `engine/db.py` as:
- `COORDINATOR_SCHEMA_POSTGRES`
- `AUDIT_SCHEMA_POSTGRES`
- `CHECKPOINT_SCHEMA_POSTGRES`

These are **schema-compatible** with the SQLite versions. No column changes, no type changes beyond:
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`
- `REAL` → `DOUBLE PRECISION`

### 3. Switch Backend

```bash
# Environment variables
CC_DB_BACKEND=postgres
CC_DB_DSN=postgresql://ccadmin:<password>@cognitive-core-db.postgres.database.azure.com:5432/cognitive_core

# Or via Azure App Configuration
az appconfig kv set --name cc-app-config --key CC_DB_BACKEND --value postgres
az appconfig kv set --name cc-app-config --key CC_DB_DSN --value "postgresql://..."
```

No code changes required. The `create_backend()` factory in `engine/db.py` handles the switch.

### 4. Install Python Driver

```bash
pip install psycopg[binary]  # Preferred (psycopg v3)
# or
pip install psycopg2-binary  # Alternative
```

### 5. Backfill Historical Data (Optional)

```bash
# Export from SQLite
sqlite3 coordinator.db ".dump instances" > instances.sql
sqlite3 coordinator.db ".dump audit_events" > audit.sql

# Import to PostgreSQL (with minor SQL adjustments)
psql $CC_DB_DSN -f instances.sql
psql $CC_DB_DSN -f audit.sql
```

### 6. Remove Startup Limits

Once on PostgreSQL, remove the SQLite scaling guards:
- `sqlite.max_replicas` → no longer enforced
- `sqlite.max_worker_concurrency` → increase to match CPU cores
- Scale API replicas as needed

---

## Estimated Effort

| Task | Effort |
|------|--------|
| Provision Flexible Server | 1 hour |
| Run schema DDL | 15 minutes |
| Switch config + install driver | 30 minutes |
| Backfill data (optional) | 1-2 hours |
| Validation testing | 2-4 hours |
| **Total** | **~1 day** |

---

## Connection Pooling (Production)

For production PostgreSQL, configure connection pooling:

```python
# In engine/db.py PostgresBackend
# pool_min and pool_max are constructor parameters
db = create_backend("postgres", dsn="...", pool_min=2, pool_max=10)
```

Consider PgBouncer or Azure's built-in connection pooling for >10 replicas.
