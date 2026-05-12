#!/usr/bin/env bash
# SoHoAI RAG ingestion wrapper.
# Run by systemd rag-ingest.service or directly for manual/debug use.
# The daemon acquires the NFS lock (rag.ingest_lock in config.yaml) internally;
# if another daemon is already running anywhere, it exits with a clear message.
#
# Env overrides:
#   RAG_WORKERS      file-level parallelism (default 3, GPU embed; use 1 for CPU embed)
#   RAG_BATCH        chunk-level Ollama concurrency (default 20, GPU; use 5 for CPU)
#   RAG_LOGFILE      log file path (default: NAS logs dir, see below)
#
# GPU embed (Ollama on Server 2, RTX 5070): RAG_WORKERS=3 RAG_BATCH=20
# CPU embed (Ollama on Server 1, local):    RAG_WORKERS=1 RAG_BATCH=5
set -euo pipefail

VENV="$HOME/Gin-AI/.Gin-AI-python-3.12/bin/activate"
PROJECT="/mnt/nfs/Florian/Gin-AI/projects/SoHoAI"
LOGFILE="${RAG_LOGFILE:-/mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log}"
WORKERS="${RAG_WORKERS:-3}"
BATCH="${RAG_BATCH:-20}"

# Users to sync — add entries here to run rag_sync_nfs.py for additional users.
# Each user is synced sequentially before the ingestion daemon starts.
RAG_SYNC_USERS=("florian")

mkdir -p "$(dirname "$LOGFILE")"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOGFILE"; }

log "=== RAG ingestion starting  workers=$WORKERS  batch=$BATCH ==="

# shellcheck disable=SC1090
source "$VENV"
cd "$PROJECT"

for user in "${RAG_SYNC_USERS[@]}"; do
    log "--- rag_sync_nfs.py --user $user ---"
    python utils/rag_sync_nfs.py --user "$user" 2>&1 | tee -a "$LOGFILE"
done

log "--- rag_ingest_daemon.py ---"
# Daemon acquires the NFS lock (rag.ingest_lock in config.yaml) internally.
# If another daemon is running anywhere, it prints a message and exits cleanly.
# Do NOT tee here: --log-file writes directly to $LOGFILE; tee would double log lines.
# stderr goes to journald via StandardError=journal in the service unit.
python utils/rag_ingest_daemon.py \
    --workers "$WORKERS" --batch "$BATCH" --log-file "$LOGFILE"

log "--- SQLite WAL checkpoint ---"
DB_PATH="/mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db"
sqlite3 "$DB_PATH" "PRAGMA wal_checkpoint(TRUNCATE);" 2>&1 | tee -a "$LOGFILE" || \
  log "WARN: WAL checkpoint failed (non-fatal)"

log "--- sqlite-qdrant-snapshot.sh (keep 12) ---"
bash scripts/sqlite-qdrant-snapshot.sh --keep 12 2>&1 | tee -a "$LOGFILE"

log "=== RAG ingestion complete ==="
