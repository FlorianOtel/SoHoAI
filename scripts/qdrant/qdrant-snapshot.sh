#!/usr/bin/env bash
# Take a Qdrant snapshot for the documents collection and save to NFS.
# Also copies rag_state.db alongside each snapshot so every Qdrant .snapshot
# has a matched rag_state-<timestamp>.db — a complete, self-contained restore pair.
# Snapshots are downloaded to NFS for DR recovery; local Qdrant copies are deleted.
# Usage: bash scripts/qdrant/qdrant-snapshot.sh [--keep N]  (default: keep 3)
set -euo pipefail

QDRANT_URL="http://192.168.1.93:6333"
COLLECTION="documents"
SNAPSHOTS_NFS_DIR="/mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/${COLLECTION}"
SQLITE_DB="/mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db"
KEEP=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Ensure NFS directory exists
mkdir -p "${SNAPSHOTS_NFS_DIR}"

# Checkpoint SQLite WAL first so rag_state.db is fully self-contained before
# the Qdrant snapshot is created — both reflect the same logical point in time.
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Checkpointing SQLite WAL..."
sqlite3 "${SQLITE_DB}" "PRAGMA wal_checkpoint(TRUNCATE);"

# Create snapshot
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Creating snapshot for '${COLLECTION}'..."
RESPONSE=$(curl -sf -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots")
NAME=$(python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])" <<< "$RESPONSE")
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Created: ${NAME}"

# Download snapshot to NFS
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Downloading to NFS..."
TEMP_FILE="${SNAPSHOTS_NFS_DIR}/${NAME}.tmp"
FINAL_FILE="${SNAPSHOTS_NFS_DIR}/${NAME}"

curl -sf --max-time 3600 \
    "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${NAME}" \
    -o "${TEMP_FILE}"

# Verify download is non-empty
FILE_SIZE=$(stat -f%z "${TEMP_FILE}" 2>/dev/null || stat -c%s "${TEMP_FILE}" 2>/dev/null || echo "0")
if [[ "${FILE_SIZE}" -le 0 ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ERROR: Downloaded snapshot is empty (${FILE_SIZE} bytes)" >&2
    rm -f "${TEMP_FILE}"
    exit 1
fi

mv "${TEMP_FILE}" "${FINAL_FILE}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Download complete ($(numfmt --to=iec-i --suffix=B "${FILE_SIZE}" 2>/dev/null || echo "${FILE_SIZE} bytes"))"

# Delete snapshot from Qdrant (local NVMe)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Deleting snapshot from Qdrant local storage..."
curl -sf -X DELETE "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${NAME}" > /dev/null
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Qdrant local copy deleted"

# Copy rag_state.db alongside the Qdrant snapshot with a matching timestamp.
# NAME is e.g. documents-8393594205839792-2026-05-12-01-00-01.snapshot
# Extract the timestamp suffix (everything after the shard-id segment).
SNAP_TS="${NAME#*-*-}"          # → 2026-05-12-01-00-01.snapshot
SNAP_TS="${SNAP_TS%.snapshot}"  # → 2026-05-12-01-00-01
SQLITE_DST="${SNAPSHOTS_NFS_DIR}/rag_state-${SNAP_TS}.db"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Copying rag_state.db → $(basename "${SQLITE_DST}")..."
cp "${SQLITE_DB}" "${SQLITE_DST}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  SQLite copy complete ($(numfmt --to=iec-i --suffix=B "$(stat -c%s "${SQLITE_DST}" 2>/dev/null || echo 0)" 2>/dev/null || echo "?"))"

# Rotate NFS snapshots: keep only the most recent KEEP files
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Rotating snapshots (keeping ${KEEP} most recent)..."
SNAPSHOTS_COUNT=$(find "${SNAPSHOTS_NFS_DIR}" -maxdepth 1 -type f -name "*.snapshot" | wc -l)
if [[ "${SNAPSHOTS_COUNT}" -gt "${KEEP}" ]]; then
    EXCESS=$((SNAPSHOTS_COUNT - KEEP))
    # Sort by mtime (oldest first) and delete the excess
    find "${SNAPSHOTS_NFS_DIR}" -maxdepth 1 -type f -name "*.snapshot" -printf '%T@ %p\n' \
        | sort -n \
        | head -n "${EXCESS}" \
        | cut -d' ' -f2- \
        | while read -r old_file; do
            rm -f "${old_file}"
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Deleted old snapshot: $(basename "${old_file}")"
            # Delete the matched SQLite copy (same timestamp, .db extension)
            old_base="$(basename "${old_file}" .snapshot)"  # documents-...-2026-05-01-01-00-01
            old_ts="${old_base#*-*-}"                       # 2026-05-01-01-00-01
            old_sqlite="${SNAPSHOTS_NFS_DIR}/rag_state-${old_ts}.db"
            rm -f "${old_sqlite}"
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Deleted old SQLite copy: rag_state-${old_ts}.db"
        done
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Snapshot pairs in NFS: $(find "${SNAPSHOTS_NFS_DIR}" -maxdepth 1 -type f -name "*.snapshot" | wc -l)/${KEEP}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Snapshot complete"
