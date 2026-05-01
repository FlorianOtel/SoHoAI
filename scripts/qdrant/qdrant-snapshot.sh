#!/usr/bin/env bash
# Take a Qdrant snapshot for the documents collection and clean up old ones.
# Snapshots are written to NFS for DR recovery.
# Usage: bash scripts/qdrant/qdrant-snapshot.sh [--keep N]  (default: keep 3)
set -euo pipefail

QDRANT_URL="http://192.168.1.93:6333"
COLLECTION="documents"
SNAPSHOTS_NFS_DIR="/mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/${COLLECTION}"
KEEP=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Creating snapshot for '${COLLECTION}'..."
RESPONSE=$(curl -sf -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots")
NAME=$(python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])" <<< "$RESPONSE")
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Created: ${NAME}"

# Keep only the last KEEP snapshots; delete older ones via the API.
SNAPSHOT_LIST=$(curl -sf "${QDRANT_URL}/collections/${COLLECTION}/snapshots")
python3 - "$KEEP" "$QDRANT_URL" "$COLLECTION" <<'EOF'
import sys, json, urllib.request

keep = int(sys.argv[1])
url  = sys.argv[2]
col  = sys.argv[3]

with urllib.request.urlopen(f"{url}/collections/{col}/snapshots") as r:
    snaps = json.load(r)["result"]

# Sort oldest-first by creation_time
snaps.sort(key=lambda s: s.get("creation_time", ""))
to_delete = snaps[:-keep] if len(snaps) > keep else []

for s in to_delete:
    name = s["name"]
    req = urllib.request.Request(
        f"{url}/collections/{col}/snapshots/{name}",
        method="DELETE",
    )
    with urllib.request.urlopen(req) as r:
        json.load(r)
    print(f"Deleted old snapshot: {name}")

print(f"Snapshots kept: {max(len(snaps) - len(to_delete), 0)}/{keep}")
EOF
