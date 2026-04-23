#!/usr/bin/env python3
"""
Show RAG ingestion queue status and Qdrant collection stats.

Usage (run from project root):
    python utils/rag_status.py
    python utils/rag_status.py --user florian
    python utils/rag_status.py --ignored                 # detailed listing: ignored files + rationale
    python utils/rag_status.py --watch /tmp/ingest.log   # live monitor
    python utils/rag_status.py --list-pending            # print every pending file path (pipeable)
    python utils/rag_status.py --list-pending 50         # limit to first 50
    python utils/rag_status.py --list-pending --user florian
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime as _DT
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_engine.collection import DOCUMENTS_COLLECTION, get_client
from rag_engine.schema import FIELD_OWNER
from rag_engine.state import StateDB


# ---------------------------------------------------------------------------
# Log parsing patterns for --watch mode
# ---------------------------------------------------------------------------

_RE_TIMESTAMP = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)')


def _parse_ts(line: str) -> float | None:
    """Return Unix timestamp from the log-line prefix, or None."""
    m = _RE_TIMESTAMP.match(line)
    if not m:
        return None
    try:
        return _DT.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f').timestamp()
    except ValueError:
        return None


# "Starting ingest: 507 files pending"
_RE_START = re.compile(r'Starting ingest: (\d+) files pending')

# "Processing [270 done / 10 failed]: /mnt/nfs/Florian/.../stockdata2.csv"
_RE_PROCESSING = re.compile(r'Processing \[(\d+) done / (\d+) failed\]: (.+)')

# "Chunked CFM Radar - 2022 Annual Report.pdf → 33 chunks  strategy=flat  file_type=pdf"
# Group 1: filename, 2: chunk count, 3: strategy, 4: file_type
_RE_CHUNKED = re.compile(r'Chunked (.+?) → (\d+) chunks\s+strategy=(\S+)\s+file_type=(\S+)')

# "Ingested CFM Radar - 2022 Annual Report.pdf: 33 points  owner=florian  tag=—"
# Group 1: filename, 2: point count
_RE_INGESTED = re.compile(r'Ingested (.+?): (\d+) points')

# "Embedding progress: 50/6806  stockdata2.csv"
# Group 1: done, 2: total, 3: filename (trailing whitespace stripped by caller)
_RE_EMBED_PROG = re.compile(r'Embedding progress: (\d+)/(\d+)\s+(.+)')


def parse_log(log_path: str) -> dict:
    """
    Single-pass scan of a rag_ingest_daemon log file.

    Tracks all in-flight files simultaneously (required for --workers > 1).
    With multiple workers, Processing/Chunked/Embedding lines from different
    files interleave in the log. The old single-state machine would reset on
    each new Processing line, losing progress for any file whose Chunked line
    appeared before a later Processing line.

    Returns state for the most interesting in-flight file:
      - prefers the file currently in 'embedding' phase with the most recent
        progress timestamp (i.e. the file the GPU is actively working on)
      - falls back to the last 'parsing' file, then the last 'done' file

    Keys in the returned dict (same interface as before):
      files_done        int|None
      files_failed      int|None
      current_file_path str|None
      total_chunks      int|None
      chunks_embedded   int
      initial_pending   int|None
      phase             str        -- idle | parsing | embedding | done
      file_type         str|None
      strategy          str|None
      embed_start_ts    float|None
      last_progress_ts  float|None
    """
    # Per-file state keyed by full NFS path
    files: dict[str, dict] = {}
    # basename → full path (for associating Chunked/Embedded/Ingested with Processing)
    name_to_path: dict[str, str] = {}
    initial_pending: int | None = None

    with open(log_path, 'r', errors='replace') as fh:
        for line in fh:

            m = _RE_START.search(line)
            if m:
                initial_pending = int(m.group(1))
                continue

            m = _RE_PROCESSING.search(line)
            if m:
                full_path = m.group(3).strip()
                fname = Path(full_path).name
                name_to_path[fname] = full_path
                files[full_path] = {
                    'files_done':        int(m.group(1)),
                    'files_failed':      int(m.group(2)),
                    'current_file_path': full_path,
                    'total_chunks':      None,
                    'chunks_embedded':   0,
                    '_last_embed_prog':  None,
                    'phase':             'parsing',
                    'file_type':         None,
                    'strategy':          None,
                    'embed_start_ts':    None,
                    'last_progress_ts':  None,
                }
                continue

            m = _RE_CHUNKED.search(line)
            if m:
                full_path = name_to_path.get(m.group(1))
                if full_path and full_path in files:
                    files[full_path].update({
                        'total_chunks':   int(m.group(2)),
                        'strategy':       m.group(3),
                        'file_type':      m.group(4),
                        'phase':          'embedding',
                        'embed_start_ts': _parse_ts(line),
                    })
                continue

            m = _RE_INGESTED.search(line)
            if m:
                full_path = name_to_path.get(m.group(1))
                if full_path and full_path in files:
                    fs = files[full_path]
                    fs['chunks_embedded'] = fs['total_chunks'] or int(m.group(2))
                    fs['phase'] = 'done'
                continue

            m = _RE_EMBED_PROG.search(line)
            if m:
                full_path = name_to_path.get(m.group(3).strip())
                if full_path and full_path in files:
                    files[full_path]['_last_embed_prog'] = (int(m.group(1)), int(m.group(2)))
                    files[full_path]['last_progress_ts'] = _parse_ts(line)
                continue

    # Resolve _last_embed_prog for every tracked file
    for fs in files.values():
        lp = fs.pop('_last_embed_prog')
        if lp is not None and fs['phase'] == 'embedding':
            fs['chunks_embedded'] = lp[0]

    # Pick the most interesting file to display:
    #   1. embedding phase — most recently active (highest last_progress_ts)
    #   2. parsing phase   — most recently started (last in dict order)
    #   3. done phase      — most recently completed (last in dict order)
    #   4. nothing seen    — return idle sentinel
    if not files:
        return {
            'files_done': None, 'files_failed': None, 'current_file_path': None,
            'total_chunks': None, 'chunks_embedded': 0, 'initial_pending': initial_pending,
            'phase': 'idle', 'file_type': None, 'strategy': None,
            'embed_start_ts': None, 'last_progress_ts': None,
        }

    embedding = [fs for fs in files.values() if fs['phase'] == 'embedding']
    if embedding:
        best = max(embedding, key=lambda fs: fs.get('last_progress_ts') or 0.0)
    else:
        # Prefer parsing over done; within each phase take last-seen (dict insertion order)
        parsing = [fs for fs in files.values() if fs['phase'] == 'parsing']
        best = parsing[-1] if parsing else list(files.values())[-1]

    best['initial_pending'] = initial_pending
    return best


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _clear() -> None:
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def _fmt_duration(secs: float) -> str:
    """Format a duration in seconds as '1h 23m 45s' or '23m 45s'."""
    secs = max(0.0, secs)
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _bar(done: int, total: int, width: int = 38) -> str:
    """ASCII progress bar: [####···]"""
    if total <= 0:
        return '[' + '·' * width + ']'
    filled = int(width * done / total)
    return '[' + '#' * filled + '·' * (width - filled) + ']'


def _row(label: str, value: str, lw: int = 12) -> str:
    return f"  {label:<{lw}}: {value}"


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_mode(log_path: str, state_db: StateDB, interval: int = 2) -> None:
    """
    Continuously display per-file chunk progress and overall queue stats.

    Reads log_path on every refresh for file-level chunk progress.
    Reads SQLite (state_db) for overall queue counts.
    Refreshes every `interval` seconds until Ctrl-C.
    """
    if not Path(log_path).exists():
        sys.exit(f"ERROR: log file not found: {log_path}")

    try:
        while True:
            log = parse_log(log_path)

            try:
                db = state_db.get_counts()
                db_err = None
            except Exception as exc:
                db = {}
                db_err = str(exc)

            _clear()
            now = time.strftime('%H:%M:%S')
            print(f"=== RAG Ingest Monitor  (refresh: {interval}s · Ctrl-C to stop) ===\n")

            # ---- Current file ----------------------------------------
            if log['current_file_path']:
                path  = log['current_file_path']
                fname = Path(path).name
                parts = Path(path).parts
                # Show last 4 path components to keep it readable
                short = os.path.join(*parts[-4:]) if len(parts) > 4 else path

                print(_row("Ingesting", fname))
                print(_row("Path", f".../{short}"))
                if log['file_type']:
                    strat = f"  ({log['strategy']})" if log['strategy'] else ""
                    print(_row("Type", f"{log['file_type']}{strat}"))
                print(_row("Phase", log['phase']))
                print()

                total = log['total_chunks']
                done  = log['chunks_embedded']

                if total is None:
                    print("  Chunks    : parsing in progress…")
                else:
                    pct_done      = done / total * 100 if total else 0.0
                    pct_left      = 100.0 - pct_done
                    remaining_cnt = total - done
                    print(_row("Chunks", f"{done:,} / {total:,}"))
                    print(f"  {_bar(done, total)}  {pct_done:.1f}% done · {pct_left:.1f}% remaining  ({remaining_cnt:,} left)")

                    # ETA: derived from embed_start_ts (Chunked line) and last progress ts
                    embed_start_ts   = log.get('embed_start_ts')
                    last_progress_ts = log.get('last_progress_ts')
                    if (
                        embed_start_ts and last_progress_ts
                        and last_progress_ts > embed_start_ts
                        and done > 0
                    ):
                        rate = done / (last_progress_ts - embed_start_ts)  # chunks/sec
                        elapsed_secs  = time.time() - embed_start_ts
                        eta_secs      = remaining_cnt / rate
                        eta_abs_str   = time.strftime('%H:%M:%S', time.localtime(time.time() + eta_secs))
                        print(
                            f"  Elapsed : {_fmt_duration(elapsed_secs)}"
                            f"  ·  Rate: {rate * 60:.1f} chunks/min"
                            f"  ·  ETA: ~{_fmt_duration(eta_secs)} ({eta_abs_str})"
                        )
            else:
                print("  No file currently being processed.")

            # ---- Overall progress ------------------------------------
            print()
            print("Overall (from DB):")

            if db_err:
                print(f"  DB unavailable: {db_err}")
            else:
                total_f  = db.get('total', 0)
                done_f   = db.get('completed', 0)
                pend_f   = db.get('pending', 0)
                proc_f   = db.get('processing', 0)
                ignor_f  = db.get('ignored', 0)
                pct_f    = done_f / total_f * 100 if total_f else 0.0

                print(_row("Files", f"{done_f:,} / {total_f:,}  ({pct_f:.1f}%)"))
                print(f"  {_bar(done_f, total_f)}")
                print()
                print(f"  {'pending':<12}: {pend_f:>6,}")
                print(f"  {'processing':<12}: {proc_f:>6,}")
                print(f"  {'completed':<12}: {done_f:>6,}")
                print(f"  {'ignored':<12}: {ignor_f:>6,}")
                print(f"  {'─' * 20}")
                print(f"  {'total':<12}: {total_f:>6,}")

            print(f"\n  Updated: {now}")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped.")


# ---------------------------------------------------------------------------
# Original one-shot status display
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Show RAG pipeline status")
    parser.add_argument("--user", metavar="OWNER", help="Filter counts by owner (e.g. florian)")
    parser.add_argument("--ignored", action="store_true", help="Detailed listing of ignored files with retry count and last error")
    parser.add_argument(
        "--watch", metavar="LOG_FILE",
        help="Continuously monitor a rag_ingest_daemon log file (error if not given)",
    )
    parser.add_argument(
        "--list-pending", dest="list_pending",
        nargs="?", type=int, const=-1, default=None, metavar="LIMIT",
        help="Print pending file paths (one per line). Optional LIMIT caps output. Honors --user.",
    )
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})
    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/HomeAI--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    state_db = StateDB(db_path)

    # --watch mode: live monitor from a log file
    if args.watch is not None:
        watch_mode(args.watch, state_db, interval=2)
        state_db.close()
        return

    # --list-pending mode: print pending file paths, one per line (pipeable)
    if args.list_pending is not None:
        limit = args.list_pending if args.list_pending > 0 else 10**9
        for path in state_db.fetch_pending(limit=limit, owner=args.user):
            print(path)
        state_db.close()
        return

    # --- SQLite queue counts ---
    if args.user:
        conn = state_db._conn
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM ingestion_queue "
            "WHERE owner = ? GROUP BY status",
            (args.user,),
        )
        raw = {row["status"]: row["n"] for row in cur.fetchall()}
        counts = {
            "pending":    raw.get("pending", 0),
            "processing": raw.get("processing", 0),
            "completed":  raw.get("completed", 0),
            "ignored":    raw.get("ignored", 0),
            "total":      sum(raw.values()),
        }
        scope = f"owner={args.user}"
    else:
        counts = state_db.get_counts()
        scope = "all users"

    print(f"\nIngestion queue ({scope}):")
    print(f"  pending    : {counts['pending']}")
    print(f"  processing : {counts['processing']}")
    print(f"  completed  : {counts['completed']}")
    print(f"  ignored    : {counts['ignored']}")
    print(f"  ─────────────────")
    print(f"  total      : {counts['total']}")

    if counts["total"] > 0:
        pct = counts["completed"] / counts["total"] * 100
        print(f"  progress   : {pct:.1f}%")

    # --- Qdrant stats ---
    try:
        qdrant = get_client(rag_cfg.get("qdrant_url", "http://192.168.1.93:6333"))
        existing = {c.name for c in qdrant.get_collections().collections}
        if DOCUMENTS_COLLECTION in existing:
            total_pts = qdrant.count(DOCUMENTS_COLLECTION, exact=True).count
            if args.user:
                user_pts = qdrant.count(
                    DOCUMENTS_COLLECTION,
                    count_filter=Filter(must=[
                        FieldCondition(key=FIELD_OWNER, match=MatchValue(value=args.user))
                    ]),
                    exact=True,
                ).count
                print(f"\nQdrant '{DOCUMENTS_COLLECTION}' collection:")
                print(f"  total points    : {total_pts}")
                print(f"  {args.user:15s} : {user_pts}")
            else:
                print(f"\nQdrant '{DOCUMENTS_COLLECTION}' collection:")
                print(f"  total points    : {total_pts}")
        else:
            print(f"\nQdrant: collection '{DOCUMENTS_COLLECTION}' does not exist yet")
    except Exception as exc:
        print(f"\nQdrant: unavailable ({exc})")

    # --- Ignored files ---
    if args.ignored:
        ignored = state_db.get_ignored(owner=args.user or None)
        if ignored:
            print(f"\nIgnored files ({len(ignored)}) — permanently skipped after retry exhaustion:")
            for row in ignored:
                print(f"  [{row['retry_count']} retries] {row['file_path']}")
                if row["skip_reason"]:
                    print(f"    → {row['skip_reason']}")
        else:
            print("\nNo ignored files.")

    state_db.close()


if __name__ == "__main__":
    main()
