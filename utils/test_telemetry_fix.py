#!/usr/bin/env python3
"""Unit tests for the LiteLLM telemetry fix.

Tests:
  - _read_active_orchestra_session_id() — session detection from .lck files
  - Opus 4.7 cache rates registered via litellm.register_model()
  - SSE usage parsing (_parse_sse_chunk logic extracted for testability)

Offline: no network, no server, no Redis required.

Run:
  python utils/test_telemetry_fix.py
  (from worktree root, with venv activated)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Set dummy credentials before importing main (which constructs the FastAPI app)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-telemetry-test")
os.environ.setdefault("OLLAMA_API_KEY", "test-key-for-telemetry-test")

sys.path.insert(0, ".")

try:
    import main as _main
    from main import _read_active_orchestra_session_id, _ORCHESTRA_LCK_RE
except Exception as e:
    print(f"FAIL: Could not import from main.py: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_lck(directory: Path, name: str, cc_pid: int) -> Path:
    """Write a mock .lck file."""
    p = directory / name
    p.write_text(f"cc_pid={cc_pid}\n")
    return p


# ---------------------------------------------------------------------------
# Tests: _read_active_orchestra_session_id()
# ---------------------------------------------------------------------------

def test_read_active_session_no_dir():
    """Returns None when active-sessions directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        # Point to a non-existent subdir
        fake_dir = Path(tmp) / "nonexistent"
        orig = _main._ACTIVE_SESSIONS_DIR
        _main._ACTIVE_SESSIONS_DIR = fake_dir
        try:
            result = _read_active_orchestra_session_id()
            assert result is None, f"Expected None for missing dir, got {result!r}"
        finally:
            _main._ACTIVE_SESSIONS_DIR = orig
    return []


def test_read_active_session_native_only():
    """native-<UUID>.lck files are ignored; returns None."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_lck(d, "native-61ae555b-0ec5-4789-a3ee-5dce7802521a.lck", os.getpid())
        orig = _main._ACTIVE_SESSIONS_DIR
        _main._ACTIVE_SESSIONS_DIR = d
        try:
            result = _read_active_orchestra_session_id()
            assert result is None, f"Expected None for native-only dir, got {result!r}"
        finally:
            _main._ACTIVE_SESSIONS_DIR = orig
    return []


def test_read_active_session_finds_orchestra():
    """Correctly reads orchestra session ID from a live .lck file."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        session_id = "20260510T204552Z-2645364"
        # Use the current process PID — it's guaranteed to be alive
        _write_lck(d, f"{session_id}.lck", os.getpid())
        orig = _main._ACTIVE_SESSIONS_DIR
        _main._ACTIVE_SESSIONS_DIR = d
        try:
            result = _read_active_orchestra_session_id()
            assert result == session_id, f"Expected {session_id!r}, got {result!r}"
        finally:
            _main._ACTIVE_SESSIONS_DIR = orig
    return []


def test_read_active_session_stale_pid():
    """Skips a .lck whose cc_pid is dead (uses PID 99999999 as guaranteed-dead)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # PID 99999999 is almost certainly not alive on any Linux system
        _write_lck(d, "20260510T204552Z-9999.lck", 99999999)
        orig = _main._ACTIVE_SESSIONS_DIR
        _main._ACTIVE_SESSIONS_DIR = d
        try:
            result = _read_active_orchestra_session_id()
            assert result is None, f"Expected None for stale PID, got {result!r}"
        finally:
            _main._ACTIVE_SESSIONS_DIR = orig
    return []


def test_read_active_session_native_plus_orchestra():
    """Ignores native-*.lck and returns orchestra session ID when both exist."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_lck(d, "native-61ae555b-0ec5-4789-a3ee-5dce7802521a.lck", os.getpid())
        session_id = "20260510T184532Z-1234567"
        _write_lck(d, f"{session_id}.lck", os.getpid())
        orig = _main._ACTIVE_SESSIONS_DIR
        _main._ACTIVE_SESSIONS_DIR = d
        try:
            result = _read_active_orchestra_session_id()
            assert result == session_id, f"Expected {session_id!r}, got {result!r}"
        finally:
            _main._ACTIVE_SESSIONS_DIR = orig
    return []


# ---------------------------------------------------------------------------
# Tests: Opus 4.7 cache rates
# ---------------------------------------------------------------------------

def test_opus_cache_rates_registered():
    """After litellm.register_model() at module import, Opus 4.7 cache rates are non-zero."""
    import litellm
    info = litellm.get_model_info("claude-opus-4-7")
    failures = []
    cc_rate = (info or {}).get("cache_creation_input_token_cost", 0.0) or 0.0
    cr_rate = (info or {}).get("cache_read_input_token_cost", 0.0) or 0.0
    if cc_rate == 0.0:
        failures.append(
            f"claude-opus-4-7 cache_creation_input_token_cost is 0.0 "
            f"(expected 0.00001875); litellm.register_model() may not have run"
        )
    if cr_rate == 0.0:
        failures.append(
            f"claude-opus-4-7 cache_read_input_token_cost is 0.0 "
            f"(expected 0.0000015); litellm.register_model() may not have run"
        )
    expected_cc = 0.00001875
    expected_cr = 0.0000015
    if abs(cc_rate - expected_cc) > 1e-12:
        failures.append(f"claude-opus-4-7 cache_creation rate {cc_rate} != {expected_cc}")
    if abs(cr_rate - expected_cr) > 1e-14:
        failures.append(f"claude-opus-4-7 cache_read rate {cr_rate} != {expected_cr}")
    return failures


# ---------------------------------------------------------------------------
# Tests: SSE usage parsing (logic extracted from _forward_stream)
# ---------------------------------------------------------------------------

def _run_sse_parser(chunks: list[bytes]) -> dict:
    """Simulate the _parse_sse_chunk logic used in _forward_stream."""
    usage_agg = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }
    streamed_model = ["unknown"]
    line_buf = [""]

    def parse(raw: bytes) -> None:
        line_buf[0] += raw.decode("utf-8", errors="ignore")
        while "\n" in line_buf[0]:
            line, line_buf[0] = line_buf[0].split("\n", 1)
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
                ev_type = ev.get("type")
                if ev_type == "message_start":
                    msg = ev.get("message", {})
                    u = msg.get("usage", {})
                    for k in ("input_tokens", "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                        usage_agg[k] += u.get(k, 0)
                    if msg.get("model"):
                        streamed_model[0] = msg["model"]
                elif ev_type == "message_delta":
                    usage_agg["output_tokens"] += ev.get("usage", {}).get("output_tokens", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    for chunk in chunks:
        parse(chunk)
    return {"usage": usage_agg, "model": streamed_model[0]}


def _sse_line(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def test_sse_parsing_basic():
    """message_start + content_block_delta + message_delta → correct token counts."""
    chunks = [
        _sse_line("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7-20250514",
                "usage": {
                    "input_tokens": 25,
                    "cache_creation_input_tokens": 1500,
                    "cache_read_input_tokens": 8000,
                    "output_tokens": 1,
                },
            },
        }),
        _sse_line("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }),
        _sse_line("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 236},
        }),
        b"event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n",
    ]
    result = _run_sse_parser(chunks)
    failures = []
    u = result["usage"]
    if u["input_tokens"] != 25:
        failures.append(f"input_tokens {u['input_tokens']} != 25")
    if u["cache_creation_input_tokens"] != 1500:
        failures.append(f"cache_creation_input_tokens {u['cache_creation_input_tokens']} != 1500")
    if u["cache_read_input_tokens"] != 8000:
        failures.append(f"cache_read_input_tokens {u['cache_read_input_tokens']} != 8000")
    if u["output_tokens"] != 236:
        failures.append(f"output_tokens {u['output_tokens']} != 236")
    if result["model"] != "claude-opus-4-7-20250514":
        failures.append(f"model {result['model']!r} != 'claude-opus-4-7-20250514'")
    return failures


def test_sse_parsing_split_chunks():
    """SSE data split across multiple byte chunks is handled correctly."""
    full_event = (
        'event: message_start\n'
        'data: {"type": "message_start", "message": {"id": "x", "type": "message", '
        '"role": "assistant", "model": "claude-opus-4-7", '
        '"usage": {"input_tokens": 100, "cache_creation_input_tokens": 500, '
        '"cache_read_input_tokens": 2000, "output_tokens": 0}}}\n\n'
        'event: message_delta\n'
        'data: {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 50}}\n\n'
    )
    # Split into small chunks to exercise line-buffering
    raw = full_event.encode()
    chunks = [raw[i:i+20] for i in range(0, len(raw), 20)]
    result = _run_sse_parser(chunks)
    failures = []
    u = result["usage"]
    if u["input_tokens"] != 100:
        failures.append(f"split: input_tokens {u['input_tokens']} != 100")
    if u["cache_creation_input_tokens"] != 500:
        failures.append(f"split: cache_creation {u['cache_creation_input_tokens']} != 500")
    if u["cache_read_input_tokens"] != 2000:
        failures.append(f"split: cache_read {u['cache_read_input_tokens']} != 2000")
    if u["output_tokens"] != 50:
        failures.append(f"split: output_tokens {u['output_tokens']} != 50")
    return failures


def test_sse_parsing_no_usage():
    """Stream with no usage events produces zero counts (no crash)."""
    chunks = [
        b"event: ping\ndata: {\"type\": \"ping\"}\n\n",
        b"event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n",
    ]
    result = _run_sse_parser(chunks)
    failures = []
    if sum(result["usage"].values()) != 0:
        failures.append(f"no-usage stream produced non-zero: {result['usage']}")
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("read_active_session_no_dir", test_read_active_session_no_dir),
        ("read_active_session_native_only", test_read_active_session_native_only),
        ("read_active_session_finds_orchestra", test_read_active_session_finds_orchestra),
        ("read_active_session_stale_pid", test_read_active_session_stale_pid),
        ("read_active_session_native_plus_orchestra", test_read_active_session_native_plus_orchestra),
        ("opus_cache_rates_registered", test_opus_cache_rates_registered),
        ("sse_parsing_basic", test_sse_parsing_basic),
        ("sse_parsing_split_chunks", test_sse_parsing_split_chunks),
        ("sse_parsing_no_usage", test_sse_parsing_no_usage),
    ]

    all_failures = []
    for name, fn in tests:
        print(f"  {name}...", end=" ", flush=True)
        try:
            failures = fn()
            if failures:
                print("FAIL")
                for f in failures:
                    print(f"    - {f}")
                all_failures.extend(failures)
            else:
                print("ok")
        except Exception as e:
            msg = f"EXCEPTION in {name}: {e}"
            print(f"FAIL\n    - {msg}")
            all_failures.append(msg)

    if all_failures:
        print(f"\nFAIL: {len(all_failures)} assertion(s) failed")
        return 1

    print(f"\nOK: {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
