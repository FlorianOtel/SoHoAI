#!/usr/bin/env python3
"""
SoHoAI — Terminal Chat Client (Phase 1)

A simple but functional CLI that talks to the orchestrator API.
Supports:
  - Multi-turn conversation with memory
  - Model selection (/internal, /model)
  - Chat management (/list, /load, /export, /save)
  - RAG toggle (/rag on|off|only|search)
  - Feedback (/thumbsup, /thumbsdown)
  - Input history and editing (up/down arrows, emacs-style Ctrl+A/E/K/U/W)

Usage:
    python cli_chat.py [--server http://SERVER_IP:8000] [--user florian]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

import httpx
import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

# Allow in-process RAG retrieval for /rag search
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_engine.collection import DOCUMENTS_COLLECTION, get_client  # noqa: E402
from rag_engine.search import search_rag  # noqa: E402


# -- Config --------------------------------------------------------------------

# Resolve config.yaml relative to this script's directory so the path is
# correct regardless of the working directory the caller uses.
# ("../config.yaml" only works when run from utils/, but the documented
# invocation is `python utils/cli_chat.py` from the project root.)
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

try:
    with open(_CONFIG_PATH) as f:
        _CONFIG = yaml.safe_load(f)
        DEFAULT_SERVER = f"http://{_CONFIG.get('server1_ip', '192.168.1.93')}:8000"
        RAG_CFG = _CONFIG.get("rag", {}) or {}
except Exception:
    _CONFIG = {}
    DEFAULT_SERVER = "http://192.168.1.93:8000"
    RAG_CFG = {}

HELP_TEXT = """
Commands:
  /help                 Show this help
  /new                  Start a new chat
  /model [name]         Show/switch external model (e.g., gpt4, claude; auto=default)
  /internal             Switch to internal LLM (Qwen3.5-4B on GPU)
  /cloud                Force next message to use cloud model
  /rag [on|off|only]    Set RAG mode (default: off)
  /rag status           Show RAG config (mode, user, top_k, Qdrant points)
  /rag search <query>   Inspect retrieval hits for a query (no LLM call)
  /user <id>            Set user_id for RAG ownership filter (e.g. florian)
  /list                 List saved chats
  /load <chat_id>       Load a previous chat
  /export               Export current chat as Markdown
  /save                 Save current chat as Markdown to NAS
  /feedback up|down     Rate the last assistant message
  /health               Check system health
  /quit                 Exit

Line editing:
  ↑/↓                   Previous/next input from history
  Ctrl+A / Ctrl+E       Move to start/end of line
  Ctrl+K / Ctrl+U       Kill from cursor/start to end
  Ctrl+W                Delete word backward
"""


class CLIChat:
    def __init__(self, server_url: str, user_id: str | None):
        self.server = server_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)
        self.chat_id = str(uuid.uuid4())
        self.model: str | None = None  # Explicit model override; None = use router default (external)
        self.force_cloud = False
        self.rag_mode = "off"  # opt-in — caller enables via /rag on or /rag only
        self.user_id = user_id
        self.turn_count = 0
        self.rag_cfg = RAG_CFG
        self.qdrant_url = RAG_CFG.get("qdrant_url", "http://192.168.1.93:6333")
        self.top_k = int(RAG_CFG.get("top_k", 5))

        # Initialize prompt session with history and emacs-style editing
        history_path = Path.home() / ".cli_chat_history"
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            enable_history_search=True,
        )

    # -- RAG helpers --------------------------------------------------------

    def _qdrant_points(self) -> int | None:
        try:
            r = httpx.get(
                f"{self.qdrant_url}/collections/{DOCUMENTS_COLLECTION}",
                timeout=3,
            )
            r.raise_for_status()
            return r.json()["result"]["points_count"]
        except Exception:
            return None

    def _mq_suffix(self) -> str:
        mq = self.rag_cfg.get("multi_query", {}) or {}
        if not mq.get("enabled"):
            return ""
        n = mq.get("n_variants", 3)
        lam = mq.get("lambda", 0.5)
        return f"  multi_query=true (n={n}, λ={lam:.2f})"

    def preflight(self) -> None:
        """One-shot banner line describing the RAG setup."""
        pts = self._qdrant_points()
        user = self.user_id or "none (no ownership filter)"
        if pts is None:
            print(f"  RAG: Qdrant unreachable at {self.qdrant_url}")
        else:
            print(
                f"  RAG: {self.rag_mode}  user={user}  top_k={self.top_k}"
                f"{self._mq_suffix()}  "
                f"({pts} points in '{DOCUMENTS_COLLECTION}')"
            )

    def _rag_search_inline(self, query: str) -> str:
        """Run search_rag() in-process and format hits for display."""
        async def _search():
            client = get_client(self.qdrant_url)
            return await search_rag(
                query=query,
                user_id=self.user_id,
                limit=self.top_k,
                qdrant_client=client,
                rag_cfg=self.rag_cfg,
            )

        try:
            results = asyncio.run(_search())
        except Exception as e:
            return f"  [ERROR] RAG search failed: {e}"

        if not results:
            return "  (no results)"
        lines = [f"  {len(results)} hit(s) for {query!r}  user={self.user_id!r}:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"    {i} {r['score']:.4f}  {r['file_name'][:40]:<40}  {r['source_path']}"
            )
        return "\n".join(lines)

    # -- chat ---------------------------------------------------------------

    def send_message(self, content: str) -> str:
        """Send a message and return the assistant's reply."""
        payload = {
            "chat_id": self.chat_id,
            "messages": [{"role": "user", "content": content}],
            "model": self.model,
            "force_cloud": self.force_cloud,
            "rag_mode": self.rag_mode,
            "user_id": self.user_id,
            "stream": False,
        }
        # Reset one-shot flags
        self.force_cloud = False

        try:
            resp = self.client.post(f"{self.server}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.turn_count += 1

            model_used = data.get("model_used", "?")
            content = data["message"]["content"]
            sources = data.get("rag_sources") or []

            header = f"  [{model_used}]"
            source_block = ""
            if sources:
                source_block = "\n  Sources:\n" + "\n".join(f"    - {s}" for s in sources)
            elif self.rag_mode != "off":
                source_block = f"\n  (RAG {self.rag_mode} — no relevant context found)"

            return f"{header}\n{content}{source_block}"

        except httpx.HTTPStatusError as e:
            return f"  [ERROR] Server returned {e.response.status_code}: {e.response.text}"
        except httpx.ConnectError:
            return f"  [ERROR] Cannot connect to {self.server}"

    def handle_command(self, cmd: str) -> str | None:
        """Handle slash commands. Returns display text or None to continue."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            return HELP_TEXT

        elif command == "/new":
            self.chat_id = str(uuid.uuid4())
            self.turn_count = 0
            return "  Started new chat."

        elif command == "/model":
            if not arg:
                if self.model is None:
                    mode_info = "AUTO (router default: external/Sonnet)"
                elif self.model in ("internal", "internal/qwen3-4b"):
                    mode_info = "INTERNAL (Qwen3.5-4B on GPU)"
                else:
                    mode_info = f"EXTERNAL ({self.model})"
                return f"  Current LLM mode: {mode_info}"

            if arg.lower() == "internal":
                self.model = "internal/qwen3-4b"
                return "  Switched to internal LLM (Qwen3.5-4B on GPU)."
            else:
                # Assume any other argument is an external model name
                self.model = arg if arg != "auto" else None
                return f"  External model set to: {self.model or 'auto'}."

        elif command == "/internal":
            self.model = "internal/qwen3-4b"
            return "  Switched to internal LLM (Qwen3.5-4B on GPU)."

        elif command == "/cloud":
            self.force_cloud = True
            return "  Next message will use cloud model."

        elif command == "/rag":
            sub = arg.strip().split(maxsplit=1)
            head = sub[0].lower() if sub else ""

            if head == "on":
                self.rag_mode = "on"
                return "  RAG: on"
            if head == "off":
                self.rag_mode = "off"
                return "  RAG: off"
            if head == "only":
                self.rag_mode = "only"
                return "  RAG: only (grounded — no prior knowledge)"
            if head == "search":
                if len(sub) < 2 or not sub[1].strip():
                    return "  Usage: /rag search <query>"
                return self._rag_search_inline(sub[1].strip())
            if head in ("", "status"):
                pts = self._qdrant_points()
                pts_str = f"{pts} points" if pts is not None else "Qdrant unreachable"
                return (
                    f"  RAG: {self.rag_mode}  "
                    f"user={self.user_id or 'none'}  "
                    f"top_k={self.top_k}{self._mq_suffix()}  ({pts_str})"
                )
            return "  Usage: /rag [on|off|only|status|search <query>]"

        elif command == "/user":
            if not arg.strip():
                return f"  user_id: {self.user_id or 'none'}"
            self.user_id = arg.strip()
            return f"  user_id set to: {self.user_id}"

        elif command == "/list":
            try:
                resp = self.client.get(f"{self.server}/v1/chats?limit=15")
                chats = resp.json()
                if not chats:
                    return "  No saved chats."
                lines = ["  Recent chats:"]
                for c in chats:
                    lines.append(f"    {c['chat_id'][:8]}  {c['title']}  ({c['turn_count']} turns)")
                return "\n".join(lines)
            except Exception as e:
                return f"  [ERROR] {e}"

        elif command == "/load":
            if not arg:
                return "  Usage: /load <chat_id or prefix>"
            # Support prefix matching
            try:
                resp = self.client.get(f"{self.server}/v1/chats")
                chats = resp.json()
                match = next((c for c in chats if c["chat_id"].startswith(arg)), None)
                if match:
                    self.chat_id = match["chat_id"]
                    self.turn_count = match["turn_count"]
                    return f"  Loaded: {match['title']} ({self.turn_count} turns)"
                return "  Chat not found."
            except Exception as e:
                return f"  [ERROR] {e}"

        elif command == "/export":
            try:
                resp = self.client.get(f"{self.server}/v1/chats/{self.chat_id}/export/markdown")
                if resp.status_code == 200:
                    print(resp.text)
                    return None
                return "  No chat to export."
            except Exception as e:
                return f"  [ERROR] {e}"

        elif command == "/save":
            try:
                resp = self.client.post(f"{self.server}/v1/chats/{self.chat_id}/export/save")
                data = resp.json()
                return f"  Saved to: {data.get('path', '?')}"
            except Exception as e:
                return f"  [ERROR] {e}"

        elif command in ("/feedback", "/thumbsup", "/thumbsdown"):
            signal = "thumbs_up"
            if command == "/thumbsdown" or arg.lower() == "down":
                signal = "thumbs_down"
            try:
                self.client.post(
                    f"{self.server}/v1/chats/{self.chat_id}/feedback",
                    params={"message_index": self.turn_count * 2 - 1, "signal": signal},
                )
                return f"  Feedback recorded: {signal}"
            except Exception as e:
                return f"  [ERROR] {e}"

        elif command == "/health":
            try:
                resp = self.client.get(f"{self.server}/health")
                return f"  {json.dumps(resp.json(), indent=2)}"
            except Exception as e:
                return f"  [ERROR] {e}"

        elif command in ("/quit", "/exit", "/q"):
            print("  Goodbye!")
            sys.exit(0)

        else:
            return f"  Unknown command: {command}. Type /help for commands."

    def run(self):
        """Main REPL loop with history and emacs-style line editing."""
        print("╔══════════════════════════════════════╗")
        print("║       SoHoAI — Terminal Chat     ║")
        print("║  Type /help for commands, /quit to   ║")
        print("║  exit. Just type to chat.            ║")
        print("║  ↑/↓ for history, Ctrl+A/E for line  ║")
        print("╚══════════════════════════════════════╝")
        self.preflight()
        print()

        while True:
            try:
                user_input = self.prompt_session.prompt("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                result = self.handle_command(user_input)
                if result:
                    print(result)
            else:
                print()
                response = self.send_message(user_input)
                print(response)
                print()


def main():
    parser = argparse.ArgumentParser(description="SoHoAI CLI Chat")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Orchestrator URL")
    parser.add_argument(
        "--user",
        default=None,
        help="user_id for RAG ownership filter (e.g. florian). "
             "Omit to search all documents (dev mode — pre-OAuth).",
    )
    args = parser.parse_args()

    chat = CLIChat(args.server, user_id=args.user)
    chat.run()


if __name__ == "__main__":
    main()