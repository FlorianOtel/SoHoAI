#!/usr/bin/env python3
"""
HomeAI-Lab — Terminal Chat Client (Phase 1)

A simple but functional CLI that talks to the orchestrator API.
Supports:
  - Multi-turn conversation with memory
  - Model selection (/model specialist)
  - Chat management (/list, /load, /delete, /export)
  - RAG toggle (/rag on)
  - Feedback (/thumbsup, /thumbsdown)

Usage:
    python cli_chat.py [--server http://SERVER_IP:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import httpx
import yaml


# -- Config --------------------------------------------------------------------

# Resolve config.yaml relative to this script's directory so the path is
# correct regardless of the working directory the caller uses.
# ("../config.yaml" only works when run from utils/, but the documented
# invocation is `python utils/cli_chat.py` from the project root.)
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

try:
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
        DEFAULT_SERVER = f"http://{config.get('server1_ip', '192.168.1.93')}:8000"
except Exception:
    DEFAULT_SERVER = "http://192.168.1.93:8000"

HELP_TEXT = """
Commands:
  /help              Show this help
  /new               Start a new chat
  /model <name>      Switch model (specialist, external)
  /cloud             Force next message to use cloud model
  /rag on|off        Toggle RAG augmentation
  /list              List saved chats
  /load <chat_id>    Load a previous chat
  /export            Export current chat as Markdown
  /save              Save current chat as Markdown to NAS
  /feedback up|down  Rate the last assistant message
  /health            Check system health
  /quit              Exit
"""


class CLIChat:
    def __init__(self, server_url: str):
        self.server = server_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)
        self.chat_id = str(uuid.uuid4())
        self.model: str | None = None
        self.force_cloud = False
        self.use_rag = False
        self.turn_count = 0

    def send_message(self, content: str) -> str:
        """Send a message and return the assistant's reply."""
        payload = {
            "chat_id": self.chat_id,
            "messages": [{"role": "user", "content": content}],
            "model": self.model,
            "force_cloud": self.force_cloud,
            "use_rag": self.use_rag,
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
            sources = data.get("rag_sources")

            header = f"  [{model_used}]"
            source_line = ""
            if sources:
                source_line = f"\n  Sources: {', '.join(sources)}"

            return f"{header}\n{content}{source_line}"

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
                return f"  Current model: {self.model or 'auto'}"
            self.model = arg if arg != "auto" else None
            return f"  Model set to: {self.model or 'auto'}"

        elif command == "/cloud":
            self.force_cloud = True
            return "  Next message will use cloud model."

        elif command == "/rag":
            self.use_rag = arg.lower() == "on"
            return f"  RAG: {'on' if self.use_rag else 'off'}"

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
        """Main REPL loop."""
        print("╔══════════════════════════════════════╗")
        print("║       HomeAI-Lab — Terminal Chat     ║")
        print("║  Type /help for commands, /quit to   ║")
        print("║  exit. Just type to chat.            ║")
        print("╚══════════════════════════════════════╝")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
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
    parser = argparse.ArgumentParser(description="HomeAI-Lab CLI Chat")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Orchestrator URL")
    args = parser.parse_args()

    chat = CLIChat(args.server)
    chat.run()


if __name__ == "__main__":
    main()
