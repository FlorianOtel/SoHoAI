#!/usr/bin/env python3
"""
Two-turn tool-use smoke test for the /v1/messages LiteLLM path.

Turn 1: POST with tools=[get_file_size(path:str)->int] and user prompt
        "What is the size of the file /etc/hostname?"
        → Expect response contains a tool_use block for get_file_size.

Turn 2: POST with turn-1 history + tool_result (synthetic: 42 bytes)
        → Expect final text answer referencing "42".

Targets: ollama-cloud/qwen3-coder-next, ollama-cloud/deepseek-v4-pro,
         internal/gemma-4-e4b (informational, non-gating).

Exit code: 0 if all ollama-cloud targets PASS, non-zero if any fail.
           Gemma failure is printed but does not affect exit code.
"""

import argparse
import json
import sys
import uuid
from typing import Optional

import httpx


TOOL_DEF = {
    "name": "get_file_size",
    "description": "Return the size in bytes of the file at the given path.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "absolute path to the file"}
        },
        "required": ["path"],
    },
}


def parse_sse_event(line: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a single SSE line.
    Returns (field_name, value) if it's a field line, or (None, None) if it's empty.
    """
    if not line or line.startswith(":"):
        return None, None
    if ":" in line:
        field, _, value = line.partition(":")
        value = value.lstrip(" ")
        return field, value
    return line, ""


def run_two_turn(
    model: str,
    server: str,
    stream: bool,
    max_tokens: int,
    timeout: float,
) -> tuple[bool, str]:
    """
    Run a two-turn tool-use smoke test for a given model.

    Returns:
        (passed: bool, detail: str)
        - On pass: (True, "")
        - On fail: (False, "<diagnostic detail>")
    """
    try:
        client = httpx.Client(timeout=timeout)

        # ===== TURN 1: Request tool use =====
        turn1_messages = [
            {
                "role": "user",
                "content": "What is the size of the file /etc/hostname?",
            }
        ]

        turn1_body = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": [TOOL_DEF],
            "messages": turn1_messages,
            "stream": stream,
        }

        # Turn 1: Extract tool_use block
        tool_use_block = None
        tool_use_id = None

        if stream:
            # Streaming: parse SSE events
            with client.stream(
                "POST", f"{server}/v1/messages", json=turn1_body
            ) as response:
                if response.status_code != 200:
                    return (
                        False,
                        f"turn1 HTTP {response.status_code}: {response.text[:200]}",
                    )

                # Accumulate SSE events
                current_event = None
                current_data = None
                content_blocks = {}  # index -> {"type", "id", "name", "input_str"}
                text_block_open = False

                for line in response.iter_lines():
                    if not line:
                        # End of event
                        if current_event == "content_block_start" and current_data:
                            data = json.loads(current_data)
                            content_block = data.get("content_block", {})
                            block_type = content_block.get("type")
                            if block_type == "tool_use":
                                idx = data.get("index", 0)
                                content_blocks[idx] = {
                                    "type": "tool_use",
                                    "id": content_block.get("id"),
                                    "name": content_block.get("name"),
                                    "input_str": "",
                                }
                        elif (
                            current_event == "content_block_delta"
                            and current_data
                        ):
                            data = json.loads(current_data)
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")
                            idx = data.get("index", 0)
                            if delta_type == "input_json_delta":
                                partial_json = delta.get("partial_json", "")
                                if idx in content_blocks:
                                    content_blocks[idx][
                                        "input_str"
                                    ] += partial_json
                        current_event = None
                        current_data = None
                        continue

                    field, value = parse_sse_event(line)
                    if field == "event":
                        current_event = value
                    elif field == "data":
                        current_data = value

                # Extract tool_use block with type "tool_use" and parse accumulated input_str → input dict
                for idx, block in content_blocks.items():
                    if block.get("type") == "tool_use":
                        input_str = block.get("input_str", "")
                        try:
                            block["input"] = json.loads(input_str) if input_str else {}
                        except json.JSONDecodeError:
                            block["input"] = {"_raw": input_str}
                        tool_use_block = block
                        tool_use_id = block.get("id")
                        break

        else:
            # Non-streaming: read JSON response
            response = client.post(f"{server}/v1/messages", json=turn1_body)
            if response.status_code != 200:
                return (
                    False,
                    f"turn1 HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            content = data.get("content", [])
            for block in content:
                if block.get("type") == "tool_use":
                    tool_use_block = block
                    tool_use_id = block.get("id")
                    break

        # Validate tool_use block
        if not tool_use_block:
            content_types = (
                [b.get("type") for b in (data.get("content", []))]
                if not stream
                else []
            )
            return (
                False,
                f"turn1: no tool_use block in response (got types: {content_types})",
            )

        if tool_use_block.get("name") != "get_file_size":
            return (
                False,
                f"turn1: expected tool name 'get_file_size', got '{tool_use_block.get('name')}'",
            )

        tool_input = tool_use_block.get("input", {})
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {}

        if tool_input.get("path") != "/etc/hostname":
            return (
                False,
                f"turn1: expected path '/etc/hostname', got '{tool_input.get('path')}'",
            )

        # ===== TURN 2: Provide tool result =====
        if not tool_use_id:
            tool_use_id = f"toolu_{uuid.uuid4().hex[:16]}"

        turn2_messages = [
            {"role": "user", "content": "What is the size of the file /etc/hostname?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "get_file_size",
                        "input": tool_input,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "42",
                    }
                ],
            },
        ]

        turn2_body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": turn2_messages,
            "stream": stream,
        }

        # Turn 2: Extract final text response
        final_text = ""

        if stream:
            with client.stream(
                "POST", f"{server}/v1/messages", json=turn2_body
            ) as response:
                if response.status_code != 200:
                    return (
                        False,
                        f"turn2 HTTP {response.status_code}: {response.text[:200]}",
                    )

                current_event = None
                current_data = None

                for line in response.iter_lines():
                    if not line:
                        if current_event == "content_block_delta" and current_data:
                            data = json.loads(current_data)
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                final_text += delta.get("text", "")
                        current_event = None
                        current_data = None
                        continue

                    field, value = parse_sse_event(line)
                    if field == "event":
                        current_event = value
                    elif field == "data":
                        current_data = value

        else:
            response = client.post(f"{server}/v1/messages", json=turn2_body)
            if response.status_code != 200:
                return (
                    False,
                    f"turn2 HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            content = data.get("content", [])
            for block in content:
                if block.get("type") == "text":
                    final_text += block.get("text", "")

        # Validate final text contains "42"
        if "42" not in final_text:
            preview = final_text[:100] if final_text else "(empty)"
            return (False, f"turn2: '42' not in response: '{preview}'")

        client.close()
        return (True, "")

    except Exception as e:
        return (False, f"exception: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-turn tool-use smoke test for LiteLLM path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default="http://192.168.1.93:8001",
        help="Base URL of SoHoAI orchestrator (default: http://192.168.1.93:8001)",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Use streaming responses (default)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming, use non-streaming responses",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Test a single model (omit to test all targets)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Max tokens for LLM responses (default: 1500)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="HTTP timeout in seconds (default: 120)",
    )

    args = parser.parse_args()

    # Define targets: (model_name, informational_only)
    TARGETS = [
        ("ollama-cloud/qwen3-coder-next", False),
        ("ollama-cloud/deepseek-v4-pro", False),
        ("internal/gemma-4-e4b", True),
    ]

    # Filter by --model if specified
    if args.model:
        TARGETS = [(m, info) for m, info in TARGETS if m == args.model]
        if not TARGETS:
            print(f"Error: model '{args.model}' not in targets", file=sys.stderr)
            sys.exit(1)

    stream_label = "stream" if args.stream else "no-stream"
    any_fail = False

    for model, informational_only in TARGETS:
        passed, detail = run_two_turn(
            model, args.server, args.stream, args.max_tokens, args.timeout
        )

        if passed:
            status = "PASS" if not informational_only else "INFO"
            print(
                f"[{model}]  {status}  ({stream_label})  "
                f"— turn1: tool_use get_file_size(path='/etc/hostname')  turn2: \"42\""
            )
        else:
            if informational_only:
                print(
                    f"[{model}]  INFO  ({stream_label})  "
                    f"— FAIL {detail}"
                )
            else:
                print(
                    f"[{model}]  FAIL  ({stream_label})  "
                    f"— {detail}"
                )
                any_fail = True

    if any_fail:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
