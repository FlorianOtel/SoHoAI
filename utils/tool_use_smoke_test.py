#!/usr/bin/env python3
"""
Two-turn tool-use smoke test for the /v1/messages LiteLLM path.

Turn 1: POST with tools=[get_file_size(path:str)->int] and user prompt
        "What is the size of the file /etc/hostname?"
        → Expect response contains a tool_use block for get_file_size.

Turn 2: POST with turn-1 history + tool_result (synthetic: 42 bytes)
        → Expect final text answer referencing "42".

Parallel test (--parallel):
Turn 1: POST with 5 tools (get_file_size, get_file_owner, get_file_permissions,
        get_file_modification_time, get_file_line_count) and prompt asking
        for all five simultaneously.
        → Expect response contains exactly 5 tool_use blocks.

Turn 2: POST with 5 tool_result blocks in a single user message.
        → Expect final text referencing "42", "root", "644", "2024-03-15", and "17".

Targets: ollama-cloud/qwen3-coder-next, ollama-cloud/deepseek-v4-pro,
         ollama-cloud/kimi-k2.6, ollama-cloud/glm-5.1,
         internal/qwen3-4b (informational, non-gating).

Exit code: 0 if all ollama-cloud targets PASS, non-zero if any fail.
           Qwen3 failure is printed but does not affect exit code.
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

# Five tool definitions for the parallel test
PARALLEL_TOOL_DEFS = [
    {
        "name": "get_file_size",
        "description": "Return the size in bytes of the file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path to the file"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_file_owner",
        "description": "Return the username of the owner of the file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path to the file"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_file_permissions",
        "description": "Return the octal permissions string (e.g. '644') of the file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path to the file"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_file_modification_time",
        "description": "Return the last modification time of the file as an ISO date string (YYYY-MM-DD).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path to the file"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_file_line_count",
        "description": "Return the number of lines in the file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path to the file"}
            },
            "required": ["path"],
        },
    },
]

# Synthetic results returned by the harness in turn 2
PARALLEL_TOOL_RESULTS = {
    "get_file_size": "42",
    "get_file_owner": "root",
    "get_file_permissions": "644",
    "get_file_modification_time": "2024-03-15",
    "get_file_line_count": "17",
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


def _extract_tool_use_blocks_from_sse(response) -> list[dict]:
    """
    Parse SSE stream and extract all tool_use content blocks.
    Returns a list of dicts with keys: id, name, input (parsed JSON dict).
    """
    current_event = None
    current_data = None
    content_blocks: dict[int, dict] = {}  # index → {id, name, input_json}

    for line in response.iter_lines():
        if not line:
            # End of event — process accumulated event
            if current_event == "content_block_start" and current_data:
                data = json.loads(current_data)
                content_block = data.get("content_block", {})
                block_type = content_block.get("type")
                if block_type == "tool_use":
                    idx = data.get("index", 0)
                    content_blocks[idx] = {
                        "id": content_block.get("id"),
                        "name": content_block.get("name"),
                        "input_json": "",
                    }
            elif current_event == "content_block_delta" and current_data:
                data = json.loads(current_data)
                delta = data.get("delta", {})
                delta_type = delta.get("type")
                idx = data.get("index", 0)
                if delta_type == "input_json_delta":
                    partial_json = delta.get("partial_json", "")
                    if idx in content_blocks:
                        content_blocks[idx]["input_json"] += partial_json
            current_event = None
            current_data = None
            continue

        field, value = parse_sse_event(line)
        if field == "event":
            current_event = value
        elif field == "data":
            current_data = value

    # Parse accumulated JSON strings into input dicts
    result = []
    for idx in sorted(content_blocks.keys()):
        block = content_blocks[idx]
        input_json = block.get("input_json", "")
        try:
            parsed_input = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError:
            parsed_input = {"_raw": input_json}
        result.append({
            "id": block["id"],
            "name": block["name"],
            "input": parsed_input,
        })
    return result


def _extract_text_from_sse(response) -> str:
    """Parse SSE stream and accumulate all text_delta content."""
    current_event = None
    current_data = None
    text = ""

    for line in response.iter_lines():
        if not line:
            if current_event == "content_block_delta" and current_data:
                data = json.loads(current_data)
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text += delta.get("text", "")
            current_event = None
            current_data = None
            continue

        field, value = parse_sse_event(line)
        if field == "event":
            current_event = value
        elif field == "data":
            current_data = value

    return text


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
            with client.stream(
                "POST", f"{server}/v1/messages", json=turn1_body
            ) as response:
                if response.status_code != 200:
                    return (
                        False,
                        f"turn1 HTTP {response.status_code}: {response.text[:200]}",
                    )
                blocks = _extract_tool_use_blocks_from_sse(response)

            if blocks:
                tool_use_block = blocks[0]
                tool_use_id = tool_use_block.get("id")

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
                final_text = _extract_text_from_sse(response)

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


def run_parallel_tool_call_test(
    model: str,
    server: str,
    stream: bool,
    max_tokens: int,
    timeout: float,
) -> tuple[bool, str]:
    """
    Run a two-turn parallel tool-use test for a given model.

    Turn 1: Send 5 tool definitions + prompt asking for all 5 simultaneously.
            Expect exactly 5 tool_use blocks in a single assistant response.

    Turn 2: Send 5 tool_result blocks in one user message.
            Expect final text containing "42", "root", "644", "2024-03-15", and "17".

    Returns:
        (passed: bool, detail: str)
        - On pass: (True, "")
        - On fail: (False, "<diagnostic detail>")
    """
    try:
        client = httpx.Client(timeout=timeout)

        # ===== TURN 1: Request parallel tool calls =====
        turn1_messages = [
            {
                "role": "user",
                "content": (
                    "What is the size, owner, permissions, modification time, and line count "
                    "of the file /etc/hostname? Call all five tools in parallel."
                ),
            }
        ]

        turn1_body = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": PARALLEL_TOOL_DEFS,
            "messages": turn1_messages,
            "stream": stream,
        }

        tool_use_blocks: list[dict] = []

        if stream:
            with client.stream(
                "POST", f"{server}/v1/messages", json=turn1_body
            ) as response:
                if response.status_code != 200:
                    return (
                        False,
                        f"turn1 HTTP {response.status_code}: {response.text[:200]}",
                    )
                tool_use_blocks = _extract_tool_use_blocks_from_sse(response)

        else:
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
                    tool_use_blocks.append(block)

        # Validate: must have exactly 5 tool_use blocks
        if len(tool_use_blocks) != 5:
            tool_names = [b.get("name") for b in tool_use_blocks]
            return (
                False,
                f"turn1: expected 5 tool_use blocks, got {len(tool_use_blocks)}: {tool_names}",
            )

        # Validate: all expected tool names are present
        expected_names = {
            "get_file_size", "get_file_owner", "get_file_permissions",
            "get_file_modification_time", "get_file_line_count",
        }
        actual_names = {b.get("name") for b in tool_use_blocks}
        if actual_names != expected_names:
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            return (
                False,
                f"turn1: tool name mismatch — missing: {missing}, extra: {extra}",
            )

        # ===== TURN 2: Provide all 3 tool results in one user message =====
        # Reconstruct assistant content blocks (tool_use blocks as seen)
        assistant_content = []
        for block in tool_use_blocks:
            tool_input = block.get("input", {})
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {}
            block_id = block.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"
            assistant_content.append({
                "type": "tool_use",
                "id": block_id,
                "name": block.get("name"),
                "input": tool_input,
            })

        # Build tool_result list: one entry per tool_use block
        tool_results = []
        for ac_block in assistant_content:
            name = ac_block["name"]
            result_value = PARALLEL_TOOL_RESULTS.get(name, "unknown")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": ac_block["id"],
                "content": result_value,
            })

        turn2_messages = [
            {
                "role": "user",
                "content": (
                    "What is the size, owner, permissions, modification time, and line count "
                    "of the file /etc/hostname? Call all five tools in parallel."
                ),
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
            {
                "role": "user",
                "content": tool_results,
            },
        ]

        turn2_body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": turn2_messages,
            "stream": stream,
        }

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
                final_text = _extract_text_from_sse(response)

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

        # Validate: final text must mention all five synthetic results
        missing_vals = []
        for expected in ("42", "root", "644", "2024-03-15", "17"):
            if expected not in final_text:
                missing_vals.append(expected)

        if missing_vals:
            preview = final_text[:150] if final_text else "(empty)"
            return (
                False,
                f"turn2: missing {missing_vals} in response: '{preview}'",
            )

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
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run the parallel-tool-call test (3 simultaneous tools) instead of the standard single-tool test",
    )

    args = parser.parse_args()

    # Define targets: (model_name, informational_only)
    TARGETS = [
        ("ollama-cloud/qwen3-coder-next", False),
        ("ollama-cloud/deepseek-v4-pro", False),
        ("ollama-cloud/kimi-k2.6", False),
        ("ollama-cloud/glm-5.1", False),
        ("internal/qwen3-4b", True),
    ]

    # Filter by --model if specified
    if args.model:
        TARGETS = [(m, info) for m, info in TARGETS if m == args.model]
        if not TARGETS:
            print(f"Error: model '{args.model}' not in targets", file=sys.stderr)
            sys.exit(1)

    stream_label = "stream" if args.stream else "no-stream"
    any_fail = False

    if args.parallel:
        # Run parallel tool-call test
        print(f"=== Parallel tool-call test ({stream_label}, 5 tools) ===")
        for model, informational_only in TARGETS:
            passed, detail = run_parallel_tool_call_test(
                model, args.server, args.stream, args.max_tokens, args.timeout
            )

            if passed:
                status = "PASS" if not informational_only else "INFO"
                print(
                    f"[{model}]  {status}  ({stream_label})  "
                    f"— turn1: 5×tool_use (size+owner+perms+mtime+lines)  "
                    f"turn2: \"42\"+\"root\"+\"644\"+\"2024-03-15\"+\"17\""
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

    else:
        # Run standard single-tool test
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
