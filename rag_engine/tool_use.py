"""System-prompt-driven tool-use for RAG. Provider-agnostic.

Exported functions:
  build_tool_spec()          -> str   # TOOLS section for the system prompt
  parse_tool_call(text)      -> dict | None   # extract first <tool_call> block
  format_tool_result(chunks) -> str   # render chunks as a tool-role message
"""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

def build_tool_spec() -> str:
    """Return the TOOLS section string. Static for now (one tool)."""
    return """## TOOLS

You have access to the following tools. Call a tool by writing a single
<tool_call>…</tool_call> block as your entire reply — no prose before or
after, no additional text.

### search_documents(query: str) -> list[Document]

Searches the user's personal corpus and returns up to N relevant chunks.
Each result has:
  - source_path: full NFS path (cite this in your answer)
  - score:       relevance score (0–1, higher = more relevant)
  - content:     chunk text (use this to answer)

Call this tool when you need information from the user's documents.
DO NOT call it for greetings, identity questions, or general knowledge.

Example invocation:
<tool_call>
{"name": "search_documents", "arguments": {"query": "AWS certifications"}}
</tool_call>"""

def parse_tool_call(assistant_text: str) -> dict[str, Any] | None:
    """Find first <tool_call>…</tool_call> block. Return {"name":..., "arguments":...}
    or None if no valid block found. Malformed JSON -> None + log warning."""
    m = _TOOL_CALL_RE.search(assistant_text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning("tool_call JSON decode failed: %s (text=%r)", e, m.group(1))
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    call.setdefault("arguments", {})
    return call

def format_tool_result(chunks: list[dict]) -> str:
    """Render retrieved chunks into a tool-role message body.

    Compact JSON-ish format for the LLM; not the prompt the user sees.
    """
    if not chunks:
        return "search_documents returned no results."
    lines = [f"search_documents returned {len(chunks)} result(s):\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] score={c.get('score', 'N/A'):.4f}  source={c.get('source_path', 'N/A')}\n"
            f"{c.get('content', 'N/A')}\n"
        )
    return "\n".join(lines)