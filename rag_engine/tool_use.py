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
    """Return the TOOLS section string. Static (one tool)."""
    return """## TOOLS

You have access to the following tools. Call a tool by writing a single
<tool_call>…</tool_call> block as your entire reply — no prose before or
after, no additional text.

### search_documents(query: str, file_types: list[str] | None) -> list[Document]

Searches the user's personal corpus and returns up to N relevant chunks.
Each result has:
  - source_path: full NFS path (cite this in your answer)
  - score:       relevance score (0–1, higher = more relevant)
  - content:     chunk text (use this to answer)

Parameters:
  - query:      what to search for (required)
  - file_types: optional list — restrict to one or more file types.
                Valid values: "pdf", "docx", "pptx", "ppt", "txt", "md",
                "yaml", "ipynb", "claude_chat"
                Omit to search all content types.

Call this tool when you need information from the user's documents or past
sessions. DO NOT call it for greetings, identity questions, or general knowledge.

Example — general search:
<tool_call>
{"name": "search_documents", "arguments": {"query": "AWS certifications"}}
</tool_call>

Example — presentations only:
<tool_call>
{"name": "search_documents", "arguments": {"query": "quarterly review", "file_types": ["pptx", "ppt"]}}
</tool_call>

Example — past Claude Code sessions only:
<tool_call>
{"name": "search_documents", "arguments": {"query": "cost estimate bug", "file_types": ["claude_chat"]}}
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
    """Render retrieved chunks into a tool-role message body."""
    if not chunks:
        return "search_documents returned no results."
    lines = [f"search_documents returned {len(chunks)} result(s):\n"]
    for i, c in enumerate(chunks, 1):
        stitle = c.get("session_title", "")
        title_part = f"  title={stitle}\n" if stitle else ""
        lines.append(
            f"[{i}] score={c.get('score', 'N/A'):.4f}  source={c.get('source_path', 'N/A')}\n"
            f"{title_part}"
            f"{c.get('content', 'N/A')}\n"
        )
    return "\n".join(lines)