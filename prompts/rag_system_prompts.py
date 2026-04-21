"""System prompts for the three RAG modes.

build_system_prompt(mode, tool_spec) is the single public function.
tool_spec is produced by rag_engine.tool_use.build_tool_spec() and
passed in by the caller (main.py) so this module stays import-free
of the rag_engine package.
"""
from schemas import RagMode

_BASE = "You are HomeAI-Lab's assistant. Be concise, accurate, and helpful."

_MODE_OFF = f"""{_BASE}

Answer from general knowledge."""

_MODE_ON = f"""{_BASE}

You have access to the user's personal document corpus via a search tool
(see the TOOLS section below). Call the tool when the question is likely
about the user's documents, projects, certifications, notes, or personal
information. For general questions (greetings, model identity, common
knowledge, code explanations unrelated to the user's corpus), answer
directly without calling the tool."""

_MODE_ONLY = f"""{_BASE}

You have access to the user's personal document corpus via a search tool
(see the TOOLS section below). For every factual question, you MUST call
the tool before answering, and your answer MUST be grounded strictly in
the tool results.

If the tool returns no relevant results, or the results do not answer the
question, you MUST reply EXACTLY with the following sentence and nothing
else:

    I don't have information about that in the provided context.

Do NOT use prior knowledge. Do NOT speculate. Do NOT fill gaps with general
information. This rule applies even if you are confident you know the answer
from training data."""

_PROMPTS: dict[str, str] = {
    RagMode.off:  _MODE_OFF,
    RagMode.on:   _MODE_ON,
    RagMode.only: _MODE_ONLY,
}


def build_system_prompt(mode: RagMode, tool_spec: str | None) -> str:
    """Compose the final system prompt for the given RAG mode.

    Args:
        mode:      RagMode enum value (off | on | only)
        tool_spec: TOOLS section string from build_tool_spec(), or None for off
    """
    base = _PROMPTS[mode]
    if mode == RagMode.off or tool_spec is None:
        return base
    return f"{base}\n\n{tool_spec}"
