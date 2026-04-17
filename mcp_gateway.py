"""
MCP (Model Context Protocol) Gateway — Phase 3.

Routes tool calls from the LLM to appropriate MCP servers.
Placeholder with interface defined.

Architecture:
  LLM generates tool_call → orchestrator parses it →
  MCP client sends to MCP server → result fed back to LLM

Planned MCP servers:
  - filesystem (read/write files on NAS)
  - web_search (web lookup)
  - calendar / email (optional integrations)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPGateway:
    """Proxy for MCP tool servers."""

    def __init__(self, config: dict):
        self.config = config
        self.servers: dict[str, Any] = {}
        logger.info("MCP gateway initialized (stub — Phase 3)")

    async def register_server(self, name: str, server_config: dict):
        """Register an MCP server endpoint."""
        # Phase 3: use mcp SDK to connect
        raise NotImplementedError("MCP registration coming in Phase 3")

    async def call_tool(self, server: str, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call on a registered MCP server."""
        raise NotImplementedError("MCP tool calls coming in Phase 3")

    async def list_tools(self) -> list[dict]:
        """List all available tools across registered servers."""
        raise NotImplementedError("MCP tool listing coming in Phase 3")
