#!/bin/bash
# =============================================================================
# Gin AI — MCP Server Setup
# Run this on Server 1 to install and validate the MCP server
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║   Gin AI — MCP Server Setup          ║"
echo "╚══════════════════════════════════════════╝"
echo

# -- 1. Install MCP SDK -------------------------------------------------------
echo "→ Installing MCP Python SDK..."
pip install "mcp[cli]" pydantic --break-system-packages 2>/dev/null || \
pip install "mcp[cli]" pydantic

echo

# -- 2. Validate the server compiles -------------------------------------------
echo "→ Validating server syntax..."
python -m py_compile nfs_files_mcp_server.py
echo "  ✓ Server compiles OK"
echo

# -- 3. Show configured roots -------------------------------------------------
echo "→ Configured project root:"
echo "  GIN_AI_PROJECT_DIR=${GIN_AI_PROJECT_DIR:-$HOME/Gin-AI}"
echo

# -- 4. Test with MCP inspector (if available) --------------------------------
if command -v npx &> /dev/null; then
    echo "→ You can test interactively with:"
    echo "  npx @modelcontextprotocol/inspector python nfs_files_mcp_server.py"
    echo
fi

# -- 5. Print connection instructions -----------------------------------------
echo "═══════════════════════════════════════════"
echo "  CONNECT TO CLAUDE"
echo "═══════════════════════════════════════════"
echo
echo "Option A: Claude Code (recommended for dev)"
echo "  The .mcp.json file is already in the project root."
echo "  Just run 'claude' from this directory — it auto-detects it."
echo
echo "Option B: Claude Desktop"
echo "  Add the following to your Claude Desktop config:"
echo "    macOS: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo "    Linux: ~/.config/claude/claude_desktop_config.json"
echo
echo "  Merge the contents of claude_desktop_config.json into your config,"
echo "  replacing YOUR_USER with your actual username."
echo
echo "Option C: Remote HTTP (for cross-server access)"
echo "  python nfs_files_mcp_server.py --http --port 3001"
echo "  Then connect from Server 2 or any MCP client to:"
echo "    http://192.168.1.93:3001"
echo
echo "Option D: Connect to claude.ai (this chat)"
echo "  Run the server in HTTP mode (Option C above), then use"
echo "  the MCP connector in claude.ai settings to add it."
echo
echo "═══════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════"
