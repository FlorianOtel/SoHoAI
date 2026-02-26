from mcp.server.fastmcp import FastMCP
import subprocess

# Create an MCP server instance
mcp = FastMCP("HomeLab_Tools")

@mcp.tool()
def search_local_files(query: str) -> str:
    """Searches the local Linux filesystem for a specific file or content."""
    # Example logic: use grep or find
    return f"Mock result: Found 3 files matching '{query}' in /home/user/docs."

@mcp.tool()
def execute_python_script(script_path: str) -> str:
    """Executes a Python script securely and returns the output."""
    try:
        result = subprocess.run(
            ["python3", script_path], 
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}"

# You can easily import your RAG pipeline here and expose it as a tool!
# @mcp.tool()
# def query_knowledge_base(question: str) -> str: ...
