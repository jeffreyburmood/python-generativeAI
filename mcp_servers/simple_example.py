""" this example is from the Building GraphRAG Python MCP Tools Neo4j course

    https://github.com/neo4j-graphacademy/genai-mcp-build-custom-tools-python """

from fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Text Analysis Server")  # (1)


# Tool for deterministic counting
@mcp.tool()  # (2)
def count_letters(text: str, search: str) -> int:
    """Count occurrences of a letter in the text."""
    return text.lower().count(search.lower())


# Resource for reference data
@mcp.resource("greeting://{who}")  # (3)
def get_greeting(who: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {who}!"


# Prompt template for common task
@mcp.prompt(title="Count Letters")  # (4)
def count_letters_prompt(text: str, search: str) -> str:
    """Template for letter counting task."""
    return f"Count the occurrences of the letter '{search}' in the text:\n\n{text}"


# ...

if __name__ == "__main__":
    mcp.run()

# run the server using stdio transport:
# fastmcp run simple_example.py

# run the server using HTTP transport:
# change run command to:
# mcp.run(
#     transport="streamable-http",
#     host="127.0.0.1",
#     port=8000,
#     path="/mcp"
# )
#
# fastmcp run server.py --transport http --host 127.0.0.1 --port 8000