""" this example is from the Building GraphRAG Python MCP Tools Neo4j course

    https://github.com/neo4j-graphacademy/genai-mcp-build-custom-tools-python """

from fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Strawberry", instructions="we are learning stuff here")

@mcp.tool()
def count_letters(text: str, search: str) -> int:
    """Count occurrences of a letter in the text"""
    return text.lower().count(search.lower())

@mcp.prompt()
def list_fruits_prompt() -> str:
    """List all fruits"""
    return "Use the fruits resource to get all fruits"

@mcp.resource("cupboard://fruits")
def fruits() -> list[str]:
    """Get all fruits"""
    return ["apple", "strawberry", "banana"]

@mcp.resource("cupboard://fruits/{name}")
def fruit(name: str) -> dict[str, str]:
    """Get a fruit by name"""
    return {
        "name": name,
        "color": "yellow" if __import__("random").random() < 0.5 else "red",
        "taste": "sour" if __import__("random").random() < 0.5 else "sweet",
    }

# Run the server when executed directly
if __name__ == "__main__":
    mcp.run(
      transport="streamable-http"
    )
