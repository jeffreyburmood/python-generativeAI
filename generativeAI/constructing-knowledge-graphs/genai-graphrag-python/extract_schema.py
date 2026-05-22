import os
from dotenv import load_dotenv
load_dotenv()

from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
from neo4j_graphrag.llm import OpenAILLM
from rich import print
import asyncio

schema_extractor = SchemaFromTextExtractor(
    llm = OpenAILLM(
        model_name="gpt-5-nano",
    ),
    use_structured_output=True,
)

text = """
Neo4j is a graph database management system (GDBMS) developed by Neo4j Inc.
"""
more_text = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
"""

# Extract the schema from the text
extracted_schema = asyncio.run(schema_extractor.run(text=more_text))

print(extracted_schema)
