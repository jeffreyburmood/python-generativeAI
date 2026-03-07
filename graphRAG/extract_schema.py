""" This code is form the Neo4j Workshop: GenAI and Graphs """

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
from neo4j_graphrag.llm import OpenAILLM
import asyncio

schema_extractor = SchemaFromTextExtractor(
    llm=OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0}
    )
)

text = """
Large Language Models (LLMs) are a type of artificial intelligence model designed to understand and generate human-like text.
"""

# Extract the schema from the text
extracted_schema = asyncio.run(schema_extractor.run(text=text))

print(extracted_schema)
