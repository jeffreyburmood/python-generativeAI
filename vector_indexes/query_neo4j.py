""" use the embedding to query the Neo4j chunkVector vector index """

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = llm.embeddings.create(
        input="What does Hallucination mean?",
        model="text-embedding-ada-002"
    )

embedding = response.data[0].embedding

# Connect to Neo4j
# First, import the LangChain Neo4jGraph class and create an object which will
# connect to the Neo4j sandbox
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Run query
result = graph.query("""
CALL db.index.vector.queryNodes('chunkVector', 6, $embedding)
YIELD node, score
RETURN node.text, score
""", {"embedding": embedding})

# Display results
# The embedding is passed to the query method as a key/value pair in a dictionary.
#
# Finally, iterate through the result and print the node.text and score values.
for row in result:
    print(row['node.text'], row['score'])