"""
This code file is from the Neo4j & GenerativeAI Fundamentals course

A retriever is a component that takes unstructured data (typically a users query) and retrieves relevant data.

You will create a vector retriever that find similar movies based on a movie plot. The retriever will use the moviePlots vector index you used to search for similar movies using Cypher.

To find similar movies using a retriever you need to:

    - Connect to a Neo4j database

    - Create an embedder to convert users queries into vectors

    - Create a retriever that uses the moviePlots vector index

    - Use the retriever to search for similar movies using the users query

    - Parse the results
"""

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create retriever
retriever = VectorRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# Search for similar items
result = retriever.search(query_text="Toys coming alive", top_k=5)

# Parse results
for item in result.items:
    print(item.content, item.metadata["score"])

# Close the database connection
driver.close()