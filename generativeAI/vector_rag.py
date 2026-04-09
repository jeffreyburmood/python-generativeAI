"""
This code file is from the Neo4j & GenerativeAI Fundamentals course.

(based on vector_retriever.py)

You can use a retriever as part of a RAG (Retrieval-Augmented Generation) pipeline to provide context to a LLM.

In this lesson, you will use the vector retriever you created to pass additional context to an LLM allowing it to
generate more accurate and relevant responses.

The program includes the code to connect to Neo4j and create the vector retriever.

You will add the code to:

    - Create and configure the LLM

    - Create the GraphRAG pipeline to use the vector retriever

    - Submit a query to the RAG pipeline

Parse the results

The GraphRAG pipeline will:

    - Use the retriever to find relevant context based on the user’s query.

    - Pass the user’s query and the retrieve context to the LLM.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

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

# Create the LLM
llm = OpenAILLM(model_name="gpt-5.2")


# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
query_text = "Find me movies about toys coming alive"

# You can also return the context that was used to generate the response. This can be useful in understanding how the LLM generated the response.
#
# Add the return_context=True parameter to the search method:
response = rag.search(
    query_text=query_text,
    retriever_config={"top_k": 5}
)

print(response.answer)


# Close the database connection
driver.close()