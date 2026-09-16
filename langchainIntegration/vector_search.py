import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Create the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create Vector
plot_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

# Search for similar movie plots
plot = "Toys come alive"
# result = plot_vector.similarity_search(plot, k=3)
# You can filter the results of the similarity_search method by using the filter parameter.
# The filter parameter allows you to specify a condition to filter the results, for example, only return movies with a
# revenue greater than 200 million:
result = plot_vector.similarity_search(
    plot,
    k=3,
    filter={"revenue": {"$gte": 200000000}}
)
print(result)

# Parse the documents
for doc in result:
    print(f"Title: {doc.metadata['title']}")
    print(f"Plot: {doc.page_content}\n")
