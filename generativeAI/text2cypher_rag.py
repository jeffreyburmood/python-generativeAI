"""
This code file is from the Neo4j & GenerativeAI Fundamentals course.

Vector and full text retrievers are great for finding relevant data based on semantic similarity or keyword matching.

To answer more specific questions, you may need to perform more complex queries to find data relating to specific nodes, relationships, or properties.

For example, you want to find:

The age of an actor.
    - Who acted in a movie.
    - Movie recommendations based on rating.
    - Text to Cypher retrievers allow you to convert natural language queries into Cypher queries that can be executed against the graph.

[User Query]
"What year was the movie Babe released?"
[Generated Cypher Query]
MATCH (m:Movie)
WHERE m.title = 'Babe'
RETURN m.released
[Cypher Result]
1995
[LLM Response]
"The movie Babe was released in 1995."

"""

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create Cypher LLM
t2c_llm=OpenAILLM(
    model_name="gpt-5-mini",
    model_params={
        "reasoning_effort": "high"
    }
)

# The TextToCypherRetriever will automatically read the whole graph schema from the database.
#
# You can provide a custom schema to the retriever if you want to limit the nodes, relationships and properties that
# are used to generate Cypher queries. Limiting the scope of the schema can help improve the accuracy of the generated
# Cypher queries, particularly if the graph contains a lot of nodes and relationships.

# # Specify your own Neo4j schema
# neo4j_schema = """
# Node properties:
# Person {name: STRING, born: INTEGER}
# Movie {tagline: STRING, title: STRING, released: INTEGER}
# Genre {name: STRING}
# User {name: STRING}
#
# Relationship properties:
# ACTED_IN {role: STRING}
# RATED {rating: INTEGER}
#
# The relationships:
# (:Person)-[:ACTED_IN]->(:Movie)
# (:Person)-[:DIRECTED]->(:Movie)
# (:User)-[:RATED]->(:Movie)
# (:Movie)-[:IN_GENRE]->(:Genre)
# """

# Add the schema to the TextToCypherRetriever:
#
# You will need to create the TextToCypherRetriever retriever that will generate Cypher queries and return results.
# # Build the retriever
# retriever = Text2CypherRetriever(
#     driver=driver,
#     neo4j_database=os.getenv("NEO4J_DATABASE"),
#     llm=t2c_llm,
#     neo4j_schema=neo4j_schema,
#     examples=examples,
# )

# The TextToCypherRetriever requires an LLM to generate the Cypher queries:
# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    llm=t2c_llm,
)

llm = OpenAILLM(model_name="gpt-5.2")
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = "Which movies did Hugo Weaving acted in?"
query_text = "What are examples of Action movies?"

response = rag.search(
    query_text=query_text,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()