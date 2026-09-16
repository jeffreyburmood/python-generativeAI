import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate

# Cypher template
# You can provide examples of questions and relevant Cypher queries to help the LLM generate more accurate Cypher queries.
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, 
for example "The 39 Steps" becomes "39 Steps, The".

Schema:
{schema}
Examples:
1. Question: Get user ratings?
   Cypher: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE u.name = "User name" RETURN r.rating AS userRating
2. Question: Get average rating for a movie?
   Cypher: MATCH (m:Movie)<-[r:RATED]-(u:User) WHERE m.title = 'Movie Title' RETURN avg(r.rating) AS userRating
3. Question: Get movies for a genre?
   Cypher: MATCH ((m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = 'Genre Name' RETURN m.title AS movieTitle
   
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_template
)

# Initialize the LLM
cypher_model = init_chat_model("gpt-5.2", model_provider="openai")
# You can use different LLMs to generate the Cypher query and the answer.
# cypher_model = init_chat_model(
#     "gpt-5-mini",
#     model_provider="openai",
#     reasoning={"effort": "high"},
# )

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

# Create the Cypher QA chain
# cypher_qa = GraphCypherQAChain.from_llm(
#     graph=graph,
#     llm=cypher_model,
#     allow_dangerous_requests=True,
#     verbose=True,
# )
#
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=cypher_model,
    cypher_llm=cypher_model,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    verbose=True,
)
# You can restrict the schema by either providing the GraphCypherQAChain with a list of node labels and relationship
# types to include or exclude from the schema.
# Create the Cypher QA chain
# cypher_qa = GraphCypherQAChain.from_llm(
#     graph=graph,
#     llm=model,
#     include_types=["Movie", "ACTED_IN", "Person"],
#     allow_dangerous_requests=True,
#     verbose=True,
# )
#
# Alternatively, if you wanted to exclude ratings data, you could provide User and RATED as the types to the
# exclude_types parameter:
# # Create the Cypher QA chain
# cypher_qa = GraphCypherQAChain.from_llm(
#     graph=graph,
#     llm=model,
#     exclude_types=["User", "RATED"],
#     allow_dangerous_requests=True,
#     verbose=True,
# )

# Invoke the chain
question = "Who acted in the movie The Matrix?"
response = cypher_qa.invoke({"query": question})
print(response["result"])
