""" """
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
from neo4j import GraphDatabase
from textblob import TextBlob


COURSES_PATH = "data/"

loader = DirectoryLoader(COURSES_PATH, glob="lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)

# Create a function to get the embedding
#For each chunk, you have to create an embedding of the text and extract the metadata.

def get_embedding(llm, text):
    response = llm.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
    return response.data[0].embedding

# Create a function to get the course data
# Splits the document source path to extract the course, module, and lesson names
#
# Constructs the url using the extracted names
#
# Extracts the text from the chunk
#
# Creates an embedding using the get_embedding function
#
# Returns a dictionary containing the extracted data

# By adding topics to the graph, you can use them to find related content.
#
# Topics are also universal and can be used to find related content across content from different
# sources. For example, if you added technical documentation to this graph, you could use the topics
# to find related lessons and documentation.
#
# Combining data from different sources and understanding their relationships is the starting point
# for creating a knowledge graph.

def get_course_data(llm, chunk):
    data = {}

    path = chunk.metadata['source'].split(os.path.sep)

    # data['course'] = path[-6]
    # data['module'] = path[-4]
    # data['lesson'] = path[-2]
    data['course'] = 'llm-fundamentals'
    data['module'] = '1-introduction'
    data['lesson'] = '1-neo4j-and-genai'
    data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    data['text'] = chunk.page_content
    data['embedding'] = get_embedding(llm, data['text'])
    data['topics'] = TextBlob(data['text']).noun_phrases

    return data

# To create the graph, you will need to:
#
# Create an OpenAI object to generate the embeddings
#
# Connect to the Neo4j database
#
# Iterate through the chunks
#
# Extract the course data from each chunk
#
# Create the nodes and relationships in the graph

# Create OpenAI object
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(
        os.getenv('NEO4J_USERNAME'),
        os.getenv('NEO4J_PASSWORD')
    )
)
driver.verify_connectivity()

# Create a function to run the Cypher query
def create_chunk(tx, data):
    tx.run("""
        MERGE (c:Course {name: $course})
        MERGE (c)-[:HAS_MODULE]->(m:Module{name: $module})
        MERGE (m)-[:HAS_LESSON]->(l:Lesson{name: $lesson, url: $url})
        MERGE (l)-[:CONTAINS]->(p:Paragraph{text: $text})
        WITH p
        CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)

        FOREACH (topic in $topics |
            MERGE (t:Topic {name: topic})
            MERGE (p)-[:MENTIONS]->(t)
        )
        """,
           data
        )

# Iterate through the chunks and create the graph
for chunk in chunks:
    with driver.session(database="neo4j") as session:
        session.execute_write(
            create_chunk,
            get_course_data(llm, chunk)
        )

# Close the neo4j driver
driver.close()