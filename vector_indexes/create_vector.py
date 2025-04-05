""" You can now load the content and chunk it using Python and LangChain.

You will split the lesson content into chunks of text, around 1500 characters long, with each chunk
containing one or more paragraphs. You can determine the paragraph in the content with two newline
characters (\n\n)."""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

COURSES_PATH = "data/"

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob="lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

# Create a text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

# Split documents into chunks

# The content isn’t split simply by a character (\n\n) or on a fixed number of characters.
# The process is more complicated. Chunks should be up to maximum size but conform to the character split.
#
# In this example, the split_documents method does the following:
#
# Splits the documents into paragraphs (using the separator - \n\n)
# Combines the paragraphs into chunks of text that are up 1500 characters (chunk_size)
#
# if a single paragraph is longer than 1500 characters, the method will not split the paragraph but create a chunk
# larger than 1500 characters
# Adds the last paragraph in a chunk to the start of the next paragraph to create an overlap between chunks.
#
# if the last paragraph in a chunk is more than 200 characters (chunk_overlap) it will not be added to the next chunk
# This process ensures that:
#
# Chunks are never too small.
# That a paragraph is never split between chunks.
# That chunks are significantly different, and the overlap doesn’t result in a lot of repeated content.

chunks = text_splitter.split_documents(docs)

print(chunks)

# Create a Neo4j vector store

# Once you have chunked the content, you can use the LangChain Neo4jVector and OpenAIEmbeddings classes
# to create the embeddings, the vector index, and store the chunks in a Neo4j graph database.

# The Neo4jVector.from_documents method:
#
# Creates embeddings for each chunk using the OpenAIEmbeddings object.
#
# Creates nodes with the label Chunk and the properties text and embedding in the Neo4j database.
#
# Creates a vector index called chunkVector.

neo4j_db = Neo4jVector.from_documents(
    chunks,
    OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="neo4j",
    index_name="chunkVector",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
)