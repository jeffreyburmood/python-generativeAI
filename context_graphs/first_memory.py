""" this code is from the Neo4j Academy - Context Graphs: Agent Memory with Neo4j course """

"""Module 1 challenge - store your first memory.

Complete the TODOs, then run:  python 1_first_memory.py
"""

import asyncio
import os

from dotenv import load_dotenv
from neo4j_agent_memory import ExtractionConfig, MemoryClient, MemorySettings
from neo4j_agent_memory.config import Neo4jConfig

load_dotenv()

NEO4J_URI='neo4j+s://f7c186b4.databases.neo4j.io:7687'
NEO4J_HTTP_URI='https://f7c186b4.databases.neo4j.io:443'
NEO4J_USERNAME='f7c186b4'
NEO4J_PASSWORD='DUo7jmk1wijCRfwWx1o_I1sxZuuxhFjxS--Cv0Y1lss'
NEO4J_DATABASE='f7c186b4'

settings = MemorySettings(
    neo4j=Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ["NEO4J_DATABASE"]
    ),
    # The extraction pipeline runs spaCy and GLiNER before falling back to the
    # language model. Those two need extra packages (and a GPU-sized download),
    # so this course uses the LLM stage on its own. Without this, every stored
    # message logs two extractor failures before it succeeds.
    extraction=ExtractionConfig(enable_spacy=False, enable_gliner=False),
)

# Every challenge in this course writes to the same session.
SESSION_ID = "graphacademy"


async def main():
    # TODO 1: Open a MemoryClient using `async with`
    #
    # TODO 2: Store one message with memory.short_term.add_message(), passing
    #         SESSION_ID, the role "user", and any content you like.
    #         Print the id of the message it returns.
    async with MemoryClient(settings) as memory:
        message = await memory.short_term.add_message(
            SESSION_ID, "user", "My name is Jeffrey and I like concise answers."
        )
        print(f"Stored message {message.id} in session {SESSION_ID}")


asyncio.run(main())