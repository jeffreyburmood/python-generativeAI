"""Module 2 challenge - build a conversation memory layer. Completed."

What short-term memory provides

Short-term memory solves this with three capabilities:

Context injection — the most recent messages are retrieved and injected into each new prompt automatically. The agent
always has the current conversation context.
Semantic search — past messages are embedded and indexed. You can find messages that discuss the same topic even if t
hey use different words.
Graph connectivity — messages are linked to the entities mentioned in them and to the reasoning traces they triggered.
You can traverseFollowing relationships from one node to the next to reach other parts of a graph. from a message to what the agent knew and what it decided.
Understanding the session boundary

Short-term memory is scoped to a session. When the session ends, the conversation history is no longer the agent's
active working memory — but the entities extracted from it persist in long-term memory. The session ends; the
knowledge remains.

Short-term memory has three defining characteristics that distinguish it from the other memory layers.

Characteristic	Short-Term Memory
Lifetime	Ephemeral — exists for the duration of a session
Access pattern	Sequential — recent messages first
Primary question	"What did you discuss?"

A Conversation node acts as the entry point and links to its messages through a chain of FIRST_MESSAGE and NEXT_MESSAGE
relationshipsA named, directed connection between two nodes. Every relationship has a type, a start node and an end node.

The Conversation also has a HAS_MESSAGE relationship to every message it contains. FIRST_MESSAGE and NEXT_MESSAGE give
you the order; HAS_MESSAGE gives you the whole set without walking the chain.

Each Message node stores:

role — user, assistant, or system
content — the message text
timestamp — when the message was created
embeddingInformation represented as a numerical vector, positioned so that similar information sits close together.
— a vector used for semantic search across message history

The session_id is a property of the Conversation, not of each Message. To find the messages for a session you match the
conversation first:

MATCH (c:Conversation {session_id: 'demo_fraud_review'})-[:HAS_MESSAGE]->(m:Message)
RETURN m.role AS role, m.content AS content, m.timestamp AS timestamp
ORDER BY m.timestamp

When you add a message to short-term memory, the library performs these steps automatically:

1. Creates the Message node and links it into the Conversation chain
2. Runs the entity extraction pipeline on the message content
3. Promotes discovered entities into the long-term memory layer
4. Links the message to each entity it found with a MENTIONS relationship

This automatic extraction connects short-term and long-term memory within the same graph.

A session corresponds to one conversation. You choose the session ID — any string works. The library creates the
Conversation nodeA vertex in a graph. In a property graph it can carry labels and properties. automatically the
first time you call add_message() with that ID:
Example:
    async with MemoryClient(settings) as memory:
    # Add messages — the session is created automatically on first use
    await memory.short_term.add_message(
        session_id="user_123",
        role="user",
        content="Review Jessica Norris account for credit limit increase"
    )
    await memory.short_term.add_message(
        session_id="user_123",
        role="assistant",
        content="I will retrieve Jessica's full profile now."
    )

    # Remove a session and all its messages when done
    await memory.short_term.clear_session("user_123")

Calling add_message() with a new session_id creates the Conversation node automatically. clear_session() removes the
session and all its linked Message nodes when the conversation ends.

What add_message does beyond storage

When you call add_message(), the library runs an entity extraction pipeline on the message content. Any named entities
the pipeline finds — people, organizations, locations, events, objects — are automatically created or merged as entity
nodesA vertex in a graph. In a property graph it can carry labels and properties. in the long-term memory graph.
MENTIONS AND EXTRACTED_FROM
MENTIONS is written for you every time extraction runs. There is a second, opt-in edge in the other direction,
(Entity)-[:EXTRACTED_FROM]->(Message), which you write yourself with link_entity_to_message() when you want to record
provenance explicitly.

To retrieve a conversation and its messages, or search message history by semantic similarity:
async with MemoryClient(settings) as memory:
    # Retrieve the full conversation and its messages
    conversation = await memory.short_term.get_conversation("user_123")
    for msg in conversation.messages:
        print(msg.role, msg.content)

    # Semantic search over message history
    results = await memory.short_term.search_messages(
        query="credit limit increase",
        session_id="user_123",
        limit=10
    )

get_conversation() returns the Conversation object with all linked messages in order. search_messages() uses vector
similarity on the embedding propertyA named value stored on a node or a relationship. of each Message node.

To understand the linked-list structure the library builds, write the equivalent query manually:
MATCH (c:Conversation {session_id: 'demo_fraud_review'})-[:FIRST_MESSAGE]->(first:Message)
CALL (first) {
  MATCH (first)-[:NEXT_MESSAGE*0..]->(msg:Message)
  RETURN msg ORDER BY msg.timestamp DESC LIMIT 5
}
RETURN msg.role AS role, msg.content AS content, msg.timestamp AS timestamp

Starting from FIRST_MESSAGE and following NEXT_MESSAGE traverses the linked list in order. This traversalFollowing
relationships from one node to the next to reach other parts of a graph. is structurally impossible in a flat table
without a sort column — and the pattern becomes more capable when combined with entity lookups in Module 3.

The session id is matched on the Conversation, because that is where it is stored.

Use list_sessions() to retrieve all sessions and get_conversation_summary() to generate a summary of what was discussed:
async with MemoryClient(settings) as memory:
    # List all sessions
    sessions = await memory.short_term.list_sessions()

    # Auto-generate a plain-English summary of the conversation
    summary = await memory.short_term.get_conversation_summary("user_123")

list_sessions() returns all Conversation nodes, useful for building session history views in a UI.
get_conversation_summary() uses the stored messages to produce a concise summary of the conversation — useful for
creating long-term memory from short-term context.
"""

import asyncio
import os

from dotenv import load_dotenv
from neo4j_agent_memory import ExtractionConfig, MemoryClient, MemorySettings
from neo4j_agent_memory.config import Neo4jConfig

load_dotenv()

settings = MemorySettings(
    neo4j=Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ["NEO4J_DATABASE"],
    ),
    # The extraction pipeline runs spaCy and GLiNER before falling back to the
    # language model. Those two need extra packages (and a GPU-sized download),
    # so this course uses the LLM stage on its own. Without this, every stored
    # message logs two extractor failures before it succeeds.
    extraction=ExtractionConfig(enable_spacy=False, enable_gliner=False),
)

SESSION_ID = "graphacademy"

EXCHANGE = [
    ("user", "My name is Jeffrey and I want concise answers."),
    ("assistant", "Noted, Jeffrey. I will keep my answers short."),
    ("user", "What do you know about me?"),
]


async def main():
    async with MemoryClient(settings) as memory:
        for role, content in EXCHANGE:
            message = await memory.short_term.add_message(SESSION_ID, role, content)
            print(f"Stored {role} message {message.id}")

        print(f"\nConversation for session {SESSION_ID}:")
        conversation = await memory.short_term.get_conversation(SESSION_ID)
        for message in conversation.messages:
            print(f"  {message.role}: {message.content}")


asyncio.run(main())