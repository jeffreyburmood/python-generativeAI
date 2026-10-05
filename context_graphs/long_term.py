"""Module 3 challenge - promote entities to long-term memory. Completed.

Long-term memory is a persistent knowledge graphA representation of real-world entities and their relationships, stored
according to organizing principles, typically in a graph database. of entities, relationshipsA named, directed
connection between two nodes. Every relationship has a type, a start node and an end node., and learned preferences.
Information extracted from conversations is automatically promoted to this layer and linked across sessions — it does
not reset when a conversation ends.

Characteristic	Long-Term Memory
Lifetime	Persistent — indefinite retention
Access pattern	Semantic search + graph traversalFollowing relationships from one node to the next to reach other parts of a graph.
Primary question	"What do you know about this entity?"

The library classifies entities using the POLE+O model — Person, Object, Location, Event, Organization. This is the
classification scheme built into the library; other ontologies exist, but the POLE+O model is the one the library uses.
 The five types provide broad coverage for the named entities that appear in most business conversations, and each type
  supports configurable subtypes (for example, Person/Customer, Location/Address) for domain-specific precision.

Every entity nodeA vertex in a graph. In a property graph it can carry labels and properties. carries the Entity label
A tag on a node that groups it with other nodes of the same kind. A node can carry more than one., plus a second label
for its type. That means you can match every entity with (:Entity), or narrow to one type with (:Entity&Person).

Entity Type	Examples	Neo4j Labels
Person	Customers, employees, stakeholders	:Entity:Person
Object	Accounts, products, documents, policies	:Entity:Object
Location	Addresses, regions, countries, branches	:Entity:Location
Event	Transactions, decisions, meetings, incidents	:Entity:Event
Organization	Companies, teams, departments, regulators	:Entity:Organization

Entities are connected by a single RELATED_TO relationship. The name of the relationship is held in its type property
A named value stored on a node or a relationship. rather than in the relationship type itself, so the library can
record any relationship an extractor discovers without changing the schema.

Example:
    To find who someone works for, you match the relationship and filter on its type:
    MATCH (p:Entity&Person {name: 'Jessica Norris'})-[r:RELATED_TO]->(org:Entity&Organization)
    WHERE r.type = 'WORKS_AT'
    RETURN p.name AS person, org.name AS organization, r.valid_from AS since, r.valid_until AS until

Storing learned preferences

Long-term memory also stores agent and user preferences as typed nodes, enabling personalization across sessions:
await memory.long_term.add_preference(
    category="communication",
    preference="Prefers concise responses",
    context="Confirmed during onboarding"
)
This creates a Preference node in the graph linked to the current user. On future sessions, the agent retrieves matching
preferences before generating a response — so a user who prefers concise answers gets shorter replies automatically,
without being asked again.

When you call add_message(), the library does more than store the message text — it automatically runs an entity
extraction pipeline that connects short-term and long-term memory.
The three stages are spaCy, GLiNER2, and LLMA model trained on text to predict the next token, and so to generate
language. fallback — each increasing in accuracy and cost.

This is what keeps the three memory layers connected: a user message that mentions "Jessica Norris" automatically
creates or merges an :Entity:Person nodeA vertex in a graph. In a property graph it can carry labels and properties.,
which can then be retrieved in future sessions.

Understanding the three-stage pipeline

Stage	Model	Capability	Cost
Stage 1	spaCy	Fast statistical Named Entity Recognition; strong on common types (persons, organizations, locations);
pre-trained	Low
Stage 2	GLiNER2	Zero-shot NER — can identify entity types not seen during training; better on domain-specific
vocabulary (financial instruments, medical terms)	Medium
Stage 3	LLM fallback	LLM-based extraction; most accurate for complex or ambiguous text; used when earlier stages
produce low-confidence results	High

The pipeline runs these stages in order and merges the results using a configurable strategy.

Configuring merge strategies

To configure which stages run and how their results are combined, pass an ExtractionConfig to MemorySettings:
Example:
from neo4j_agent_memory.extraction import ExtractionConfig

settings = MemorySettings(
    neo4j=Neo4jConfig(...),
    extraction=ExtractionConfig(
        stages=["spacy", "gliner", "llm"],
        merge_strategy="confidence",  # union | confidence | cascade
        deduplication=True,
        enrichment=True
    )
)
This configuration enables all three stages, applies the confidence merge strategy, and turns on both deduplication
and enrichment. The merge_strategy value controls how entities extracted by different stages are combined:

Strategy	Behavior
union	Include all entities found by any extractor — highest recall, may include duplicates
confidence	Weight entities by confidence score across extractors — balanced precision and recall
cascade	Use Stage 1; fall back to Stage 2 if empty; fall back to Stage 3 if still empty — fastest adequate result

Understanding deduplication and enrichment

Deduplication — resolves aliases. "Brian Chesky" and "Chesky" merge into the same :Entity:Person node using fuzzy and
semantic matching. This prevents entity graph fragmentation across sessions.
Enrichment — an optional step that performs a Wikipedia lookup to add structured metadata to entities, and geocoding
to add coordinates to :Entity:Location nodes.

Understanding when extraction runs

Extraction runs automatically when:

add_message() is called with role="user" or role="assistant"
The message content contains text long enough to extract entities from
You do not need to call the extraction pipeline explicitly — it runs as part of the message storage operation.

You can also add entities and preferences, and retrieve the knowledge graphA representation of real-world entities
and their relationships, stored according to organizing principles, typically in a graph database. directly using
the long-term memory API.

In this lesson, you will learn the six long-term memory methods and when to use each: add_entity(), add_preference(),
search_entities(), get_related_entities(), add_fact(), and search_preferences().

Storing entities and preferences

Use add_entity() to store a typed entity and add_preference() to record a user preference:
Example:
async with MemoryClient(settings) as memory:
    # Add a typed entity using the POLE+O classification
    entity, dedup = await memory.long_term.add_entity(
        name="Jessica Norris",
        entity_type="PERSON",
        subtype="CUSTOMER",
        description="High-value customer, flagged for compliance review April 2025",
        attributes={"risk_score": 0.415}
    )

    # Add a user or agent preference
    await memory.long_term.add_preference(
        category="communication",
        preference="Prefers concise responses",
        context="Confirmed during onboarding"
    )

Storing facts and searching preferences

Use add_fact() to store a temporal fact that links two entities across time, and search_preferences() to retrieve
stored preferences:
Example:
async with MemoryClient(settings) as memory:
    # Store a fact with temporal validity
    await memory.long_term.add_fact(
        subject="Jessica Norris",
        predicate="managed",
        obj="Acme Corp account",
        valid_from=datetime(2024, 1, 1),
        valid_until=datetime(2025, 3, 31)
    )

    # Retrieve preferences by category
    prefs = await memory.long_term.search_preferences(
        query="communication"
    )

Searching for and retrieving entities

Use search_entities() to find entities by semantic similarity and get_related_entities() to walk out from an entity
you already have:
Example:
async with MemoryClient(settings) as memory:
    # Semantic search across all entity types
    entities = await memory.long_term.search_entities(
        query="Jessica Norris accounts",
        limit=10
    )

    # Walk out from a known entity, two hops
    jessica = await memory.long_term.get_entity_by_name("Jessica Norris")
    related = await memory.long_term.get_related_entities(jessica, depth=2)

    for entity, relationship in related:
        print(relationship.type, entity.name)


For a whole-graph snapshot across all three memory layers, use memory.get_graph() on the client rather than the
long-term namespace.
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

MESSAGE = "I am Alex Mercer and I work at Ridgeline Labs."


async def main():
    async with MemoryClient(settings) as memory:
        # Extraction is on by default, so storing the message is enough to
        # promote the people and organizations it names into long-term memory.
        message = await memory.short_term.add_message(SESSION_ID, "user", MESSAGE)
        print(f"Stored message {message.id}")

        preference = await memory.long_term.add_preference(
            category="communication",
            preference="Prefers concise answers",
            context="Stated in the first conversation",
        )
        print(f"Stored preference {preference.id}")

        print("\nEntities in long-term memory matching 'Alex':")
        entities = await memory.long_term.search_entities(query="Alex", limit=5)
        for entity in entities:
            print(f"  {entity.name} ({entity.entity_type})")


asyncio.run(main())