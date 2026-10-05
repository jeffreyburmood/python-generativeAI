"""Module 4 challenge - build and run the agent. Completed."""

import asyncio
import os

from dotenv import load_dotenv
from neo4j_agent_memory import ExtractionConfig, MemoryClient, MemorySettings
from neo4j_agent_memory.config import Neo4jConfig
from neo4j_agent_memory.integrations.pydantic_ai import (
    create_memory_tools,
    record_agent_trace,
)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

settings = MemorySettings(
    neo4j=Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    ),
    # The extraction pipeline runs spaCy and GLiNER before falling back to the
    # language model. Those two need extra packages (and a GPU-sized download),
    # so this course uses the LLM stage on its own. Without this, every stored
    # message logs two extractor failures before it succeeds.
    extraction=ExtractionConfig(enable_spacy=False, enable_gliner=False),
)

SESSION_ID = "graphacademy"

model = OpenAIChatModel(
    "gpt-4o-mini",
    provider=OpenAIProvider(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    ),
)

PROMPTS = [
    "My name is Alex and I prefer concise answers.",
    "What do you know about me?",
]


async def main():
    async with MemoryClient(settings) as memory:
        tools = create_memory_tools(memory)
        agent = Agent(model, tools=tools)

        for prompt in PROMPTS:
            print(f"\n> {prompt}")
            result = await agent.run(prompt)
            print(result.output)

            trace = await record_agent_trace(
                memory.reasoning, SESSION_ID, result, task=prompt
            )
            print(f"[recorded trace {trace.id}]")


asyncio.run(main())