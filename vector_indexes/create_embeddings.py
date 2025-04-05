""" Publicly available Large Language Models (LLMs) will typically have an API that you can use to create embeddings for text.

    For example, OpenAI has an API that you can use to create embeddings for text."""

# You should be able to identify:
#
# The OpenAI class requires an API key to be passed to it.
# The llm.embeddings.create method is used to create an embedding for a piece of text.
# The text-embedding-ada-002 model is used to create the embedding.
# The response.data[0].embedding attribute is used to access the embedding.

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = llm.embeddings.create(
        input="Text to create embeddings for",
        model="text-embedding-ada-002"
    )

print(response.data[0].embedding)