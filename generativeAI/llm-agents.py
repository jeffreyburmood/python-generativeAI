""" An example of creating LLM agents that use Tools to help with processing a prompt """

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from uuid import uuid4
from langchain_community.tools import YouTubeSearchTool

youtube = YouTubeSearchTool()

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

llm = ChatOpenAI(openai_api_key="sk-...")

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="pleaseletmein"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. You find movies from a genre or plot.",
        ),
        ("human", "{input}"),
    ]
)

movie_chat = prompt | llm | StrOutputParser()


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Tools are interfaces that an agent can interact with. You can create custom tools
# able to perform any functionality you want.
#
# In this example, the Tool is created from a function. The function is the movie_chat.invoke method.
def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)

tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

# Agents support multiple tools, so you pass them to the agent as a list (tools).
# There are different types of agents that you can create. This example creates a
# ReAct - Reasoning and Acting) agent type.

# An agent requires a prompt. You could create a prompt, but in this example,
# the program pulls a pre-existing prompt from the Langsmith Hub.

# The hwcase17/react-chat prompt instructs the model to provide an answer using the tools
# available in a specific format.

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    q = input("> ")

    response = chat_agent.invoke(
        {
            "input": q
        },
        {"configurable": {"session_id": SESSION_ID}},
    )

    print(response["output"])