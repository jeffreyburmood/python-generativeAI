""" This code shows how to use a retriever in a langchain to retrieve unstructured data
    to use when generating a response to a query. The approach generates embeddings for
     a vector store to use in generating the response """

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

OPENAI_API_KEY = "sk-..."

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="pleaseletmein"
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)

response = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(response)