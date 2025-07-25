from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_retriever import index_tables, retrieve
from query_converter import query_decomposer
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates responses to queries based on retrieved documents. Use the following documents to answer the user's question:\n\n{documents}"),
    ("user", "{query}"),
])

def generator(query: str, documents: str) -> str:
    """
    Generate a response to a query based on retrieved documents.
    """
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "documents": documents})


def main(query: str):
    vectorstore = index_tables(["survey1", "survey2"])
    decomposed_queries = query_decomposer(query)
    
    all_documents = ""
    for decomposed_query in decomposed_queries:
        documents = retrieve(decomposed_query, vectorstore)
        all_documents += documents + "\n\n"
    
    final_response = generator(query, all_documents)
    print("Final Response:")
    print(final_response)