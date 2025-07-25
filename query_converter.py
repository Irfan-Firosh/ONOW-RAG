import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


def query_decomposer(query: str) -> str:
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    return chain.invoke({"question": query})


if __name__ == "__main__":
    query = "How do I implement a machine learning algorithm and what are the best practices for data preprocessing?"
    print(query_decomposer(query)[0])