from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from dbconnector import SupabaseConnector
import os
import openai
from langchain.schema import Document
import json
from typing import Optional

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def indexing_txt():
    loader = DirectoryLoader(
        "docs/",
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore

def index_table(table_name):
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    db = SupabaseConnector(supabase_url, supabase_key)
    
    records = db.get_records(table_name, limit=1)
    if not records:
        print(f"No records found in {table_name}, cannot get column names.")
        return
    
    column_names = list(records[0].keys())
    schema_text = " | ".join(column_names)
    
    schema_doc = Document(
        page_content=schema_text,
        metadata={"table": table_name, "type": "sql_schema", "source": f"{table_name}:schema"}
    )
    return schema_doc

def index_tables(tables):
    persist_directory = "chroma_db"
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=persist_directory)
    for table in tables:
        vectorstore.add_documents([index_table(table)])
    vectorstore.persist()
    return vectorstore

def retrieve(query: str, vectorstore: Chroma, top_tables: int = 2, record_limit: Optional[int] = None):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    documents = retriever.invoke(query)

    unique_tables = []
    for doc in documents:
        table_name = doc.metadata.get("table")
        if table_name and table_name not in unique_tables:
            unique_tables.append(table_name)
        if len(unique_tables) >= top_tables:
            break

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    db = SupabaseConnector(supabase_url, supabase_key)

    output_chunks = []
    for table in unique_tables:
        try:
            records = db.get_records(table, limit=record_limit)
            formatted_records = json.dumps(records, indent=2)
            output_chunks.append(f"Table: {table}\nRecords (limit {record_limit}):\n{formatted_records}")
        except Exception as exc:
            output_chunks.append(f"Table: {table}\nError fetching data: {exc}")

    return "\n\n".join(output_chunks)

if __name__ == "__main__":
    vectorstore = index_tables(["survey1", "survey2"])
    result = retrieve("Give me the seed order", vectorstore)
    print(result)

