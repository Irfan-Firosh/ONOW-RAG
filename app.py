import streamlit as st
from generator import generator
from rag_retriever import index_tables, retrieve
from query_converter import query_decomposer
import os

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_vectorstore():
    if st.session_state.vectorstore is None:
        with st.spinner("Loading documents and creating embeddings..."):
            st.session_state.vectorstore = index_tables(["survey1", "survey2"])
        st.success("Vector store initialized successfully!")

def process_query(query: str) -> str:
    if st.session_state.vectorstore is None:
        return "Error: Vector store not initialized. Please wait for initialization to complete."
    
    try:
        decomposed_queries = query_decomposer(query)
        
        all_documents = ""
        for decomposed_query in decomposed_queries:
            documents = retrieve(decomposed_query, st.session_state.vectorstore)
            all_documents += documents + "\n\n"
        
        response = generator(query, all_documents)
        return response
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f1f1f;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stChatMessage {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 1rem;
    }
    .stButton > button {
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)

initialize_vectorstore()

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(prompt)
            st.write(response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)
