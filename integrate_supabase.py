#!/usr/bin/env python3
"""
Integration script showing how to use Supabase with the existing RAG system.
This script demonstrates storing chat history and document embeddings.
"""

import os
import streamlit as st
from dbconnector import SupabaseConnector, initialize_database
from generator import generator
from rag_retriever import indexing, retrieve
from query_converter import query_decomposer

class RAGWithSupabase:
    """
    Enhanced RAG system with Supabase integration for persistent storage.
    """
    
    def __init__(self):
        """Initialize the RAG system with Supabase integration."""
        self.vectorstore = None
        self.db = None
        self.initialize_supabase()
        self.initialize_vectorstore()
    
    def initialize_supabase(self):
        """Initialize Supabase connection."""
        try:
            # Get Supabase credentials from environment
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if supabase_url and supabase_key:
                self.db = initialize_database(supabase_url, supabase_key)
                st.success("✅ Supabase connection established!")
            else:
                st.warning("⚠️ Supabase credentials not found. Running without database storage.")
                self.db = None
                
        except Exception as e:
            st.error(f"❌ Failed to initialize Supabase: {e}")
            self.db = None
    
    def initialize_vectorstore(self):
        """Initialize the vector store."""
        if self.vectorstore is None:
            with st.spinner("Loading documents and creating embeddings..."):
                self.vectorstore = indexing()
            st.success("✅ Vector store initialized successfully!")
    
    def process_query(self, query: str, user_id: str = "default_user") -> str:
        """Process a query through the RAG system and store in database."""
        if self.vectorstore is None:
            return "Error: Vector store not initialized. Please wait for initialization to complete."
        
        try:
            # Decompose the query
            decomposed_queries = query_decomposer(query)
            
            # Collect documents from all decomposed queries
            all_documents = ""
            for decomposed_query in decomposed_queries:
                documents = retrieve(decomposed_query, self.vectorstore)
                all_documents += documents + "\n\n"
            
            # Generate response
            response = generator(query, all_documents)
            
            # Store in database if available
            if self.db:
                try:
                    metadata = {
                        "decomposed_queries": decomposed_queries,
                        "document_count": len(all_documents.split('\n\n')),
                        "query_type": "rag_query"
                    }
                    
                    self.db.store_chat_history(
                        user_id=user_id,
                        query=query,
                        response=response,
                        metadata=metadata
                    )
                    
                except Exception as e:
                    st.warning(f"⚠️ Failed to store chat history: {e}")
            
            return response
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            
            # Store error in database if available
            if self.db:
                try:
                    self.db.store_chat_history(
                        user_id=user_id,
                        query=query,
                        response=error_msg,
                        metadata={"error": True, "error_type": str(type(e))}
                    )
                except:
                    pass
            
            return error_msg
    
    def get_chat_history(self, user_id: str = "default_user", limit: int = 10):
        """Retrieve chat history for a user."""
        if self.db:
            try:
                return self.db.get_chat_history(user_id, limit)
            except Exception as e:
                st.error(f"Failed to retrieve chat history: {e}")
                return []
        return []
    
    def get_database_stats(self):
        """Get database statistics."""
        if self.db:
            try:
                return self.db.get_database_stats()
            except Exception as e:
                st.error(f"Failed to get database stats: {e}")
                return {}
        return {}

def main():
    """Main Streamlit application with Supabase integration."""
    st.set_page_config(
        page_title="RAG Chatbot with Supabase",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot with Supabase Integration")
    st.markdown("Ask questions about machine learning, Python programming, web development, data science, and artificial intelligence!")
    
    # Initialize RAG system with Supabase
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGWithSupabase()
    
    # Sidebar for database information
    with st.sidebar:
        st.header("🗄️ Database Status")
        
        if st.session_state.rag_system.db:
            st.success("✅ Connected to Supabase")
            
            # Show database stats
            if st.button("📊 Show Database Stats"):
                stats = st.session_state.rag_system.get_database_stats()
                if stats:
                    st.write("**Database Statistics:**")
                    for table, count in stats.items():
                        st.write(f"- {table}: {count} records")
                else:
                    st.write("No statistics available")
            
            # Show chat history
            if st.button("📚 Show Chat History"):
                history = st.session_state.rag_system.get_chat_history(limit=5)
                if history:
                    st.write("**Recent Chat History:**")
                    for entry in history:
                        with st.expander(f"Query: {entry['query'][:50]}..."):
                            st.write(f"**Query:** {entry['query']}")
                            st.write(f"**Response:** {entry['response'][:200]}...")
                            st.write(f"**Date:** {entry['created_at']}")
                else:
                    st.write("No chat history found")
        else:
            st.error("❌ Not connected to Supabase")
            st.info("Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables to enable database features.")
    
    # Main chat interface
    st.markdown("---")
    st.subheader("💬 Chat")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process query and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.process_query(prompt)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, LangChain, OpenAI, and Supabase")
    
    # Show database connection status
    if st.session_state.rag_system.db:
        st.success("💾 Chat history is being saved to Supabase database")
    else:
        st.info("💡 Enable Supabase to save chat history and get persistent storage")

if __name__ == "__main__":
    main() 