# ONOW-RAG

A Retrieval-Augmented Generation system that connects to Supabase databases to retrieve and analyze table data through natural language queries.

## Overview

This system allows you to query database tables using natural language. It uses vector embeddings to understand table schemas and retrieves relevant data from your Supabase database.

## Features

- Natural language querying of database tables
- Vector-based schema matching using Chroma
- Supabase integration for data retrieval
- Streamlit web interface
- Query decomposition for complex questions
- Configurable data retrieval limits

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ONOW-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Usage

### Command Line

```python
from rag_retriever import index_tables, retrieve

# Index your tables
vectorstore = index_tables(["survey1", "survey2"])

# Query the data
result = retrieve("Give me the seed order", vectorstore)
print(result)
```

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

## Components

- `rag_retriever.py`: Core retrieval logic and database indexing
- `dbconnector.py`: Supabase connection and data operations
- `generator.py`: Response generation using OpenAI
- `query_converter.py`: Query decomposition and processing
- `app.py`: Streamlit web interface

## Configuration

The system automatically indexes table schemas and stores them in a local Chroma vector database. You can configure:

- Number of top tables to retrieve (default: 2)
- Record limit per table (default: 100, use None for all records)
- Vector similarity thresholds
- OpenAI model settings

## Database Setup

Ensure your Supabase database has the required tables. The system will automatically extract column names from existing data to create schema embeddings.

## Architecture

The system works by:
1. Indexing table schemas as vector embeddings
2. Converting natural language queries to vector representations
3. Finding the most similar table schemas
4. Retrieving actual data from the matched tables
5. Formatting and returning the results
