# RAG Retrieval System

A comprehensive Retrieval-Augmented Generation (RAG) system that implements advanced retrieval techniques including **Least-to-Most query conversion** and **semantic routing**. This system is designed to provide highly relevant document retrieval for your platform.

## Features

### 🔍 Advanced Query Processing
- **Least-to-Most Query Conversion**: Breaks down complex queries into simpler, searchable steps
- **Query Complexity Analysis**: Automatically determines when query conversion is needed
- **Intelligent Query Optimization**: Converts steps into optimized search queries

### 🛣️ Semantic Routing
- **Multi-Category Routing**: Routes queries to technical, general, or creative categories
- **Route-Specific Strategies**: Different retrieval strategies for each route type
- **Confidence-Based Routing**: Uses similarity thresholds for accurate routing
- **Intent Analysis**: Detects query intent (how-to, what-is, compare, etc.)

### 📚 Document Management
- **Local Document Database**: Stores documents in a local file system
- **Automatic Chunking**: Splits documents into overlapping chunks for better retrieval
- **Metadata Tracking**: Maintains comprehensive document and chunk metadata

### 🔍 Vector-Based Retrieval
- **FAISS Vector Store**: High-performance similarity search using FAISS
- **Semantic Embeddings**: Uses Sentence Transformers for semantic understanding
- **Route-Specific Filtering**: Applies different filters based on query route
- **Deduplication**: Removes duplicate results and re-ranks based on multiple factors

### 🗄️ Database Integration
- **Supabase Integration**: Full PostgreSQL database support with Supabase
- **Chat History Storage**: Persistent conversation history for users
- **Document Embeddings**: Vector storage for similarity search
- **CRUD Operations**: Complete Create, Read, Update, Delete functionality
- **Vector Similarity Search**: Advanced vector search with pgvector extension

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Converter │───▶│ Semantic Router │
└─────────────────┘    │  (Least-to-Most) │    │                 │
                       └──────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Retrieved      │◀───│   Vector Store   │◀───│  Route Strategy │
│  Documents      │    │   (FAISS)        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│ Document        │    │  Embedding       │
│ Database        │    │  Model           │
│ (Local Files)   │    │  (Sentence       │
└─────────────────┘    │  Transformers)   │
                       └──────────────────┘
```

## Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ONOW-RAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional):
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Quick Start

### Basic Usage

```python
from rag_retriever import RAGRetriever

# Initialize the retriever
retriever = RAGRetriever()

# Create sample data
retriever.create_sample_data()

# Perform a search
query = "What is machine learning?"
results = retriever.retrieve(query)

# Access results
for result in results['results']:
    print(f"Document: {result['metadata']['doc_title']}")
    print(f"Score: {result['similarity_score']}")
    print(f"Content: {result['chunk'][:200]}...")
```

### Advanced Usage

```python
# Complex query (triggers query conversion)
complex_query = "How do I implement a machine learning algorithm and what are the best practices for data preprocessing?"
results = retriever.retrieve(complex_query)

# Check query conversion
if results['metadata']['conversion_metadata']:
    print("Converted queries:")
    for query in results['metadata']['conversion_metadata']['converted_queries']:
        print(f"  - {query}")

# Check routing information
routing = results['metadata']['routing_result']
print(f"Route: {routing['route']} (confidence: {routing['confidence']})")

# Add your own documents
doc_id = retriever.add_document(
    title="My Custom Document",
    content="Your document content here...",
    metadata={"source": "custom", "category": "tutorial"}
)
```

## Configuration

The system is configured through the `config.py` file. Key settings include:

```python
class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-3.5-turbo"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Vector Database Configuration
    VECTOR_DB_PATH = "vector_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Semantic Routing Configuration
    ROUTING_THRESHOLDS = {
        "technical": 0.8,
        "general": 0.6,
        "creative": 0.7
    }
```

## Components

### 1. Query Converter (`query_converter.py`)
- Implements Least-to-Most prompting strategy
- Breaks complex queries into simple steps
- Converts steps into optimized search queries
- Falls back to simple splitting when API is unavailable

### 2. Semantic Router (`semantic_router.py`)
- Routes queries to appropriate categories (technical, general, creative)
- Uses semantic similarity with pre-defined route examples
- Provides route-specific retrieval strategies
- Analyzes query intent and complexity

### 3. Document Database (`document_database.py`)
- Manages local document storage
- Handles document addition, retrieval, and search
- Provides sample documents for testing
- Maintains document metadata

### 4. Vector Store (`vector_store.py`)
- FAISS-based vector similarity search
- Automatic document chunking and embedding
- Route-specific result filtering
- Persistent storage of embeddings and metadata

### 5. RAG Retriever (`rag_retriever.py`)
- Main orchestrator class
- Coordinates all components
- Handles result deduplication and ranking
- Provides comprehensive API

### 6. Database Connector (`dbconnector.py`)
- Supabase integration for PostgreSQL database
- Chat history storage and retrieval
- Document embeddings storage
- Vector similarity search with pgvector
- Complete CRUD operations

## Database Integration

### Supabase Setup

The system includes comprehensive Supabase integration for persistent storage. See [SUPABASE_SETUP.md](SUPABASE_SETUP.md) for detailed setup instructions.

#### Quick Database Setup

```python
from dbconnector import initialize_database

# Initialize database with tables and indexes
db = initialize_database()

# Store chat history
db.store_chat_history(
    user_id="user123",
    query="What is machine learning?",
    response="Machine learning is...",
    metadata={"topic": "ml"}
)

# Store document embeddings
embeddings = [0.1, 0.2, 0.3, ...]  # Your vector embeddings
db.store_document_embeddings(
    document_id="doc_001",
    content="Document content...",
    embeddings=embeddings,
    metadata={"category": "AI"}
)

# Vector similarity search
similar_docs = db.search_similar_documents(
    query_embeddings=embeddings,
    limit=5,
    similarity_threshold=0.7
)
```

#### Database Features

- **Chat History**: Persistent storage of user conversations
- **Document Embeddings**: Vector storage for similarity search
- **CRUD Operations**: Full database operations support
- **Vector Search**: Advanced similarity search with pgvector
- **Automatic Indexing**: Performance optimization
- **Row Level Security**: Production-ready security

## API Reference

### RAGRetriever Class

#### Methods

- `retrieve(query, use_query_conversion=True, use_semantic_routing=True)`: Main retrieval method
- `add_document(title, content, metadata=None)`: Add a new document
- `search_documents(query, limit=10)`: Simple document search
- `get_document(doc_id)`: Get a specific document
- `get_system_statistics()`: Get system statistics
- `create_sample_data()`: Create sample documents
- `clear_all_data()`: Clear all data

#### Return Format

The `retrieve()` method returns a dictionary with:

```python
{
    'query': str,                    # Original query
    'results': List[Dict],           # Retrieved document chunks
    'metadata': {
        'processing_time': float,    # Processing time in seconds
        'total_results': int,        # Number of results
        'unique_documents': int,     # Number of unique documents
        'complexity_analysis': Dict, # Query complexity analysis
        'conversion_metadata': Dict, # Query conversion details
        'routing_result': Dict,      # Semantic routing results
        'suggestions': List[str]     # Related query suggestions
    }
}
```

## Examples

Run the example script to see the system in action:

```bash
python example_usage.py
```

This will demonstrate:
- Simple queries
- Complex query conversion
- Semantic routing
- Document addition
- Feature toggling

## Performance Considerations

- **Embedding Model**: Uses `all-MiniLM-L6-v2` for fast, accurate embeddings
- **FAISS Index**: Optimized for similarity search performance
- **Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Caching**: Embeddings and indices are persisted for faster subsequent queries

## Extending the System

### Adding New Routes
1. Update `semantic_router.py` with new route examples
2. Add route-specific strategies
3. Update routing thresholds in `config.py`

### Custom Embedding Models
1. Change `EMBEDDING_MODEL` in `config.py`
2. Ensure the model is compatible with Sentence Transformers

### Additional Query Conversion Strategies
1. Extend `QueryConverter` class with new methods
2. Update the main retrieval pipeline in `RAGRetriever`

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**: The system will fall back to simple query splitting
2. **FAISS Installation Issues**: Use `pip install faiss-cpu` for CPU-only systems
3. **Memory Issues**: Reduce chunk size or use smaller embedding models

### Performance Tips

- Use GPU acceleration for embedding generation if available
- Adjust chunk size based on your document characteristics
- Tune similarity thresholds for your use case
- Consider using larger embedding models for better accuracy

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here] # ONOW-RAG
# ONOW-RAG
