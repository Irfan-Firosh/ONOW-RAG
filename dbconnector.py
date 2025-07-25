import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from supabase import create_client, Client
import openai
import math

class SupabaseConnector:
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_ANON_KEY')
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and API key must be provided either as parameters "
                "or through environment variables SUPABASE_URL and SUPABASE_ANON_KEY"
            )
        
        self.client: Client = create_client(self.url, self.key)
    
    def insert_record(self, table_name: str, data: Dict[str, Any]) -> Dict:
        
        response = self.client.table(table_name).insert(data).execute()
        return response.data[0]
    
    def get_records(self, table_name: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict]:
        
        query = self.client.table(table_name).select("*")
        if limit is not None:
            query = query.limit(limit)
        if filters:
            for column, value in filters.items():
                if isinstance(value, (list, tuple)):
                    query = query.in_(column, value)
                else:
                    query = query.eq(column, value)
        response = query.execute()
        return response.data
    
    def update_record(self, table_name: str, record_id: int, data: Dict[str, Any]) -> Dict:
        
        response = self.client.table(table_name).update(data).eq('id', record_id).execute()
        return response.data[0]
    
    def delete_record(self, table_name: str, record_id: int) -> bool:
        
        response = self.client.table(table_name).delete().eq('id', record_id).execute()
        return len(response.data) > 0
    
    def search_vector(self, table_name: str, vector_column: str, query_embeddings: List[float], 
                     limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
        
        embeddings_json = json.dumps(query_embeddings)
        
        sql = f"""
        SELECT 
            *,
            {vector_column} <-> '{embeddings_json}'::vector as distance,
            1 - ({vector_column} <-> '{embeddings_json}'::vector) as similarity
        FROM {table_name}
        WHERE 1 - ({vector_column} <-> '{embeddings_json}'::vector) > {similarity_threshold}
        ORDER BY {vector_column} <-> '{embeddings_json}'::vector
        LIMIT {limit};
        """
        
        response = self.client.rpc('exec_sql', {'sql': sql}).execute()
        return response
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        
        # Get a sample record to understand structure
        response = self.client.table(table_name).select('*').limit(1).execute()
        
        if response.data:
            sample_record = response.data[0]
            return {
                'table_name': table_name,
                'columns': list(sample_record.keys()),
                'sample_record': sample_record
            }
        else:
            return {
                'table_name': table_name,
                'columns': [],
                'sample_record': {}
            }
    
    def list_tables(self) -> List[str]:
        
        sql = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        
        response = self.client.rpc('exec_sql', {'sql': sql}).execute()
        return [row['table_name'] for row in response.data]


    def embed_table_schemas_with_openai(self, table_names, embedding_model="text-embedding-ada-002"):
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = openai.OpenAI(api_key=api_key)
        for table_name in table_names:
            # Get a sample row to extract column names
            records = self.get_records(table_name, limit=1)
            if not records:
                print(f"No records found in {table_name}, skipping.")
                continue
            column_names = list(records[0].keys())
            schema_text = " | ".join(column_names)
            print(f"Embedding schema for {table_name}: {schema_text}")
            response = client.embeddings.create(input=[schema_text], model=embedding_model)
            embedding = response.data[0].embedding
            self.client.table('document_embeddings').insert({
                'document_id': f"{table_name}:schema",
                'content': schema_text,
                'embeddings': embedding,
                'metadata': {'table': table_name, 'type': 'schema'}
            }).execute()
            print(f"✅ Embedded schema for {table_name}")

    def embed_and_store(self, content: str, document_id: str, metadata: dict = None, embedding_model: str = "text-embedding-ada-002"):
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(input=[content], model=embedding_model)
        embedding = response.data[0].embedding
        self.client.table('document_embeddings').insert({
            'document_id': document_id,
            'content': content,
            'embeddings': embedding,
            'metadata': metadata or {}
        }).execute()
        print(f"✅ Embedded and stored: {document_id}")

    def search_document_embeddings(self, query_embedding: list, top_k: int = 5, similarity_threshold: float = 0.2) -> List[Dict]:
        """
        Search the document_embeddings table for the closest vectors to the query embedding.
        Returns a list of results ranked by cosine similarity (descending).
        """
        # Get all embeddings from the table
        response = self.client.table('document_embeddings').select('*').execute()
        all_embeddings = response.data
        
        if not all_embeddings:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for doc in all_embeddings:
            doc_embedding = doc['embeddings']
            table_name = doc['document_id'][:-7]  # Remove ":schema" suffix
        
            
            # Check if it's a string that needs to be parsed
            if isinstance(doc_embedding, str):
                try:
                    doc_embedding = json.loads(doc_embedding)
                except:
                    pass
            
            # Calculate cosine similarity manually
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            if similarity >= similarity_threshold:
                similarities.append({
                    **doc,
                    'similarity': float(similarity)
                })
        
        # Sort by similarity (descending) and return top_k results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        return dot_product / (magnitude1 * magnitude2)

# Utility function for easy access
def get_supabase_client(url: Optional[str] = None, key: Optional[str] = None) -> SupabaseConnector:
    
    return SupabaseConnector(url, key)

# Attach the function to SupabaseConnector
SupabaseConnector.embed_table_schemas_with_openai = SupabaseConnector.embed_table_schemas_with_openai

# Test main function
def main():
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    db = SupabaseConnector(supabase_url, supabase_key)

    # Assume you have a query string and want to search for similar schemas
    query = "seed type"
    response = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding

    results = db.search_document_embeddings(query_embedding, top_k=5, similarity_threshold=0)
    #print(results)
    for result in results:
        table_name = result["document_id"][:-7]  # Remove ":schema" suffix
        print(f"Table: {table_name}, Similarity: {result['similarity']}")

if __name__ == "__main__":
    main()
