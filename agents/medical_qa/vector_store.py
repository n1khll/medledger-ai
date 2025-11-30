"""Vector store operations using Supabase"""
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from llama_index.core import StorageContext
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

class SupabaseVectorStoreManager:
    """Manages vector storage operations in Supabase using direct PostgreSQL connection"""
    
    def __init__(self):
        """Initialize Supabase client and database connection"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.db_url = os.getenv("SUPABASE_DB_URL")
        self.table_name = os.getenv("VECTOR_COLLECTION_NAME", "medical_knowledge_vectors")
        
        if not all([self.supabase_url, self.supabase_key, self.db_url]):
            raise ValueError(
                "Missing Supabase configuration. Check SUPABASE_URL, SUPABASE_SERVICE_KEY, and SUPABASE_DB_URL"
            )
        
        # Initialize Supabase client (for REST API operations)
        self.supabase_client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Parse connection string for direct PostgreSQL connection
        self._parse_db_url()
        
        logger.info(f"Supabase vector store initialized: {self.table_name}")
    
    def _parse_db_url(self):
        """Parse database URL to extract connection parameters"""
        # Parse postgresql://user:pass@host:port/dbname
        import urllib.parse
        parsed = urllib.parse.urlparse(self.db_url)
        self.db_config = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip('/'),
            "user": parsed.username,
            "password": parsed.password
        }
    
    def _get_db_connection(self):
        """Get PostgreSQL connection for vector operations"""
        return psycopg2.connect(**self.db_config)
    
    def get_storage_context(self) -> StorageContext:
        """Get storage context for LlamaIndex (for compatibility)"""
        # Return a basic storage context - we'll handle vector operations directly
        return StorageContext.from_defaults()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            # Query Supabase table directly via REST API
            response = self.supabase_client.table(self.table_name).select("id", count="exact").execute()
            total_chunks = response.count if hasattr(response, 'count') else len(response.data) if response.data else 0
            
            # Get unique documents
            response = self.supabase_client.table(self.table_name).select("document_name").execute()
            unique_docs = len(set([row.get('document_name') for row in (response.data or []) if row.get('document_name')]))
            
            return {
                "total_chunks": total_chunks,
                "total_documents": unique_docs,
                "collection_name": self.table_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": self.table_name
            }
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of similar documents with content and metadata
        """
        try:
            logger.info(f"[VECTOR] Connecting to Supabase database for vector search...")
            conn = self._get_db_connection()
            register_vector(conn)
            cursor = conn.cursor()
            
            # First, check how many total documents we have
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_count = cursor.fetchone()[0]
            logger.info(f"[VECTOR] Total documents in {self.table_name}: {total_count}")
            
            # Force sequential scan to ensure results are returned (avoid index scan issues)
            cursor.execute("SET enable_indexscan = off;")
            cursor.execute("SET enable_seqscan = on;")
            
            # Convert to embedding list (pgvector handles serialization)
            # embedding_str = '[' + ','.join(f'{val:.10f}' for val in query_embedding) + ']'
            
            # Perform cosine similarity search
            query = f"""
                SELECT 
                    id,
                    content,
                    metadata,
                    document_name,
                    chunk_index,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            logger.info(f"[VECTOR] Executing vector similarity search...")
            logger.info(f"[VECTOR] Executing vector similarity search...")
            cursor.execute(query, (query_embedding, query_embedding, top_k))
            results = cursor.fetchall()
            logger.info(f"[VECTOR] Vector search query executed. Returned {len(results)} rows")
            
            cursor.close()
            conn.close()
            
            # Format results
            formatted_results = []
            for i, row in enumerate(results):
                similarity_score = float(row[5]) if row[5] is not None else 0.0
                logger.info(f"[VECTOR] Row {i+1}: doc={row[3]}, chunk={row[4]}, similarity={similarity_score:.4f}")
                formatted_results.append({
                    "id": str(row[0]),
                    "content": row[1],
                    "metadata": row[2] if row[2] else {},
                    "document_name": row[3],
                    "chunk_index": row[4],
                    "similarity": similarity_score
                })
            
            if not formatted_results:
                logger.warning(f"[VECTOR] WARNING: No results returned from vector search. Check if documents are indexed.")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[VECTOR] ERROR in vector search: {str(e)}", exc_info=True)
            return []
    
    def load_index(self):
        """Load existing vector store index for LlamaIndex compatibility"""
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core import Document
            from llama_index.core.node_parser import SimpleNodeParser
            
            # Create a custom vector store that uses our Supabase implementation
            # For now, return None and we'll handle retrieval directly in tools
            logger.info("Using direct Supabase vector search (no LlamaIndex index needed)")
            return None
        except Exception as e:
            logger.warning(f"Could not load index: {str(e)}")
            return None


from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from llama_index.core import StorageContext
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

class SupabaseVectorStoreManager:
    """Manages vector storage operations in Supabase using direct PostgreSQL connection"""
    
    def __init__(self):
        """Initialize Supabase client and database connection"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.db_url = os.getenv("SUPABASE_DB_URL")
        self.table_name = os.getenv("VECTOR_COLLECTION_NAME", "medical_knowledge_vectors")
        
        if not all([self.supabase_url, self.supabase_key, self.db_url]):
            raise ValueError(
                "Missing Supabase configuration. Check SUPABASE_URL, SUPABASE_SERVICE_KEY, and SUPABASE_DB_URL"
            )
        
        # Initialize Supabase client (for REST API operations)
        self.supabase_client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Parse connection string for direct PostgreSQL connection
        self._parse_db_url()
        
        logger.info(f"Supabase vector store initialized: {self.table_name}")
    
    def _parse_db_url(self):
        """Parse database URL to extract connection parameters"""
        # Parse postgresql://user:pass@host:port/dbname
        import urllib.parse
        parsed = urllib.parse.urlparse(self.db_url)
        self.db_config = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip('/'),
            "user": parsed.username,
            "password": parsed.password
        }
    
    def _get_db_connection(self):
        """Get PostgreSQL connection for vector operations"""
        return psycopg2.connect(**self.db_config)
    
    def get_storage_context(self) -> StorageContext:
        """Get storage context for LlamaIndex (for compatibility)"""
        # Return a basic storage context - we'll handle vector operations directly
        return StorageContext.from_defaults()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            # Query Supabase table directly via REST API
            response = self.supabase_client.table(self.table_name).select("id", count="exact").execute()
            total_chunks = response.count if hasattr(response, 'count') else len(response.data) if response.data else 0
            
            # Get unique documents
            response = self.supabase_client.table(self.table_name).select("document_name").execute()
            unique_docs = len(set([row.get('document_name') for row in (response.data or []) if row.get('document_name')]))
            
            return {
                "total_chunks": total_chunks,
                "total_documents": unique_docs,
                "collection_name": self.table_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": self.table_name
            }
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of similar documents with content and metadata
        """
        try:
            logger.info(f"[VECTOR] Connecting to Supabase database for vector search...")
            conn = self._get_db_connection()
            register_vector(conn)
            cursor = conn.cursor()
            
            # First, check how many total documents we have
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_count = cursor.fetchone()[0]
            logger.info(f"[VECTOR] Total documents in {self.table_name}: {total_count}")
            
            # Force sequential scan to ensure results are returned (avoid index scan issues)
            cursor.execute("SET enable_indexscan = off;")
            cursor.execute("SET enable_seqscan = on;")
            
            # Convert to embedding list (pgvector handles serialization)
            # embedding_str = '[' + ','.join(f'{val:.10f}' for val in query_embedding) + ']'
            
            # Perform cosine similarity search
            query = f"""
                SELECT 
                    id,
                    content,
                    metadata,
                    document_name,
                    chunk_index,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            logger.info(f"[VECTOR] Executing vector similarity search...")
            logger.info(f"[VECTOR] Executing vector similarity search...")
            cursor.execute(query, (query_embedding, query_embedding, top_k))
            results = cursor.fetchall()
            logger.info(f"[VECTOR] Vector search query executed. Returned {len(results)} rows")
            
            cursor.close()
            conn.close()
            
            # Format results
            formatted_results = []
            for i, row in enumerate(results):
                similarity_score = float(row[5]) if row[5] is not None else 0.0
                logger.info(f"[VECTOR] Row {i+1}: doc={row[3]}, chunk={row[4]}, similarity={similarity_score:.4f}")
                formatted_results.append({
                    "id": str(row[0]),
                    "content": row[1],
                    "metadata": row[2] if row[2] else {},
                    "document_name": row[3],
                    "chunk_index": row[4],
                    "similarity": similarity_score
                })
            
            if not formatted_results:
                logger.warning(f"[VECTOR] WARNING: No results returned from vector search. Check if documents are indexed.")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[VECTOR] ERROR in vector search: {str(e)}", exc_info=True)
            return []
    
    def load_index(self):
        """Load existing vector store index for LlamaIndex compatibility"""
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core import Document
            from llama_index.core.node_parser import SimpleNodeParser
            
            # Create a custom vector store that uses our Supabase implementation
            # For now, return None and we'll handle retrieval directly in tools
            logger.info("Using direct Supabase vector search (no LlamaIndex index needed)")
            return None
        except Exception as e:
            logger.warning(f"Could not load index: {str(e)}")
            return None

