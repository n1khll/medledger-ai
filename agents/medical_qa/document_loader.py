"""Document loading and processing for medical knowledge base"""
import os
from typing import List, Dict, Any
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.extractors import TitleExtractor
import sys
from pathlib import Path
import uuid
import json
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from .vector_store import SupabaseVectorStoreManager

logger = get_logger(__name__)

class DocumentLoader:
    """Handles document loading, chunking, and indexing"""
    
    def __init__(self):
        """Initialize document loader with OpenAI embeddings"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Initialize vector store manager
        self.vector_store_manager = SupabaseVectorStoreManager()
        
        # Set embedding model
        self.embed_model = OpenAIEmbedding(
            model_name="text-embedding-ada-002",
            api_key=self.openai_api_key
        )
        
        logger.info("Document loader initialized")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file and return documents"""
        logger.info(f"Loading PDF: {file_path}")
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        logger.info(f"Loaded {len(docs)} document(s)")
        return docs
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that embedding is valid (no NaN or Inf values)
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        import math
        if not embedding:
            return False
        
        if len(embedding) != 1536:  # text-embedding-ada-002 dimension
            logger.warning(f"[VALIDATION] Embedding dimension mismatch: expected 1536, got {len(embedding)}")
            return False
        
        # Check for NaN or Inf values
        for i, val in enumerate(embedding):
            if not math.isfinite(val):
                logger.error(f"[VALIDATION] Invalid embedding value at index {i}: {val} (NaN or Inf)")
                return False
        
        return True
    
    def _store_nodes_in_supabase(self, nodes: List, document_name: str):
        """
        Store nodes directly in Supabase table with embedding validation
        
        Args:
            nodes: List of LlamaIndex nodes with embeddings
            document_name: Name of the document
        """
        try:
            conn = self.vector_store_manager._get_db_connection()
            register_vector(conn)
            cursor = conn.cursor()
            
            # Prepare data for batch insert with validation
            data_to_insert = []
            skipped_count = 0
            
            logger.info(f"[STORAGE] Processing {len(nodes)} nodes for storage...")
            
            for idx, node in enumerate(nodes):
                node_id = str(uuid.uuid4())
                content = node.text
                content_length = len(content)
                metadata = json.dumps(node.metadata if node.metadata else {})
                
                # Get embedding
                embedding = node.embedding if hasattr(node, 'embedding') and node.embedding else None
                
                if not embedding:
                    logger.warning(f"[STORAGE] Node {idx} has no embedding, skipping...")
                    skipped_count += 1
                    continue
                
                # Validate embedding before storing
                if not self._validate_embedding(embedding):
                    logger.error(f"[STORAGE] Node {idx} has invalid embedding (NaN/Inf), skipping...")
                    logger.error(f"[STORAGE] Content preview: {content[:200]}...")
                    skipped_count += 1
                    continue
                
                # Convert to PostgreSQL vector format
                # Use proper formatting to avoid scientific notation issues
                # embedding_str = '[' + ','.join(f'{val:.10f}' for val in embedding) + ']'
                
                logger.info(f"[STORAGE] Node {idx}: content_length={content_length}, embedding_valid=True, embedding_dim={len(embedding)}")
                
                data_to_insert.append((
                    node_id,
                    content,
                    metadata,
                    embedding, # Pass list directly
                    document_name,
                    idx
                ))
            
            # Insert records one by one to ensure proper vector casting
            if data_to_insert:
                insert_query = f"""
                    INSERT INTO {self.vector_store_manager.table_name}
                    (id, content, metadata, embedding, document_name, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                for record in data_to_insert:
                    cursor.execute(insert_query, record)
                conn.commit()
                logger.info(f"[STORAGE] Successfully stored {len(data_to_insert)} chunks in Supabase")
                if skipped_count > 0:
                    logger.warning(f"[STORAGE] Skipped {skipped_count} chunks due to invalid embeddings")
            else:
                raise ValueError(f"No valid chunks to store! All {len(nodes)} nodes had invalid embeddings.")
            
            cursor.close()
            conn.close()
            
            return len(data_to_insert)  # Return count of successfully stored chunks
            
        except Exception as e:
            logger.error(f"[STORAGE] Error storing nodes in Supabase: {str(e)}", exc_info=True)
            raise
    
    def process_and_index_documents(
        self, 
        documents: List[Document],
        document_name: str = None
    ) -> int:
        """
        Process documents through ingestion pipeline and store in Supabase
        
        Args:
            documents: List of Document objects
            document_name: Optional name for the document
            
        Returns:
            Number of chunks created
        """
        logger.info(f"[CHUNKING] Processing {len(documents)} document(s)")
        
        # Log document sizes before chunking
        total_chars = sum(len(doc.text) for doc in documents)
        logger.info(f"[CHUNKING] Total document size: {total_chars} characters")
        
        # Create ingestion pipeline (matching chatbot structure)
        # Chunking strategy: 2048 chars per chunk, 0 overlap
        chunk_size = 2048
        chunk_overlap = 0
        logger.info(f"[CHUNKING] Chunking strategy: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.info(f"[CHUNKING] Expected chunks: ~{total_chars // chunk_size + 1} chunks")
        
        pipeline = IngestionPipeline(
            transformations=[
                # Split documents into chunks (matching chatbot: chunk_size=2048, overlap=0)
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                # Extract titles
                TitleExtractor(),
                # Generate embeddings
                self.embed_model,
            ]
        )
        
        # Run pipeline to create nodes
        logger.info("[CHUNKING] Running ingestion pipeline (chunking + embedding)...")
        nodes = pipeline.run(documents=documents)
        logger.info(f"[CHUNKING] Created {len(nodes)} nodes")
        
        # Log chunking details
        for i, node in enumerate(nodes):
            chunk_length = len(node.text)
            has_embedding = hasattr(node, 'embedding') and node.embedding is not None
            logger.info(f"[CHUNKING] Chunk {i+1}: length={chunk_length} chars, has_embedding={has_embedding}")
        
        # Add document name to metadata if provided
        if document_name:
            for node in nodes:
                if not node.metadata:
                    node.metadata = {}
                node.metadata["document_name"] = document_name
        
        # Store nodes directly in Supabase
        logger.info("[STORAGE] Storing nodes in Supabase with validation...")
        stored_count = self._store_nodes_in_supabase(nodes, document_name)
        
        logger.info(f"[STORAGE] Successfully processed and stored {stored_count} nodes")
        return stored_count
    
    def index_pdf_file(self, file_path: str, document_name: str = None) -> Dict[str, Any]:
        """
        Complete workflow: Load PDF → Process → Store in Supabase
        
        Args:
            file_path: Path to PDF file
            document_name: Optional name for the document
            
        Returns:
            Dict with indexing results
        """
        try:
            # Load PDF
            documents = self.load_pdf(file_path)
            
            # Use file name if document_name not provided
            if not document_name:
                document_name = Path(file_path).stem
            
            # Process and store in Supabase
            chunks_created = self.process_and_index_documents(documents, document_name)
            
            # Get stats
            stats = self.vector_store_manager.get_stats()
            
            logger.info(f"[INDEX] Document '{document_name}' indexed successfully:")
            logger.info(f"[INDEX]   - Chunks created: {chunks_created}")
            logger.info(f"[INDEX]   - Total chunks in DB: {stats['total_chunks']}")
            logger.info(f"[INDEX]   - Total documents in DB: {stats['total_documents']}")
            
            return {
                "status": "success",
                "document_name": document_name,
                "chunks_created": chunks_created,
                "total_chunks_in_db": stats["total_chunks"],
                "total_documents_in_db": stats["total_documents"],
                "message": f"Successfully indexed {document_name} with {chunks_created} chunks"
            }
        except Exception as e:
            logger.error(f"Error indexing PDF: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to index PDF: {str(e)}")


from typing import List, Dict, Any
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.extractors import TitleExtractor
import sys
from pathlib import Path
import uuid
import json
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from .vector_store import SupabaseVectorStoreManager

logger = get_logger(__name__)

class DocumentLoader:
    """Handles document loading, chunking, and indexing"""
    
    def __init__(self):
        """Initialize document loader with OpenAI embeddings"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Initialize vector store manager
        self.vector_store_manager = SupabaseVectorStoreManager()
        
        # Set embedding model
        self.embed_model = OpenAIEmbedding(
            model_name="text-embedding-ada-002",
            api_key=self.openai_api_key
        )
        
        logger.info("Document loader initialized")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file and return documents"""
        logger.info(f"Loading PDF: {file_path}")
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        logger.info(f"Loaded {len(docs)} document(s)")
        return docs
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that embedding is valid (no NaN or Inf values)
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        import math
        if not embedding:
            return False
        
        if len(embedding) != 1536:  # text-embedding-ada-002 dimension
            logger.warning(f"[VALIDATION] Embedding dimension mismatch: expected 1536, got {len(embedding)}")
            return False
        
        # Check for NaN or Inf values
        for i, val in enumerate(embedding):
            if not math.isfinite(val):
                logger.error(f"[VALIDATION] Invalid embedding value at index {i}: {val} (NaN or Inf)")
                return False
        
        return True
    
    def _store_nodes_in_supabase(self, nodes: List, document_name: str):
        """
        Store nodes directly in Supabase table with embedding validation
        
        Args:
            nodes: List of LlamaIndex nodes with embeddings
            document_name: Name of the document
        """
        try:
            conn = self.vector_store_manager._get_db_connection()
            register_vector(conn)
            cursor = conn.cursor()
            
            # Prepare data for batch insert with validation
            data_to_insert = []
            skipped_count = 0
            
            logger.info(f"[STORAGE] Processing {len(nodes)} nodes for storage...")
            
            for idx, node in enumerate(nodes):
                node_id = str(uuid.uuid4())
                content = node.text
                content_length = len(content)
                metadata = json.dumps(node.metadata if node.metadata else {})
                
                # Get embedding
                embedding = node.embedding if hasattr(node, 'embedding') and node.embedding else None
                
                if not embedding:
                    logger.warning(f"[STORAGE] Node {idx} has no embedding, skipping...")
                    skipped_count += 1
                    continue
                
                # Validate embedding before storing
                if not self._validate_embedding(embedding):
                    logger.error(f"[STORAGE] Node {idx} has invalid embedding (NaN/Inf), skipping...")
                    logger.error(f"[STORAGE] Content preview: {content[:200]}...")
                    skipped_count += 1
                    continue
                
                # Convert to PostgreSQL vector format
                # Use proper formatting to avoid scientific notation issues
                # embedding_str = '[' + ','.join(f'{val:.10f}' for val in embedding) + ']'
                
                logger.info(f"[STORAGE] Node {idx}: content_length={content_length}, embedding_valid=True, embedding_dim={len(embedding)}")
                
                data_to_insert.append((
                    node_id,
                    content,
                    metadata,
                    embedding, # Pass list directly
                    document_name,
                    idx
                ))
            
            # Insert records one by one to ensure proper vector casting
            if data_to_insert:
                insert_query = f"""
                    INSERT INTO {self.vector_store_manager.table_name}
                    (id, content, metadata, embedding, document_name, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                for record in data_to_insert:
                    cursor.execute(insert_query, record)
                conn.commit()
                logger.info(f"[STORAGE] Successfully stored {len(data_to_insert)} chunks in Supabase")
                if skipped_count > 0:
                    logger.warning(f"[STORAGE] Skipped {skipped_count} chunks due to invalid embeddings")
            else:
                raise ValueError(f"No valid chunks to store! All {len(nodes)} nodes had invalid embeddings.")
            
            cursor.close()
            conn.close()
            
            return len(data_to_insert)  # Return count of successfully stored chunks
            
        except Exception as e:
            logger.error(f"[STORAGE] Error storing nodes in Supabase: {str(e)}", exc_info=True)
            raise
    
    def process_and_index_documents(
        self, 
        documents: List[Document],
        document_name: str = None
    ) -> int:
        """
        Process documents through ingestion pipeline and store in Supabase
        
        Args:
            documents: List of Document objects
            document_name: Optional name for the document
            
        Returns:
            Number of chunks created
        """
        logger.info(f"[CHUNKING] Processing {len(documents)} document(s)")
        
        # Log document sizes before chunking
        total_chars = sum(len(doc.text) for doc in documents)
        logger.info(f"[CHUNKING] Total document size: {total_chars} characters")
        
        # Create ingestion pipeline (matching chatbot structure)
        # Chunking strategy: 2048 chars per chunk, 0 overlap
        chunk_size = 2048
        chunk_overlap = 0
        logger.info(f"[CHUNKING] Chunking strategy: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.info(f"[CHUNKING] Expected chunks: ~{total_chars // chunk_size + 1} chunks")
        
        pipeline = IngestionPipeline(
            transformations=[
                # Split documents into chunks (matching chatbot: chunk_size=2048, overlap=0)
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                # Extract titles
                TitleExtractor(),
                # Generate embeddings
                self.embed_model,
            ]
        )
        
        # Run pipeline to create nodes
        logger.info("[CHUNKING] Running ingestion pipeline (chunking + embedding)...")
        nodes = pipeline.run(documents=documents)
        logger.info(f"[CHUNKING] Created {len(nodes)} nodes")
        
        # Log chunking details
        for i, node in enumerate(nodes):
            chunk_length = len(node.text)
            has_embedding = hasattr(node, 'embedding') and node.embedding is not None
            logger.info(f"[CHUNKING] Chunk {i+1}: length={chunk_length} chars, has_embedding={has_embedding}")
        
        # Add document name to metadata if provided
        if document_name:
            for node in nodes:
                if not node.metadata:
                    node.metadata = {}
                node.metadata["document_name"] = document_name
        
        # Store nodes directly in Supabase
        logger.info("[STORAGE] Storing nodes in Supabase with validation...")
        stored_count = self._store_nodes_in_supabase(nodes, document_name)
        
        logger.info(f"[STORAGE] Successfully processed and stored {stored_count} nodes")
        return stored_count
    
    def index_pdf_file(self, file_path: str, document_name: str = None) -> Dict[str, Any]:
        """
        Complete workflow: Load PDF → Process → Store in Supabase
        
        Args:
            file_path: Path to PDF file
            document_name: Optional name for the document
            
        Returns:
            Dict with indexing results
        """
        try:
            # Load PDF
            documents = self.load_pdf(file_path)
            
            # Use file name if document_name not provided
            if not document_name:
                document_name = Path(file_path).stem
            
            # Process and store in Supabase
            chunks_created = self.process_and_index_documents(documents, document_name)
            
            # Get stats
            stats = self.vector_store_manager.get_stats()
            
            logger.info(f"[INDEX] Document '{document_name}' indexed successfully:")
            logger.info(f"[INDEX]   - Chunks created: {chunks_created}")
            logger.info(f"[INDEX]   - Total chunks in DB: {stats['total_chunks']}")
            logger.info(f"[INDEX]   - Total documents in DB: {stats['total_documents']}")
            
            return {
                "status": "success",
                "document_name": document_name,
                "chunks_created": chunks_created,
                "total_chunks_in_db": stats["total_chunks"],
                "total_documents_in_db": stats["total_documents"],
                "message": f"Successfully indexed {document_name} with {chunks_created} chunks"
            }
        except Exception as e:
            logger.error(f"Error indexing PDF: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to index PDF: {str(e)}")

