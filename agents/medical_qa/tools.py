"""LangChain tools for Medical Q&A agent"""
import os
from typing import List, Optional
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from llama_index.embeddings.openai import OpenAIEmbedding
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from .vector_store import SupabaseVectorStoreManager

logger = get_logger(__name__)

# Global instances (initialized on first use)
_vector_store_manager: Optional[SupabaseVectorStoreManager] = None
_embed_model: Optional[OpenAIEmbedding] = None

def _get_vector_store_manager():
    """Lazy initialization of vector store manager"""
    global _vector_store_manager
    
    if _vector_store_manager is None:
        try:
            _vector_store_manager = SupabaseVectorStoreManager()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    return _vector_store_manager

def _get_embed_model():
    """Lazy initialization of embedding model"""
    global _embed_model
    
    if _embed_model is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        _embed_model = OpenAIEmbedding(
            model_name="text-embedding-ada-002",
            api_key=openai_api_key
        )
    
    return _embed_model

def medical_knowledge_search(query: str) -> List[str]:
    """
    Search medical knowledge base for relevant information.
    
    Args:
        query: The query string to search
        
    Returns:
        List of relevant document excerpts
    """
    try:
        logger.info(f"[SEARCH] Starting medical knowledge search for query: '{query}'")
        
        # Get vector store manager
        manager = _get_vector_store_manager()
        logger.info(f"[SEARCH] Vector store manager initialized. Table: {manager.table_name}")
        
        # Get embedding model
        embed_model = _get_embed_model()
        logger.info(f"[SEARCH] Embedding model ready: text-embedding-ada-002")
        
        # Generate query embedding
        logger.info(f"[SEARCH] Generating embedding for query...")
        query_embedding = embed_model.get_query_embedding(query)
        logger.info(f"[SEARCH] Query embedding generated. Dimension: {len(query_embedding)}")
        
        # Search similar vectors
        logger.info(f"[SEARCH] Searching for similar vectors (top_k=5)...")
        results = manager.search_similar(query_embedding, top_k=5)
        logger.info(f"[SEARCH] Vector search complete. Found {len(results)} results")
        
        # Log each result with similarity scores
        for i, result in enumerate(results):
            similarity = result.get('similarity', 'N/A')
            doc_name = result.get('document_name', 'Unknown')
            content_preview = result.get('content', '')[:100] + "..." if result.get('content') else 'No content'
            logger.info(f"[SEARCH] Result {i+1}: similarity={similarity:.4f}, doc={doc_name}")
            logger.info(f"[SEARCH]   Content preview: {content_preview}")
        
        # Extract content from results
        extracted_texts = [result["content"] for result in results if result.get("content")]
        
        if extracted_texts:
            logger.info(f"[SEARCH] Retrieved {len(extracted_texts)} text chunks")
            # Log the full content of the top result for debugging
            if len(extracted_texts) > 0:
                logger.info(f"[SEARCH] Top result content (first 500 chars): {extracted_texts[0][:500]}...")
        else:
            logger.warning(f"[SEARCH] WARNING: No relevant information found in medical knowledge base")
        
        return extracted_texts if extracted_texts else ["No relevant information found in medical knowledge base."]
        
    except Exception as e:
        logger.error(f"[SEARCH] ERROR in medical_knowledge_search: {str(e)}", exc_info=True)
        return [f"Error searching medical knowledge: {str(e)}"]

def web_search(query: str) -> str:
    """
    Search the internet for general information.
    
    Args:
        query: The query string to search on the internet
        
    Returns:
        Search results as string
    """
    try:
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if not serper_api_key:
            logger.warning("SERPER_API_KEY not set. Web search unavailable.")
            return "Web search is not configured. Please set SERPER_API_KEY environment variable."
        
        serper = SerpAPIWrapper(serpapi_api_key=serper_api_key)
        result = serper.run(query)
        
        logger.info(f"Web search completed for query: {query[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}", exc_info=True)
        return f"Error performing web search: {str(e)}"

# Create LangChain tools
@tool
def medical_knowledge_search_tool(query: str) -> str:
    """
    Searches the medical knowledge base for relevant information.
    Use this tool first to find answers in medical documents.
    
    Args:
        query: The query string to search within the medical knowledge base
        
    Returns:
        String containing relevant document excerpts matching the query
    """
    results = medical_knowledge_search(query)
    # Join results into a single string
    return "\n\n".join(results) if results else "No relevant information found."

@tool
def internet_search_tool(query: str) -> str:
    """
    Searches the internet for general information.
    Use this tool when you cannot find a satisfactory answer in the medical knowledge base.
    
    Args:
        query: The query string to search on the internet
        
    Returns:
        String containing internet search results
    """
    return web_search(query)

def get_tools():
    """
    Get list of available tools for the agent
    
    Returns:
        List of LangChain tools
    """
    tools = [medical_knowledge_search_tool]
    
    # Only add web search if API key is configured
    if os.getenv("SERPER_API_KEY"):
        tools.append(internet_search_tool)
    else:
        logger.info("SERPER_API_KEY not set. Web search tool disabled.")
    
    return tools


from typing import List, Optional
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from llama_index.embeddings.openai import OpenAIEmbedding
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from .vector_store import SupabaseVectorStoreManager

logger = get_logger(__name__)

# Global instances (initialized on first use)
_vector_store_manager: Optional[SupabaseVectorStoreManager] = None
_embed_model: Optional[OpenAIEmbedding] = None

def _get_vector_store_manager():
    """Lazy initialization of vector store manager"""
    global _vector_store_manager
    
    if _vector_store_manager is None:
        try:
            _vector_store_manager = SupabaseVectorStoreManager()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    return _vector_store_manager

def _get_embed_model():
    """Lazy initialization of embedding model"""
    global _embed_model
    
    if _embed_model is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        _embed_model = OpenAIEmbedding(
            model_name="text-embedding-ada-002",
            api_key=openai_api_key
        )
    
    return _embed_model

def medical_knowledge_search(query: str) -> List[str]:
    """
    Search medical knowledge base for relevant information.
    
    Args:
        query: The query string to search
        
    Returns:
        List of relevant document excerpts
    """
    try:
        logger.info(f"[SEARCH] Starting medical knowledge search for query: '{query}'")
        
        # Get vector store manager
        manager = _get_vector_store_manager()
        logger.info(f"[SEARCH] Vector store manager initialized. Table: {manager.table_name}")
        
        # Get embedding model
        embed_model = _get_embed_model()
        logger.info(f"[SEARCH] Embedding model ready: text-embedding-ada-002")
        
        # Generate query embedding
        logger.info(f"[SEARCH] Generating embedding for query...")
        query_embedding = embed_model.get_query_embedding(query)
        logger.info(f"[SEARCH] Query embedding generated. Dimension: {len(query_embedding)}")
        
        # Search similar vectors
        logger.info(f"[SEARCH] Searching for similar vectors (top_k=5)...")
        results = manager.search_similar(query_embedding, top_k=5)
        logger.info(f"[SEARCH] Vector search complete. Found {len(results)} results")
        
        # Log each result with similarity scores
        for i, result in enumerate(results):
            similarity = result.get('similarity', 'N/A')
            doc_name = result.get('document_name', 'Unknown')
            content_preview = result.get('content', '')[:100] + "..." if result.get('content') else 'No content'
            logger.info(f"[SEARCH] Result {i+1}: similarity={similarity:.4f}, doc={doc_name}")
            logger.info(f"[SEARCH]   Content preview: {content_preview}")
        
        # Extract content from results
        extracted_texts = [result["content"] for result in results if result.get("content")]
        
        if extracted_texts:
            logger.info(f"[SEARCH] Retrieved {len(extracted_texts)} text chunks")
            # Log the full content of the top result for debugging
            if len(extracted_texts) > 0:
                logger.info(f"[SEARCH] Top result content (first 500 chars): {extracted_texts[0][:500]}...")
        else:
            logger.warning(f"[SEARCH] WARNING: No relevant information found in medical knowledge base")
        
        return extracted_texts if extracted_texts else ["No relevant information found in medical knowledge base."]
        
    except Exception as e:
        logger.error(f"[SEARCH] ERROR in medical_knowledge_search: {str(e)}", exc_info=True)
        return [f"Error searching medical knowledge: {str(e)}"]

def web_search(query: str) -> str:
    """
    Search the internet for general information.
    
    Args:
        query: The query string to search on the internet
        
    Returns:
        Search results as string
    """
    try:
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if not serper_api_key:
            logger.warning("SERPER_API_KEY not set. Web search unavailable.")
            return "Web search is not configured. Please set SERPER_API_KEY environment variable."
        
        serper = SerpAPIWrapper(serpapi_api_key=serper_api_key)
        result = serper.run(query)
        
        logger.info(f"Web search completed for query: {query[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}", exc_info=True)
        return f"Error performing web search: {str(e)}"

# Create LangChain tools
@tool
def medical_knowledge_search_tool(query: str) -> str:
    """
    Searches the medical knowledge base for relevant information.
    Use this tool first to find answers in medical documents.
    
    Args:
        query: The query string to search within the medical knowledge base
        
    Returns:
        String containing relevant document excerpts matching the query
    """
    results = medical_knowledge_search(query)
    # Join results into a single string
    return "\n\n".join(results) if results else "No relevant information found."

@tool
def internet_search_tool(query: str) -> str:
    """
    Searches the internet for general information.
    Use this tool when you cannot find a satisfactory answer in the medical knowledge base.
    
    Args:
        query: The query string to search on the internet
        
    Returns:
        String containing internet search results
    """
    return web_search(query)

def get_tools():
    """
    Get list of available tools for the agent
    
    Returns:
        List of LangChain tools
    """
    tools = [medical_knowledge_search_tool]
    
    # Only add web search if API key is configured
    if os.getenv("SERPER_API_KEY"):
        tools.append(internet_search_tool)
    else:
        logger.info("SERPER_API_KEY not set. Web search tool disabled.")
    
    return tools

