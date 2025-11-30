"""Medical Q&A Agent - FastAPI application"""
import os
import uvicorn
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from shared.logging_config import setup_logging
from .qa_engine import MedicalQAEngine
from .document_loader import DocumentLoader
from .vector_store import SupabaseVectorStoreManager
from .utils import validate_environment, get_input_schema

# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Validate environment
env_validation = validate_environment()
if not env_validation["valid"]:
    logger.error(f"Missing required environment variables: {env_validation['missing_required']}")
    logger.warning("Service may not function correctly without required variables")

logger.info("Starting Medical Q&A Agent API Server...")

# Initialize FastAPI
app = FastAPI(
    title="Medical Q&A Agent - RAG-based Medical Knowledge Chatbot",
    description="API for Medical Q&A Agent: RAG-powered medical knowledge chatbot using LangChain and Supabase",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global instances (lazy initialization)
# ─────────────────────────────────────────────────────────────────────────────
qa_engine: Optional[MedicalQAEngine] = None
document_loader: Optional[DocumentLoader] = None
vector_store_manager: Optional[SupabaseVectorStoreManager] = None

def get_qa_engine() -> MedicalQAEngine:
    """Get or initialize Q&A engine"""
    global qa_engine
    if qa_engine is None:
        qa_engine = MedicalQAEngine()
    return qa_engine

def get_document_loader() -> DocumentLoader:
    """Get or initialize document loader"""
    global document_loader
    if document_loader is None:
        document_loader = DocumentLoader()
    return document_loader

def get_vector_store_manager() -> SupabaseVectorStoreManager:
    """Get or initialize vector store manager"""
    global vector_store_manager
    if vector_store_manager is None:
        vector_store_manager = SupabaseVectorStoreManager()
    return vector_store_manager

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    """Request model for asking questions"""
    query: str = Field(..., description="Medical question or query")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")

class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    status: str
    document_name: str
    chunks_created: int
    total_chunks_in_db: int
    message: str

# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/status")
async def status(job_id: Optional[str] = Query(None)):
    """
    Masumi health/verification: when called with no query params return 200 {"status":"ok"}.
    If job_id is provided, return job status (for future use).
    """
    if not job_id:
        return {"status": "ok"}
    
    # Future: Implement job status tracking if needed
    return {"status": "pending", "job_id": job_id, "result": None}

@app.get("/availability")
async def check_availability():
    """Check if the Medical Q&A Agent is available"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "available",
            "agent": "Medical Q&A Agent",
            "message": "Medical Q&A Agent is operational"
        }
    )

@app.get("/input_schema")
async def input_schema():
    """Returns the input schema for the Medical Q&A Agent"""
    schema = get_input_schema()
    return JSONResponse(status_code=200, content=schema)

@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask a medical question and get an answer from the knowledge base.
    
    Args:
        request: AskRequest with query and optional conversation_id
        
    Returns:
        JSON response with answer, sources, and metadata
    """
    logger.info(f"Received query: {request.query[:100]}...")
    
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Get Q&A engine
        engine = get_qa_engine()
        
        # Process query
        result = engine.process_query(
            query=request.query,
            conversation_history=None  # Future: Implement conversation history
        )
        
        # Return response
        return JSONResponse(
            status_code=200,
            content={
                "query": result["query"],
                "answer": result["answer"],
                "sources": result["sources"],
                "status": result["status"],
                "conversation_id": request.conversation_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/upload_document")
async def upload_document(
    file: UploadFile = File(..., description="PDF medical document file"),
    document_name: Optional[str] = Form(None, description="Optional document name")
):
    """
    Upload a PDF document to the medical knowledge base.
    
    The document will be processed, chunked, embedded, and stored in Supabase.
    
    Args:
        file: PDF file to upload
        document_name: Optional name for the document
        
    Returns:
        Upload result with statistics
    """
    logger.info(f"Received document upload: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Create temporary file to save uploaded PDF
    temp_file = None
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # Use document name or file name
        doc_name = document_name or Path(file.filename).stem
        
        # Get document loader
        loader = get_document_loader()
        
        # Process and index document
        result = loader.index_pdf_file(temp_file, doc_name)
        
        logger.info(f"Successfully indexed document: {doc_name}")
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the medical knowledge base.
    
    Returns:
        Statistics including total chunks and documents
    """
    try:
        manager = get_vector_store_manager()
        stats = manager.get_stats()
        
        return JSONResponse(
            status_code=200,
            content=stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

@app.get("/collections")
async def get_collections():
    """
    Get list of document collections in the knowledge base.
    
    Returns:
        List of document names
    """
    try:
        manager = get_vector_store_manager()
        
        # Query Supabase for unique document names
        response = manager.supabase_client.table(manager.table_name).select("document_name").execute()
        unique_docs = list(set([row.get("document_name") for row in response.data if row.get("document_name")]))
        
        return JSONResponse(
            status_code=200,
            content={
                "collections": unique_docs,
                "count": len(unique_docs)
            }
        )
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collections: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    # Get port from environment
    port = int(os.environ.get("PORT", os.environ.get("API_PORT", 8002)))
    host = os.environ.get("API_HOST", os.environ.get("HOST", "0.0.0.0"))  # Bind to all interfaces
    
    # Display URL uses localhost for browser access (0.0.0.0 doesn't work in browsers)
    display_host = "localhost" if host == "0.0.0.0" else host
    
    print("\n" + "=" * 70)
    print("🚀 Starting Medical Q&A Agent API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://{display_host}:{port}/docs")
    print(f"Ask Question Endpoint:     http://{display_host}:{port}/ask")
    print(f"Upload Document:          http://{display_host}:{port}/upload_document")
    print(f"Health Check:              http://{display_host}:{port}/health")
    print(f"Status Check:              http://{display_host}:{port}/status")
    print(f"Stats:                    http://{display_host}:{port}/stats")
    print("\n💡 This agent provides medical Q&A using RAG with Supabase vector store")
    print("=" * 70 + "\n")
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


import uvicorn
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from shared.logging_config import setup_logging
from .qa_engine import MedicalQAEngine
from .document_loader import DocumentLoader
from .vector_store import SupabaseVectorStoreManager
from .utils import validate_environment, get_input_schema

# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Validate environment
env_validation = validate_environment()
if not env_validation["valid"]:
    logger.error(f"Missing required environment variables: {env_validation['missing_required']}")
    logger.warning("Service may not function correctly without required variables")

logger.info("Starting Medical Q&A Agent API Server...")

# Initialize FastAPI
app = FastAPI(
    title="Medical Q&A Agent - RAG-based Medical Knowledge Chatbot",
    description="API for Medical Q&A Agent: RAG-powered medical knowledge chatbot using LangChain and Supabase",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global instances (lazy initialization)
# ─────────────────────────────────────────────────────────────────────────────
qa_engine: Optional[MedicalQAEngine] = None
document_loader: Optional[DocumentLoader] = None
vector_store_manager: Optional[SupabaseVectorStoreManager] = None

def get_qa_engine() -> MedicalQAEngine:
    """Get or initialize Q&A engine"""
    global qa_engine
    if qa_engine is None:
        qa_engine = MedicalQAEngine()
    return qa_engine

def get_document_loader() -> DocumentLoader:
    """Get or initialize document loader"""
    global document_loader
    if document_loader is None:
        document_loader = DocumentLoader()
    return document_loader

def get_vector_store_manager() -> SupabaseVectorStoreManager:
    """Get or initialize vector store manager"""
    global vector_store_manager
    if vector_store_manager is None:
        vector_store_manager = SupabaseVectorStoreManager()
    return vector_store_manager

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    """Request model for asking questions"""
    query: str = Field(..., description="Medical question or query")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")

class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    status: str
    document_name: str
    chunks_created: int
    total_chunks_in_db: int
    message: str

# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/status")
async def status(job_id: Optional[str] = Query(None)):
    """
    Masumi health/verification: when called with no query params return 200 {"status":"ok"}.
    If job_id is provided, return job status (for future use).
    """
    if not job_id:
        return {"status": "ok"}
    
    # Future: Implement job status tracking if needed
    return {"status": "pending", "job_id": job_id, "result": None}

@app.get("/availability")
async def check_availability():
    """Check if the Medical Q&A Agent is available"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "available",
            "agent": "Medical Q&A Agent",
            "message": "Medical Q&A Agent is operational"
        }
    )

@app.get("/input_schema")
async def input_schema():
    """Returns the input schema for the Medical Q&A Agent"""
    schema = get_input_schema()
    return JSONResponse(status_code=200, content=schema)

@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask a medical question and get an answer from the knowledge base.
    
    Args:
        request: AskRequest with query and optional conversation_id
        
    Returns:
        JSON response with answer, sources, and metadata
    """
    logger.info(f"Received query: {request.query[:100]}...")
    
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Get Q&A engine
        engine = get_qa_engine()
        
        # Process query
        result = engine.process_query(
            query=request.query,
            conversation_history=None  # Future: Implement conversation history
        )
        
        # Return response
        return JSONResponse(
            status_code=200,
            content={
                "query": result["query"],
                "answer": result["answer"],
                "sources": result["sources"],
                "status": result["status"],
                "conversation_id": request.conversation_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/upload_document")
async def upload_document(
    file: UploadFile = File(..., description="PDF medical document file"),
    document_name: Optional[str] = Form(None, description="Optional document name")
):
    """
    Upload a PDF document to the medical knowledge base.
    
    The document will be processed, chunked, embedded, and stored in Supabase.
    
    Args:
        file: PDF file to upload
        document_name: Optional name for the document
        
    Returns:
        Upload result with statistics
    """
    logger.info(f"Received document upload: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Create temporary file to save uploaded PDF
    temp_file = None
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # Use document name or file name
        doc_name = document_name or Path(file.filename).stem
        
        # Get document loader
        loader = get_document_loader()
        
        # Process and index document
        result = loader.index_pdf_file(temp_file, doc_name)
        
        logger.info(f"Successfully indexed document: {doc_name}")
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the medical knowledge base.
    
    Returns:
        Statistics including total chunks and documents
    """
    try:
        manager = get_vector_store_manager()
        stats = manager.get_stats()
        
        return JSONResponse(
            status_code=200,
            content=stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

@app.get("/collections")
async def get_collections():
    """
    Get list of document collections in the knowledge base.
    
    Returns:
        List of document names
    """
    try:
        manager = get_vector_store_manager()
        
        # Query Supabase for unique document names
        response = manager.supabase_client.table(manager.table_name).select("document_name").execute()
        unique_docs = list(set([row.get("document_name") for row in response.data if row.get("document_name")]))
        
        return JSONResponse(
            status_code=200,
            content={
                "collections": unique_docs,
                "count": len(unique_docs)
            }
        )
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collections: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    # Get port from environment
    port = int(os.environ.get("PORT", os.environ.get("API_PORT", 8002)))
    host = os.environ.get("API_HOST", os.environ.get("HOST", "0.0.0.0"))  # Bind to all interfaces
    
    # Display URL uses localhost for browser access (0.0.0.0 doesn't work in browsers)
    display_host = "localhost" if host == "0.0.0.0" else host
    
    print("\n" + "=" * 70)
    print("🚀 Starting Medical Q&A Agent API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://{display_host}:{port}/docs")
    print(f"Ask Question Endpoint:     http://{display_host}:{port}/ask")
    print(f"Upload Document:          http://{display_host}:{port}/upload_document")
    print(f"Health Check:              http://{display_host}:{port}/health")
    print(f"Status Check:              http://{display_host}:{port}/status")
    print(f"Stats:                    http://{display_host}:{port}/stats")
    print("\n💡 This agent provides medical Q&A using RAG with Supabase vector store")
    print("=" * 70 + "\n")
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

