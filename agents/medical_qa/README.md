# Medical Q&A Agent

RAG-based medical knowledge chatbot using LangChain and Supabase vector store.

## Overview

The Medical Q&A Agent provides intelligent question-answering capabilities for medical topics using:
- **LangChain** for agent orchestration
- **LlamaIndex** for document processing and retrieval
- **Supabase** (pgvector) for vector storage
- **OpenAI** for embeddings and LLM responses

## Features

- Medical knowledge search from indexed documents
- Optional web search for general queries
- PDF document upload and indexing
- Conversation context support
- Source citations in responses

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - Status check (Masumi-friendly)
- `GET /availability` - Availability check

### Query
- `POST /ask` - Ask a medical question
  ```json
  {
    "query": "What is diabetes?",
    "conversation_id": "optional-conversation-id"
  }
  ```

### Document Management
- `POST /upload_document` - Upload PDF to index
- `GET /stats` - Get knowledge base statistics
- `GET /collections` - List document collections

### Schema
- `GET /input_schema` - Get input schema

## Environment Variables

### Required
- `OPENAI_API_KEY` - OpenAI API key
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `SUPABASE_DB_URL` - Supabase database connection string

### Optional
- `OPENAI_MODEL` - OpenAI model (default: gpt-4o-mini)
- `SERPER_API_KEY` - SerpAPI key for web search
- `VECTOR_COLLECTION_NAME` - Vector collection name (default: medical_knowledge_vectors)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
python run_medical_qa_agent.py

# Or using uvicorn directly
uvicorn agents.medical_qa.main:app --host 0.0.0.0 --port 8002 --reload
```

## Deployment

### Render

**Service Configuration:**
- Name: `medical-qa`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn agents.medical_qa.main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
Set all required environment variables in Render dashboard.

## Usage Example

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8002/ask",
    json={"query": "What are the symptoms of diabetes?"}
)
print(response.json()["answer"])

# Upload a document
with open("medical_textbook.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8002/upload_document",
        files={"file": f},
        data={"document_name": "Medical_Textbook_2024"}
    )
print(response.json())
```

## Architecture

- **Standalone Agent**: No communication with other agents
- **LangChain**: Tool-calling agent with medical knowledge search and web search
- **Supabase**: Vector storage using pgvector extension
- **FastAPI**: REST API following same pattern as other agents





RAG-based medical knowledge chatbot using LangChain and Supabase vector store.

## Overview

The Medical Q&A Agent provides intelligent question-answering capabilities for medical topics using:
- **LangChain** for agent orchestration
- **LlamaIndex** for document processing and retrieval
- **Supabase** (pgvector) for vector storage
- **OpenAI** for embeddings and LLM responses

## Features

- Medical knowledge search from indexed documents
- Optional web search for general queries
- PDF document upload and indexing
- Conversation context support
- Source citations in responses

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - Status check (Masumi-friendly)
- `GET /availability` - Availability check

### Query
- `POST /ask` - Ask a medical question
  ```json
  {
    "query": "What is diabetes?",
    "conversation_id": "optional-conversation-id"
  }
  ```

### Document Management
- `POST /upload_document` - Upload PDF to index
- `GET /stats` - Get knowledge base statistics
- `GET /collections` - List document collections

### Schema
- `GET /input_schema` - Get input schema

## Environment Variables

### Required
- `OPENAI_API_KEY` - OpenAI API key
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `SUPABASE_DB_URL` - Supabase database connection string

### Optional
- `OPENAI_MODEL` - OpenAI model (default: gpt-4o-mini)
- `SERPER_API_KEY` - SerpAPI key for web search
- `VECTOR_COLLECTION_NAME` - Vector collection name (default: medical_knowledge_vectors)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
python run_medical_qa_agent.py

# Or using uvicorn directly
uvicorn agents.medical_qa.main:app --host 0.0.0.0 --port 8002 --reload
```

## Deployment

### Render

**Service Configuration:**
- Name: `medical-qa`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn agents.medical_qa.main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
Set all required environment variables in Render dashboard.

## Usage Example

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8002/ask",
    json={"query": "What are the symptoms of diabetes?"}
)
print(response.json()["answer"])

# Upload a document
with open("medical_textbook.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8002/upload_document",
        files={"file": f},
        data={"document_name": "Medical_Textbook_2024"}
    )
print(response.json())
```

## Architecture

- **Standalone Agent**: No communication with other agents
- **LangChain**: Tool-calling agent with medical knowledge search and web search
- **Supabase**: Vector storage using pgvector extension
- **FastAPI**: REST API following same pattern as other agents



