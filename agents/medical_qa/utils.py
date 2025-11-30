"""Utility functions for Medical Q&A agent"""
import os
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

def validate_environment() -> Dict[str, bool]:
    """
    Validate that all required environment variables are set
    
    Returns:
        Dict with validation results
    """
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
        "SUPABASE_DB_URL": os.getenv("SUPABASE_DB_URL"),
    }
    
    optional_vars = {
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
        "VECTOR_COLLECTION_NAME": os.getenv("VECTOR_COLLECTION_NAME"),
    }
    
    missing_required = [var for var, value in required_vars.items() if not value]
    missing_optional = [var for var, value in optional_vars.items() if not value]
    
    return {
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "all_set": len(missing_required) == 0 and len(missing_optional) == 0
    }

def get_input_schema() -> Dict[str, Any]:
    """
    Get input schema for the Medical Q&A agent
    
    Returns:
        Dict with input schema definition
    """
    return {
        "input_data": [
            {
                "id": "query",
                "type": "string",
                "name": "Medical Query",
                "data": {
                    "description": "Medical question or query to search in the knowledge base"
                },
                "validations": [
                    {"validation": "required"}
                ]
            },
            {
                "id": "conversation_id",
                "type": "string",
                "name": "Conversation ID",
                "data": {
                    "description": "Optional conversation ID for maintaining context across multiple queries"
                }
            }
        ]
    }




import os
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

def validate_environment() -> Dict[str, bool]:
    """
    Validate that all required environment variables are set
    
    Returns:
        Dict with validation results
    """
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
        "SUPABASE_DB_URL": os.getenv("SUPABASE_DB_URL"),
    }
    
    optional_vars = {
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
        "VECTOR_COLLECTION_NAME": os.getenv("VECTOR_COLLECTION_NAME"),
    }
    
    missing_required = [var for var, value in required_vars.items() if not value]
    missing_optional = [var for var, value in optional_vars.items() if not value]
    
    return {
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "all_set": len(missing_required) == 0 and len(missing_optional) == 0
    }

def get_input_schema() -> Dict[str, Any]:
    """
    Get input schema for the Medical Q&A agent
    
    Returns:
        Dict with input schema definition
    """
    return {
        "input_data": [
            {
                "id": "query",
                "type": "string",
                "name": "Medical Query",
                "data": {
                    "description": "Medical question or query to search in the knowledge base"
                },
                "validations": [
                    {"validation": "required"}
                ]
            },
            {
                "id": "conversation_id",
                "type": "string",
                "name": "Conversation ID",
                "data": {
                    "description": "Optional conversation ID for maintaining context across multiple queries"
                }
            }
        ]
    }



