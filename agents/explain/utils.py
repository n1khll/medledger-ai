"""Utility functions for the Explain Agent - MIP-003 compliant"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Union
from jsonschema import validate, ValidationError
from dateutil import parser


# Explain Agent Input Schema (JSON Schema Draft 07)
EXPLAIN_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ExplainAgentInput",
    "type": "object",
    "required": ["patient_id", "record_text"],
    "properties": {
        "patient_id": {"type": "string"},
        "record_text": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "integer"},
                "source": {"type": "string"}
            },
            "additionalProperties": True
        }
    },
    "additionalProperties": False
}


def canonical_sha256(obj: Dict[str, Any]) -> str:
    """
    Compute the canonical hash of a dictionary object using the MIP-003 standard.
    
    Canonical hash function as per MIP-003:
    json.dumps(obj, separators=(',', ':'), sort_keys=True) → sha256
    
    Args:
        obj: Dictionary object to hash
        
    Returns:
        SHA256 hash string (hex encoded)
    """
    # Create canonical JSON string (no whitespace, sorted keys)
    canonical_json = json.dumps(obj, separators=(',', ':'), sort_keys=True)
    
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_obj.hexdigest()


# Alias for backward compatibility
canonical_hash = canonical_sha256


def to_epoch_seconds(val: Union[int, str, None]) -> int:
    """
    Convert a timestamp value to epoch seconds (Unix timestamp).
    
    Accepts:
    - Integer (already epoch seconds): returned as-is
    - ISO 8601 datetime string: converted to epoch seconds
    - None: raises ValueError
    
    Args:
        val: Timestamp value (int or ISO datetime string)
        
    Returns:
        Epoch seconds (Unix timestamp) as integer
        
    Raises:
        ValueError: If the value cannot be parsed
    """
    if val is None:
        raise ValueError("Cannot convert None to epoch seconds")
    if isinstance(val, int):
        return val
    try:
        return int(parser.isoparse(val).timestamp())
    except Exception:
        raise ValueError("Invalid time format; expected epoch int or ISO datetime")


def epoch_or_default(val: Union[int, str, None], default_secs: int = 3600) -> int:
    """
    Convert timestamp to epoch seconds or return default (now + default_secs).
    
    Args:
        val: Timestamp value (int, ISO datetime string, or None)
        default_secs: Default seconds to add to current time if val is None/missing
        
    Returns:
        Epoch seconds (Unix timestamp) as integer
    """
    if val is None:
        return int(datetime.utcnow().timestamp()) + default_secs
    try:
        return to_epoch_seconds(val)
    except (ValueError, TypeError):
        # If parsing fails, return default
        return int(datetime.utcnow().timestamp()) + default_secs


def validate_against_schema(input_data: Dict[str, Any]) -> None:
    """
    Validate input data against the Explain Agent schema using jsonschema.
    
    Uses the EXPLAIN_INPUT_SCHEMA to validate:
    - Required fields: patient_id, record_text
    - Optional fields: metadata
    - Type checking for all fields
    - No additional properties allowed
    
    Args:
        input_data: Input data dictionary
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    try:
        validate(instance=input_data, schema=EXPLAIN_INPUT_SCHEMA)
    except ValidationError as e:
        raise ValueError(str(e))


def validate_input_schema(input_data: Dict[str, Any]) -> bool:
    """
    Validate input data against the Explain Agent schema.
    
    DEPRECATED: Use validate_against_schema() instead for proper jsonschema validation.
    
    Required fields:
    - patient_id (string)
    - record_text (string)
    
    Optional fields:
    - metadata (object with optional timestamp, source, etc.)
    
    Args:
        input_data: Input data dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Use the new jsonschema-based validation
    validate_against_schema(input_data)
    return True

