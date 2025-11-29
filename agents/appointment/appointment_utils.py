"""Utility functions for the Appointment Scheduling Agent - MIP-003 compliant"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Union
from jsonschema import validate, ValidationError
from dateutil import parser


# Appointment Agent Input Schema (JSON Schema Draft 07)
APPOINTMENT_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AppointmentAgentInput",
    "type": "object",
    "required": ["user_request", "pincode", "patient_info"],
    "properties": {
        "user_request": {
            "type": "string",
            "description": "User's appointment request in plain English"
        },
        "pincode": {
            "type": "string",
            "description": "Pincode/ZIP code for hospital search",
            "pattern": "^[0-9]{5,6}$"
        },
        "patient_info": {
            "type": "object",
            "description": "Patient information (required)",
            "required": ["name", "age", "location"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "location": {"type": "string", "description": "City or location name"},
                "dob": {"type": "string", "format": "date"},
                "symptoms": {"type": "string"},
                "preferred_date": {"type": "string"},
                "preferred_time": {"type": "string"},
                "phone": {"type": "string"},
                "email": {"type": "string", "format": "email"}
            },
            "additionalProperties": True
        },
        "hospital_id": {
            "type": "string",
            "description": "Selected hospital ID (optional, may be determined by agent)"
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
    canonical_json = json.dumps(obj, separators=(',', ':'), sort_keys=True)
    hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_obj.hexdigest()


def to_epoch_seconds(val: Union[int, str, None]) -> int:
    """
    Convert a timestamp value to epoch seconds (Unix timestamp).
    
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
        return int(datetime.utcnow().timestamp()) + default_secs


def validate_against_schema(input_data: Dict[str, Any]) -> None:
    """
    Validate input data against the Appointment Agent schema using jsonschema.
    
    Args:
        input_data: Input data dictionary
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    try:
        validate(instance=input_data, schema=APPOINTMENT_INPUT_SCHEMA)
    except ValidationError as e:
        raise ValueError(str(e))


def get_input_schema() -> Dict[str, Any]:
    """
    Get the MIP-003 compliant input schema for the Appointment Agent.
    
    Returns:
        Input schema dictionary following MIP-003 format (input_data array)
    """
    # MIP-003 format: Return input_data array, not JSON Schema
    return {
        "input_data": [
            {
                "id": "user_request",
                "type": "string",
                "name": "Appointment Request",
                "data": {
                    "description": "Describe your appointment need in plain English (e.g., 'I need to see a cardiologist next week', 'Schedule a general checkup')"
                },
                "validations": [
                    {"validation": "required"}
                ]
            },
            {
                "id": "pincode",
                "type": "string",
                "name": "Pincode/ZIP Code",
                "data": {
                    "description": "Your pincode/ZIP code for finding nearby hospitals (5-6 digits)"
                },
                "validations": [
                    {"validation": "required"},
                    {
                        "validation": "pattern",
                        "value": "^[0-9]{5,6}$"
                    }
                ]
            },
            {
                "id": "patient_info",
                "type": "none",  # Complex object - use "none" type per MIP-003
                "name": "Patient Information",
                "data": {
                    "description": "Patient details object with name (string), age (integer), location (string), and optional fields like dob, symptoms, preferred_date, phone, email"
                },
                "validations": [
                    {"validation": "required"}
                ]
            },
            {
                "id": "hospital_id",
                "type": "string",
                "name": "Hospital ID",
                "data": {
                    "description": "Optional: Specific hospital ID if you want to book at a particular hospital"
                }
            }
        ]
    }

