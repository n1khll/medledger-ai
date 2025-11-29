"""Patient Database Helper - Fetch patient information from Supabase with Midnight permissions

Uses DATABASE_URL connection string for direct PostgreSQL connection to Supabase.
Supports both encrypted profile access (via Patient API) and direct appointment_profiles table.
"""
import os
from typing import Dict, Optional, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Supabase PostgreSQL connection string
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "users")
APPOINTMENT_PROFILES_TABLE = os.getenv("APPOINTMENT_PROFILES_TABLE", "appointment_profiles")

# Midnight Permission API Configuration
MIDNIGHT_API_URL = os.getenv("MIDNIGHT_API_URL", "http://localhost:3000/api")
AGENT_WALLET_ADDRESS = os.getenv("AGENT_WALLET_ADDRESS")  # AI Agent's wallet address

# Patient API Configuration (for decrypted data)
PATIENT_API_URL = os.getenv("PATIENT_API_URL", "http://localhost:3000/api")
PATIENT_API_KEY = os.getenv("PATIENT_API_KEY")  # Optional API key


async def fetch_patient_from_db(identifier_from_purchaser: str) -> Optional[Dict[str, Any]]:
    """
    Fetch patient information from Supabase using identifier_from_purchaser.
    
    This function tries multiple approaches:
    1. First tries appointment_profiles table (simple, plaintext)
    2. If not found, checks Midnight permissions and requests decrypted data
    
    Args:
        identifier_from_purchaser: User identifier (wallet address or user ID)
        
    Returns:
        Dictionary with patient information (name, city, age, etc.) or None if not found/no permission
    """
    logger.info(f"Fetching patient info for identifier: {identifier_from_purchaser}")
    
    if not DATABASE_URL:
        logger.error("DATABASE_URL not configured in environment variables")
        return None
    
    try:
        # Step 1: Get wallet address from identifier
        wallet_address = await _get_wallet_address(identifier_from_purchaser)
        if not wallet_address:
            logger.warning(f"Could not find wallet address for identifier: {identifier_from_purchaser}")
            return None
        
        logger.info(f"Found wallet address: {wallet_address}")
        
        # Step 2: Try appointment_profiles table first (simpler, no decryption needed)
        profile = await _fetch_from_appointment_profiles(wallet_address)
        if profile:
            logger.info(f"Found patient info in appointment_profiles table")
            return profile
        
        logger.info(f"No appointment profile found, trying encrypted profile with Midnight permissions...")
        
        # Step 3: Check Midnight permissions for encrypted profile
        has_permission = await _check_midnight_permission(wallet_address)
        if not has_permission:
            logger.warning(f"No Midnight permission for wallet: {wallet_address}")
            return None
        
        logger.info(f"Midnight permission verified for wallet: {wallet_address}")
        
        # Step 4: Fetch encrypted data from Supabase
        encrypted_data = await _fetch_encrypted_from_supabase(wallet_address)
        if not encrypted_data:
            logger.warning(f"No encrypted data found for wallet: {wallet_address}")
            return None
        
        # Step 5: Request decrypted data from Patient API
        decrypted_data = await _request_decrypted_data(wallet_address, encrypted_data)
        
        if decrypted_data:
            logger.info(f"Successfully retrieved decrypted patient data for wallet: {wallet_address}")
            return decrypted_data
        else:
            logger.warning(f"Could not decrypt data for wallet: {wallet_address}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching patient from DB: {str(e)}", exc_info=True)
        return None


async def _get_wallet_address(identifier: str) -> Optional[str]:
    """
    Get wallet address from identifier using direct PostgreSQL connection.
    
    If identifier is already a wallet address, return it.
    Otherwise, query Supabase users table to find the wallet address.
    """
    # Check if identifier looks like a wallet address
    if identifier.startswith("addr") or identifier.startswith("0x"):
        return identifier
    
    # Query Supabase users table
    try:
        import asyncpg
        
        if not DATABASE_URL:
            return None
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Try to find by id (UUID) or wallet_address
        query = f"""
            SELECT wallet_address
            FROM {SUPABASE_TABLE}
            WHERE id::text = $1 OR wallet_address = $1
            LIMIT 1
        """
        
        row = await conn.fetchrow(query, identifier)
        await conn.close()
        
        if row:
            return row.get("wallet_address")
        
        return None
        
    except ImportError:
        logger.error("asyncpg not installed. Install with: pip install asyncpg")
        return None
    except Exception as e:
        logger.error(f"Error getting wallet address: {str(e)}")
        return None


async def _fetch_from_appointment_profiles(wallet_address: str) -> Optional[Dict[str, Any]]:
    """
    Fetch patient info from appointment_profiles table (plaintext, no decryption needed).
    """
    try:
        import asyncpg
        
        if not DATABASE_URL:
            return None
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        query = f"""
            SELECT wallet_address, name, city, phone, email, age, dob
            FROM {APPOINTMENT_PROFILES_TABLE}
            WHERE wallet_address = $1
            LIMIT 1
        """
        
        row = await conn.fetchrow(query, wallet_address)
        await conn.close()
        
        if row:
            return {
                "name": row.get("name"),
                "city": row.get("city"),
                "phone": row.get("phone"),
                "email": row.get("email"),
                "age": row.get("age"),
                "dob": str(row.get("dob")) if row.get("dob") else None,
                "wallet_address": wallet_address
            }
        
        return None
        
    except ImportError:
        logger.error("asyncpg not installed. Install with: pip install asyncpg")
        return None
    except Exception as e:
        logger.debug(f"No appointment profile found: {str(e)}")
        return None


async def _check_midnight_permission(patient_wallet: str) -> bool:
    """
    Check if AI agent has permission to access patient data via Midnight.
    
    Args:
        patient_wallet: Patient's wallet address
        
    Returns:
        True if permission exists and is active, False otherwise
    """
    if not AGENT_WALLET_ADDRESS:
        logger.warning("AGENT_WALLET_ADDRESS not configured - skipping permission check")
        # For development, allow access if permission check is disabled
        return os.getenv("SKIP_PERMISSION_CHECK", "false").lower() == "true"
    
    try:
        import httpx
        
        # Check permissions table via your API
        url = f"{MIDNIGHT_API_URL}/permissions/check"
        headers = {
            "Content-Type": "application/json"
        }
        if PATIENT_API_KEY:
            headers["Authorization"] = f"Bearer {PATIENT_API_KEY}"
        
        payload = {
            "patient_wallet": patient_wallet,
            "requester_wallet": AGENT_WALLET_ADDRESS,
            "resource_id": "profile"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("has_permission", False)
            elif response.status_code == 404:
                # Permission doesn't exist
                logger.info(f"No permission found for patient: {patient_wallet}")
                return False
            else:
                logger.warning(f"Permission check returned status {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"Error checking Midnight permission: {str(e)}")
        # If permission check fails, deny access by default
        return False


async def _fetch_encrypted_from_supabase(wallet_address: str) -> Optional[Dict[str, Any]]:
    """
    Fetch encrypted profile data from Supabase using direct PostgreSQL connection.
    
    Args:
        wallet_address: Patient's wallet address
        
    Returns:
        Dictionary with encrypted data (profile_cipher, etc.) or None
    """
    try:
        import asyncpg
        
        if not DATABASE_URL:
            return None
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        query = f"""
            SELECT id, wallet_address, profile_cipher, created_at
            FROM {SUPABASE_TABLE}
            WHERE wallet_address = $1
            LIMIT 1
        """
        
        row = await conn.fetchrow(query, wallet_address)
        await conn.close()
        
        if row:
            return {
                "id": str(row.get("id")),
                "wallet_address": row.get("wallet_address"),
                "profile_cipher": row.get("profile_cipher"),  # BYTEA - will be bytes
                "created_at": str(row.get("created_at")) if row.get("created_at") else None
            }
        
        return None
        
    except ImportError:
        logger.error("asyncpg not installed. Install with: pip install asyncpg")
        return None
    except Exception as e:
        logger.error(f"Error fetching from Supabase: {str(e)}")
        return None


async def _request_decrypted_data(patient_wallet: str, encrypted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Request decrypted patient data from Patient API.
    
    The Patient API will:
    1. Verify Midnight permission
    2. Decrypt the profile_cipher using patient's wallet signature
    3. Return plaintext patient data
    
    Args:
        patient_wallet: Patient's wallet address
        encrypted_data: Encrypted data from Supabase
        
    Returns:
        Dictionary with decrypted patient information or None
    """
    try:
        import httpx
        import base64
        
        url = f"{PATIENT_API_URL}/profile/decrypt"
        headers = {
            "Content-Type": "application/json"
        }
        if PATIENT_API_KEY:
            headers["Authorization"] = f"Bearer {PATIENT_API_KEY}"
        
        # Convert BYTEA to base64 if needed
        profile_cipher = encrypted_data.get("profile_cipher")
        if isinstance(profile_cipher, bytes):
            profile_cipher = base64.b64encode(profile_cipher).decode('utf-8')
        
        payload = {
            "patient_wallet": patient_wallet,
            "requester_wallet": AGENT_WALLET_ADDRESS,
            "encrypted_data": {
                "id": encrypted_data.get("id"),
                "wallet_address": encrypted_data.get("wallet_address"),
                "profile_cipher": profile_cipher
            },
            "resource_id": "profile",
            "scope": "appointment_scheduling"  # Specific scope for appointment use case
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                # Extract patient info from decrypted profile
                profile = data.get("profile", {})
                return {
                    "name": profile.get("name"),
                    "city": profile.get("city"),
                    "age": profile.get("age"),
                    "dob": profile.get("dob"),
                    "phone": profile.get("phone"),
                    "email": profile.get("email"),
                    "address": profile.get("address"),
                    "wallet_address": patient_wallet
                }
            elif response.status_code == 403:
                logger.warning(f"Permission denied for wallet: {patient_wallet}")
                return None
            elif response.status_code == 404:
                logger.warning(f"Patient profile not found for wallet: {patient_wallet}")
                return None
            else:
                logger.warning(f"Decryption API returned status {response.status_code}: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error requesting decrypted data: {str(e)}")
        return None
