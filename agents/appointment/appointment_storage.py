"""Appointment Storage - Save appointments to database after booking with hospital"""
import os
from typing import Dict, Optional, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
APPOINTMENTS_TABLE = os.getenv("APPOINTMENTS_TABLE", "appointments")


async def save_appointment(
    job_id: str,
    appointment_details: Dict[str, Any],
    patient_wallet: str,
    user_request: str
) -> Optional[str]:
    """
    Book appointment with hospital, then save to database.
    
    Flow:
    1. Book with hospital (mock API for hackathon)
    2. If booking successful, save to database with hospital confirmation
    3. Return appointment ID
    
    Args:
        job_id: Job identifier
        appointment_details: Appointment details from agent
        patient_wallet: Patient's wallet address
        user_request: Original user request
        
    Returns:
        Appointment ID (database record ID) or None if failed
    """
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not configured - appointment not saved to database")
        return None
    
    try:
        import asyncpg
        
        # Step 1: Extract appointment details
        hospital_id = appointment_details.get("hospital_id")
        hospital_name = appointment_details.get("hospital_name")
        patient_name = appointment_details.get("patient_name")
        appointment_date = appointment_details.get("appointment_date")
        appointment_time = appointment_details.get("appointment_time")
        specialty = appointment_details.get("specialty")
        confirmation_number = appointment_details.get("confirmation_number")
        pincode = appointment_details.get("pincode")
        symptoms = appointment_details.get("symptoms")
        patient_phone = appointment_details.get("phone")
        patient_email = appointment_details.get("email")
        
        # Step 2: Book with hospital first (mock booking for hackathon)
        logger.info(f"Booking appointment with hospital: {hospital_name}")
        booking_result = await book_appointment_with_hospital(
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            patient_name=patient_name,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            specialty=specialty,
            patient_phone=patient_phone,
            patient_email=patient_email,
            symptoms=symptoms
        )
        
        # Step 3: Determine status based on booking result
        if booking_result.get("success"):
            appointment_status = "confirmed"  # Hospital confirmed
            hospital_confirmation = booking_result.get("hospital_confirmation_number")
            hospital_appointment_id = booking_result.get("hospital_appointment_id")
            logger.info(f"✅ Hospital booking successful: {hospital_confirmation}")
        else:
            # Booking failed - save as "pending" for retry
            appointment_status = "pending_hospital_confirmation"
            hospital_confirmation = None
            hospital_appointment_id = None
            logger.warning(f"⚠️ Hospital booking failed: {booking_result.get('message')}")
            logger.info("Appointment saved as 'pending' - can be retried later")
        
        # Step 4: Save to database with booking result
        conn = await asyncpg.connect(DATABASE_URL)
        
        query = f"""
            INSERT INTO {APPOINTMENTS_TABLE} (
                job_id,
                patient_wallet,
                patient_name,
                hospital_id,
                hospital_name,
                appointment_date,
                appointment_time,
                specialty,
                confirmation_number,
                hospital_confirmation_number,
                hospital_appointment_id,
                pincode,
                symptoms,
                user_request,
                status,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING id
        """
        
        appointment_id = await conn.fetchval(
            query,
            job_id,
            patient_wallet,
            patient_name,
            hospital_id,
            hospital_name,
            appointment_date,
            appointment_time,
            specialty,
            confirmation_number,  # Our confirmation number
            hospital_confirmation,  # Hospital's confirmation number
            hospital_appointment_id,  # Hospital's appointment ID
            pincode,
            symptoms,
            user_request,
            appointment_status,  # confirmed or pending_hospital_confirmation
            datetime.utcnow()
        )
        
        await conn.close()
        
        logger.info(f"Appointment saved to database with ID: {appointment_id}")
        logger.info(f"Status: {appointment_status}")
        if hospital_confirmation:
            logger.info(f"Hospital Confirmation: {hospital_confirmation}")
        
        return str(appointment_id)
        
    except ImportError:
        logger.error("asyncpg not installed. Install with: pip install asyncpg")
        return None
    except Exception as e:
        logger.error(f"Error saving appointment to database: {str(e)}", exc_info=True)
        return None


async def get_appointment_by_job_id(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve appointment by job_id.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Appointment dictionary or None
    """
    if not DATABASE_URL:
        return None
    
    try:
        import asyncpg
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        query = f"""
            SELECT *
            FROM {APPOINTMENTS_TABLE}
            WHERE job_id = $1
            LIMIT 1
        """
        
        row = await conn.fetchrow(query, job_id)
        await conn.close()
        
        if row:
            return dict(row)
        
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving appointment: {str(e)}")
        return None


async def get_appointments_by_patient(patient_wallet: str) -> list:
    """
    Get all appointments for a patient.
    
    Args:
        patient_wallet: Patient's wallet address
        
    Returns:
        List of appointment dictionaries
    """
    if not DATABASE_URL:
        return []
    
    try:
        import asyncpg
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        query = f"""
            SELECT *
            FROM {APPOINTMENTS_TABLE}
            WHERE patient_wallet = $1
            ORDER BY appointment_date DESC, appointment_time DESC
        """
        
        rows = await conn.fetch(query, patient_wallet)
        await conn.close()
        
        return [dict(row) for row in rows]
        
    except Exception as e:
        logger.error(f"Error retrieving appointments: {str(e)}")
        return []

