"""Hospital Booking API - Mock implementation for hackathon MVP

This simulates hospital booking for demo purposes.
In production, replace with real hospital API integration.
"""
import os
import uuid
import random
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

# Configuration
USE_MOCK_BOOKING = os.getenv("USE_MOCK_BOOKING", "true").lower() == "true"
HOSPITAL_BOOKING_API_URL = os.getenv("HOSPITAL_BOOKING_API_URL")  # For future real API


async def book_appointment_with_hospital(
    hospital_id: str,
    hospital_name: str,
    patient_name: str,
    appointment_date: str,
    appointment_time: str,
    specialty: str,
    patient_phone: Optional[str] = None,
    patient_email: Optional[str] = None,
    symptoms: Optional[str] = None
) -> Dict[str, Any]:
    """
    Book appointment with hospital (mock implementation for hackathon).
    
    This simulates a real hospital booking API call:
    - Simulates network delay
    - Returns realistic confirmation
    - Can be replaced with real API later
    
    Args:
        hospital_id: Hospital identifier
        hospital_name: Hospital name
        patient_name: Patient's name
        appointment_date: Appointment date (YYYY-MM-DD)
        appointment_time: Appointment time (HH:MM)
        specialty: Medical specialty
        patient_phone: Patient's phone number
        patient_email: Patient's email
        symptoms: Patient symptoms
        
    Returns:
        Dictionary with booking confirmation:
        {
            "success": True,
            "hospital_confirmation_number": "HOSP-12345",
            "hospital_appointment_id": "uuid",
            "message": "Appointment confirmed with hospital",
            "status": "confirmed",
            "booking_timestamp": "2024-12-20T10:30:00Z"
        }
    """
    logger.info(f"Booking appointment with {hospital_name} (ID: {hospital_id})")
    logger.info(f"Patient: {patient_name}, Date: {appointment_date} {appointment_time}, Specialty: {specialty}")
    
    # If real API URL is configured and mock is disabled, try real API
    if HOSPITAL_BOOKING_API_URL and not USE_MOCK_BOOKING:
        return await _book_with_real_api(
            hospital_id, patient_name, appointment_date, appointment_time,
            specialty, patient_phone, patient_email, symptoms
        )
    
    # Mock booking for hackathon MVP
    try:
        # Simulate API call delay (1-2 seconds)
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        # Simulate booking success (95% success rate for realistic demo)
        if random.random() > 0.05:  # 95% success
            # Generate realistic hospital confirmation number
            # Format: HOSP-{hospital_id_short}-{random_5_digits}
            hospital_short = hospital_id.replace("HOSP_", "").replace("HOSP", "")[:3]
            if not hospital_short:
                hospital_short = "001"
            confirmation_num = f"HOSP-{hospital_short}-{random.randint(10000, 99999)}"
            hospital_appointment_id = str(uuid.uuid4())
            
            logger.info(f"✅ Appointment booked successfully with {hospital_name}")
            logger.info(f"   Hospital Confirmation: {confirmation_num}")
            
            return {
                "success": True,
                "hospital_confirmation_number": confirmation_num,
                "hospital_appointment_id": hospital_appointment_id,
                "message": f"Appointment confirmed with {hospital_name}. Please arrive 15 minutes early.",
                "status": "confirmed",
                "booking_timestamp": datetime.utcnow().isoformat(),
                "hospital_name": hospital_name,
                "hospital_id": hospital_id
            }
        else:
            # Simulate booking failure (5% failure rate)
            logger.warning(f"❌ Hospital booking failed - no slots available")
            return {
                "success": False,
                "message": f"Sorry, {hospital_name} has no available slots for the requested time. Please try a different date/time.",
                "status": "failed",
                "error_code": "NO_SLOTS_AVAILABLE"
            }
            
    except Exception as e:
        logger.error(f"Error in mock booking: {str(e)}")
        return {
            "success": False,
            "message": f"Booking error: {str(e)}",
            "status": "error"
        }


async def _book_with_real_api(
    hospital_id: str,
    patient_name: str,
    appointment_date: str,
    appointment_time: str,
    specialty: str,
    patient_phone: Optional[str],
    patient_email: Optional[str],
    symptoms: Optional[str]
) -> Dict[str, Any]:
    """
    Real hospital API integration (for future use).
    
    Replace this with actual hospital API calls when available.
    """
    try:
        import httpx
        
        url = f"{HOSPITAL_BOOKING_API_URL}/{hospital_id}/appointments/book"
        headers = {
            "Content-Type": "application/json"
        }
        if os.getenv("HOSPITAL_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('HOSPITAL_API_KEY')}"
        
        payload = {
            "patient_name": patient_name,
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "specialty": specialty,
            "patient_phone": patient_phone,
            "patient_email": patient_email,
            "symptoms": symptoms
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            
            if response.status_code in [200, 201]:
                data = response.json()
                return {
                    "success": True,
                    "hospital_confirmation_number": data.get("confirmation_number"),
                    "hospital_appointment_id": data.get("appointment_id"),
                    "message": data.get("message", "Appointment confirmed"),
                    "status": "confirmed",
                    "booking_timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message": f"Hospital API error: {response.status_code}",
                    "status": "failed"
                }
                
    except Exception as e:
        logger.error(f"Real API booking error: {str(e)}")
        return {
            "success": False,
            "message": f"API error: {str(e)}",
            "status": "error"
        }


async def check_hospital_availability(
    hospital_id: str,
    appointment_date: str,
    specialty: str
) -> Dict[str, Any]:
    """
    Check hospital availability (mock for hackathon).
    
    Returns available time slots for the requested date.
    """
    # Simulate availability check
    await asyncio.sleep(0.5)
    
    # Generate mock available time slots
    available_slots = []
    for hour in range(9, 17):  # 9 AM to 5 PM
        if random.random() > 0.3:  # 70% of slots available
            available_slots.append(f"{hour:02d}:00")
            available_slots.append(f"{hour:02d}:30")
    
    return {
        "available": len(available_slots) > 0,
        "available_slots": available_slots[:10],  # Return up to 10 slots
        "date": appointment_date,
        "specialty": specialty
    }




