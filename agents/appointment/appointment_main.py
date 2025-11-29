"""Appointment Scheduling Agent - FastAPI application following MIP-003 standard"""
import os
import uvicorn
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from masumi.config import Config
from masumi.payment import Payment, Amount
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from appointment_crew_definition import AppointmentCrew
from shared.logging_config import setup_logging
from appointment_utils import (
    canonical_sha256, 
    validate_against_schema, 
    to_epoch_seconds, 
    epoch_or_default,
    get_input_schema
)
from shared.decision_log import DecisionLog

# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Retrieve API Keys and URLs
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY")
NETWORK = os.getenv("NETWORK", "Preprod")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

logger.info("Starting Appointment Scheduling Agent API Server...")
logger.info(f"PAYMENT_SERVICE_URL: {PAYMENT_SERVICE_URL}")
if AZURE_OPENAI_ENDPOINT:
    logger.info(f"Azure OpenAI configured: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Azure OpenAI Deployment: {AZURE_OPENAI_DEPLOYMENT}")

# Initialize FastAPI
app = FastAPI(
    title="Appointment Scheduling Agent - MIP-003 Compliant",
    description="API for Appointment Scheduling Agent: LLM-powered hospital search and appointment booking with Masumi payment integration",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Temporary in-memory job store (DO NOT USE IN PRODUCTION)
# ─────────────────────────────────────────────────────────────────────────────
jobs = {}
payment_instances = {}

# Initialize decision log
decision_log = DecisionLog()

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Masumi Payment Config
# ─────────────────────────────────────────────────────────────────────────────
config = Config(
    payment_service_url=PAYMENT_SERVICE_URL,
    payment_api_key=PAYMENT_API_KEY
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models - Version-safe configuration
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pydantic import __version__ as pydantic_version
    pydantic_v2 = pydantic_version.startswith("2.")
except ImportError:
    pydantic_v2 = False

class StartJobRequest(BaseModel):
    identifier_from_purchaser: str = Field(..., alias="identifierFromPurchaser")
    input_data: dict
    
    if pydantic_v2:
        model_config = {
            "populate_by_name": True,
            "json_schema_extra": {
                "example": {
                    "identifierFromPurchaser": "user-123",
                    "input_data": {
                        "user_request": "I need to see a cardiologist next week",
                        "pincode": "110001"
                    }
                }
            }
        }
    else:
        class Config:
            allow_population_by_field_name = True
            schema_extra = {
                "example": {
                    "identifierFromPurchaser": "user-123",
                    "input_data": {
                        "user_request": "I need to see a cardiologist next week",
                        "pincode": "110001"
                    }
                }
            }

class ProvideInputRequest(BaseModel):
    job_id: str
    status_id: str
    input_data: dict

class MockPaymentConfirmRequest(BaseModel):
    job_id: str

# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Task Execution
# ─────────────────────────────────────────────────────────────────────────────
async def execute_appointment_task(input_data: dict) -> dict:
    """Execute the Appointment Scheduling Agent task"""
    logger.info(f"Starting Appointment Scheduling Agent task")
    
    # Validate input data using jsonschema
    try:
        validate_against_schema(input_data)
    except ValueError as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create the Appointment Crew
    crew = AppointmentCrew(logger=logger)
    
    # Extract input data
    user_request = input_data["user_request"]
    pincode = input_data["pincode"]
    patient_info = input_data.get("patient_info", None)
    
    # Process the appointment request
    result = crew.process_appointment_request(user_request, pincode, patient_info)
    
    logger.info("Appointment Scheduling Agent task completed successfully")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MIP-003 Compliant Endpoints
# ─────────────────────────────────────────────────────────────────────────────

# 1) Input Schema
@app.get("/input_schema")
async def input_schema():
    """Returns the input schema for the Appointment Scheduling Agent (MIP-003 compliant)"""
    schema = get_input_schema()
    return JSONResponse(status_code=200, content=schema)

# 2) Availability
@app.get("/availability")
async def check_availability():
    """Check if the Appointment Scheduling Agent is available (MIP-003 compliant)"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "available",
            "agent": "Appointment Scheduling Agent",
            "timestamp": int(datetime.utcnow().timestamp())
        }
    )

# 3) Start Job
@app.post("/start_job")
async def start_job(data: StartJobRequest):
    """
    Initiates an appointment scheduling job and creates a payment request (MIP-003 compliant).
    
    Accepts:
    - user_request: Appointment request in plain English
    - pincode: Pincode/ZIP code for hospital search
    - patient_info: Optional patient information (may be collected via provide_input)
    """
    logger.info(f"Received start_job request for appointment scheduling")
    try:
        # Validate input data against schema
        try:
            validate_against_schema(data.input_data)
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {str(e)}"
            )
        
        user_request = data.input_data.get("user_request", "")
        pincode = data.input_data.get("pincode", "")
        patient_info = data.input_data.get("patient_info")
        
        # Validate required fields
        if not user_request or not pincode:
            raise HTTPException(
                status_code=400,
                detail="user_request and pincode are required"
            )
        
        if not patient_info:
            raise HTTPException(
                status_code=400,
                detail="patient_info is required with name, age, and location"
            )
        
        # Validate patient_info has required fields
        if not patient_info.get("name") or not patient_info.get("age") or not patient_info.get("location"):
            raise HTTPException(
                status_code=400,
                detail="patient_info must include name, age, and location"
            )
        
        patient_wallet = data.identifier_from_purchaser  # Will be used for saving appointment
        logger.info(f"Processing appointment request for: {patient_info.get('name')}, Age: {patient_info.get('age')}, Location: {patient_info.get('location')}, Pincode: {pincode}")
        
        job_id = str(uuid.uuid4())
        agent_identifier = os.getenv("AGENT_IDENTIFIER", "appointment_scheduling_agent")
        
        # Compute input hash using canonical hashing
        input_hash = canonical_sha256(data.input_data)
        logger.info(f"Computed input hash: {input_hash}")
        
        # Define payment amounts
        payment_amount = os.getenv("PAYMENT_AMOUNT", "10000000")
        payment_unit = os.getenv("PAYMENT_UNIT", "lovelace")
        
        amounts = [Amount(amount=payment_amount, unit=payment_unit)]
        logger.info(f"Using payment amount: {payment_amount} {payment_unit}")
        
        # Check if payment is configured
        payment_configured = bool(
            PAYMENT_SERVICE_URL and 
            PAYMENT_API_KEY and 
            agent_identifier and
            os.getenv("SELLER_VKEY")
        )
        
        payment = None
        blockchain_identifier = None
        payment_info = {}
        
        if payment_configured:
            try:
                payment = Payment(
                    agent_identifier=agent_identifier,
                    config=config,
                    identifier_from_purchaser=data.identifier_from_purchaser,
                    input_data=data.input_data,
                    network=NETWORK
                )
                
                logger.info("Creating payment request...")
                payment_request = await payment.create_payment_request()
                blockchain_identifier = payment_request["data"]["blockchainIdentifier"]
                payment.payment_ids.add(blockchain_identifier)
                payment_info = payment_request.get("data", {})
                logger.info(f"Created payment request with blockchain identifier: {blockchain_identifier}")
            except Exception as e:
                logger.warning(f"Payment setup failed: {str(e)}")
                payment_configured = False
                payment = None
        
        # Store job info
        jobs[job_id] = {
            "status": "awaiting_payment" if payment else "running",
            "payment_status": "pending" if payment else "not_required",
            "blockchain_identifier": blockchain_identifier,
            "input_data": data.input_data,
            "input_hash": input_hash,
            "output_hash": None,
            "result": None,
            "identifier_from_purchaser": data.identifier_from_purchaser,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Log decision
        decision_log.add_entry(
            job_id=job_id,
            actor="system",
            action="job_created",
            input_hash=input_hash,
            result_summary=f"Appointment scheduling job created for pincode: {pincode}"
        )
        
        # If payment is not required, process immediately
        if not payment:
            logger.info("No payment required, processing immediately")
            jobs[job_id]["status"] = "running"
            
            try:
                result = await execute_appointment_task(data.input_data)
                
                # Result should always be appointment_scheduled (no awaiting_input)
                # DB SAVING COMMENTED OUT FOR NOW - focus on LLM selection display
                # if result.get("status") == "appointment_scheduled" and result.get("appointment_details"):
                #     try:
                #         from appointment_storage import save_appointment
                #         appointment_id = await save_appointment(
                #             job_id=job_id,
                #             appointment_details=result["appointment_details"],
                #             patient_wallet=patient_wallet,
                #             user_request=user_request
                #         )
                #         if appointment_id:
                #             result["appointment_details"]["appointment_id"] = appointment_id
                #             logger.info(f"Appointment saved to database with ID: {appointment_id}")
                #     except Exception as e:
                #         logger.warning(f"Failed to save appointment: {str(e)}")
                
                # Appointment scheduled or completed
                output_hash = canonical_sha256(result)
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["result"] = result
                jobs[job_id]["output_hash"] = output_hash
                jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
                
                # Log completion
                decision_log.add_entry(
                    job_id=job_id,
                    actor="agent",
                    action="processed",
                    input_hash=input_hash,
                    output_hash=output_hash,
                    result_summary=result.get("message", "Appointment processing completed")
                )
                
                # Return MIP-003 compliant response with result (immediate completion)
                response = {
                    "id": job_id,
                    "job_id": job_id,
                    "status": "success",  # MIP-003 requires "success" or "error"
                    "blockchainIdentifier": "",  # Empty if no payment
                    "payByTime": 0,  # 0 if no payment required
                    "submitResultTime": 0,  # 0 if no payment required
                    "unlockTime": 0,  # 0 if no payment required
                    "externalDisputeUnlockTime": 0,  # 0 if no payment required
                    "agentIdentifier": agent_identifier or "",
                    "sellerVKey": os.getenv("SELLER_VKEY", ""),
                    "identifierFromPurchaser": data.identifier_from_purchaser,
                    "input_hash": input_hash,
                    # Additional fields for immediate completion (not in MIP-003 but useful)
                    "result": result,  # Include the appointment details directly
                    "output_hash": output_hash
                }
                logger.info(f"Job {job_id} completed successfully")
                return JSONResponse(status_code=200, content=response)
                
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}", exc_info=True)
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
                # Return error response in MIP-003 format
                response = {
                    "id": job_id,
                    "job_id": job_id,
                    "status": "error",  # MIP-003: "error" for failed jobs
                    "blockchainIdentifier": "",
                    "payByTime": 0,
                    "submitResultTime": 0,
                    "unlockTime": 0,
                    "externalDisputeUnlockTime": 0,
                    "agentIdentifier": agent_identifier or "",
                    "sellerVKey": os.getenv("SELLER_VKEY", ""),
                    "identifierFromPurchaser": data.identifier_from_purchaser,
                    "input_hash": input_hash,
                }
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process appointment request: {str(e)}"
                )
        else:
            # Set up payment monitoring
            async def payment_callback(blockchain_identifier: str):
                await handle_payment_status(job_id, blockchain_identifier)
            
            payment_instances[job_id] = payment
            logger.info(f"Starting payment status monitoring for job {job_id}")
            asyncio.create_task(payment.start_status_monitoring(payment_callback))
        
        # Get payment info from response
        payment_info = payment_info if payment else {}
        
        # Return MIP-003 compliant response (payment required)
        response = {
            "id": job_id,
            "job_id": job_id,
            "status": "success",  # MIP-003: "success" or "error" (job creation successful, payment pending)
            "blockchainIdentifier": blockchain_identifier or "",
            "payByTime": epoch_or_default(payment_info.get("payByTime")),
            "submitResultTime": epoch_or_default(payment_info.get("submitResultTime")),
            "unlockTime": epoch_or_default(payment_info.get("unlockTime")),
            "externalDisputeUnlockTime": epoch_or_default(payment_info.get("externalDisputeUnlockTime")),
            "agentIdentifier": agent_identifier,
            "sellerVKey": os.getenv("SELLER_VKEY", ""),
            "identifierFromPurchaser": data.identifier_from_purchaser,
            "input_hash": input_hash,
        }
        return JSONResponse(status_code=200, content=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in start_job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# 4) Status
@app.get("/status")
async def get_status(job_id: str = Query(..., description="Job identifier")):
    """Get the status of an appointment scheduling job (MIP-003 compliant)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    result = job.get("result")
    
    # Build response based on job status
    response = {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job.get("payment_status", "unknown"),
        "input_hash": job.get("input_hash"),
        "output_hash": job.get("output_hash"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at")
    }
    
    # If job is awaiting input, include input requirements
    if job["status"] == "awaiting_input" and result:
        response["input_groups"] = [{
            "id": "patient_info",
            "title": "Please provide patient information",
            "input_data": [
                {"id": field, "type": "string", "name": field.replace("_", " ").title()}
                for field in result.get("missing_fields", [])
            ]
        }]
    
    # If job has result, include it
    if result:
        response["result"] = result
    
    # If job failed, include error
    if job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    
    return JSONResponse(status_code=200, content=response)

# 5) Provide Input
@app.post("/provide_input")
async def provide_input(data: ProvideInputRequest):
    """
    Provide additional input for a job awaiting input (MIP-003 compliant).
    Used to provide patient information (name, age, DOB, symptoms, etc.)
    """
    if data.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[data.job_id]
    
    if job["status"] != "awaiting_input":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting input. Current status: {job['status']}"
        )
    
    # Merge provided input with existing input data
    existing_input = job["input_data"]
    existing_input["patient_info"] = data.input_data
    
    # Validate merged input
    try:
        validate_against_schema(existing_input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Input validation failed: {str(e)}")
    
    # Update input hash
    input_hash = canonical_sha256(existing_input)
    job["input_data"] = existing_input
    job["input_hash"] = input_hash
    job["status"] = "running"
    
    logger.info(f"Processing job {data.job_id} with provided input")
    
    try:
        # Process the appointment with new input
        result = await execute_appointment_task(existing_input)
        
        # Check if still awaiting input
        if result.get("status") == "awaiting_input":
            job["status"] = "awaiting_input"
            job["result"] = result
            return JSONResponse(
                status_code=200,
                content={
                    "status": "awaiting_input",
                    "job_id": data.job_id,
                    "result": result,
                    "message": "Additional information required"
                }
            )
        else:
            # Appointment scheduled - book with hospital and save to database
            if result.get("status") == "appointment_scheduled" and result.get("appointment_details"):
                try:
                    from appointment_storage import save_appointment, get_appointment_by_job_id
                    patient_wallet = job.get("identifier_from_purchaser", "")
                    user_request = existing_input.get("user_request", "")
                    appointment_id = await save_appointment(
                        job_id=data.job_id,
                        appointment_details=result["appointment_details"],
                        patient_wallet=patient_wallet,
                        user_request=user_request
                    )
                    if appointment_id:
                        result["appointment_details"]["appointment_id"] = appointment_id
                        # Add hospital confirmation to response
                        db_appointment = await get_appointment_by_job_id(data.job_id)
                        if db_appointment and db_appointment.get("hospital_confirmation_number"):
                            result["appointment_details"]["hospital_confirmation_number"] = db_appointment.get("hospital_confirmation_number")
                            result["appointment_details"]["hospital_appointment_id"] = db_appointment.get("hospital_appointment_id")
                except Exception as e:
                    logger.warning(f"Failed to book/save appointment: {str(e)}")
            
            # Appointment scheduled
            output_hash = canonical_sha256(result)
            job["status"] = "completed"
            job["result"] = result
            job["output_hash"] = output_hash
            job["completed_at"] = datetime.utcnow().isoformat()
            
            # Log completion
            decision_log.add_entry(
                job_id=data.job_id,
                actor="agent",
                action="processed",
                input_hash=input_hash,
                output_hash=output_hash,
                result_summary=result.get("message", "Appointment processing completed")
            )
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "job_id": data.job_id,
                    "result": result
                }
            )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process appointment: {str(e)}"
        )

# 6) Mock Payment Confirm (for local development)
@app.post("/mock_payment_confirm")
async def mock_payment_confirm(data: MockPaymentConfirmRequest):
    """Mock payment confirmation endpoint for local development"""
    job_id = data.job_id
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    blockchain_identifier = jobs[job_id].get("blockchain_identifier")
    if not blockchain_identifier:
        raise HTTPException(status_code=400, detail="No blockchain identifier for job")
    
    await handle_payment_status(job_id, blockchain_identifier)
    
    return JSONResponse(
        status_code=200,
        content={"status": "success", "job_id": job_id}
    )

# ─────────────────────────────────────────────────────────────────────────────
# Payment Status Handler
# ─────────────────────────────────────────────────────────────────────────────
async def handle_payment_status(job_id: str, blockchain_identifier: str) -> None:
    """Handle payment status updates and process job when payment confirmed"""
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found when handling payment status")
        return
    
    job = jobs[job_id]
    
    if job["status"] != "awaiting_payment":
        logger.info(f"Job {job_id} is not awaiting payment. Current status: {job['status']}")
        return
    
    logger.info(f"Payment confirmed for job {job_id}, starting appointment processing")
    job["status"] = "running"
    job["payment_status"] = "confirmed"
    
    try:
        input_data = job["input_data"]
        result = await execute_appointment_task(input_data)
        
        # Check if result indicates awaiting input
        if result.get("status") == "awaiting_input":
            job["status"] = "awaiting_input"
            job["result"] = result
            logger.info(f"Job {job_id} awaiting additional input after payment")
        else:
            # Appointment scheduled - book with hospital and save to database
            if result.get("status") == "appointment_scheduled" and result.get("appointment_details"):
                try:
                    from appointment_storage import save_appointment, get_appointment_by_job_id
                    patient_wallet = job.get("identifier_from_purchaser", "")
                    user_request = input_data.get("user_request", "")
                    appointment_id = await save_appointment(
                        job_id=job_id,
                        appointment_details=result["appointment_details"],
                        patient_wallet=patient_wallet,
                        user_request=user_request
                    )
                    if appointment_id:
                        result["appointment_details"]["appointment_id"] = appointment_id
                        # Add hospital confirmation to response
                        db_appointment = await get_appointment_by_job_id(job_id)
                        if db_appointment and db_appointment.get("hospital_confirmation_number"):
                            result["appointment_details"]["hospital_confirmation_number"] = db_appointment.get("hospital_confirmation_number")
                            result["appointment_details"]["hospital_appointment_id"] = db_appointment.get("hospital_appointment_id")
                except Exception as e:
                    logger.warning(f"Failed to book/save appointment: {str(e)}")
            
            # Appointment scheduled
            output_hash = canonical_sha256(result)
            job["status"] = "completed"
            job["result"] = result
            job["output_hash"] = output_hash
            job["completed_at"] = datetime.utcnow().isoformat()
            
            # Log completion
            decision_log.add_entry(
                job_id=job_id,
                actor="agent",
                action="processed",
                input_hash=job["input_hash"],
                output_hash=output_hash,
                result_summary=result.get("message", "Appointment processing completed")
            )
            
            logger.info(f"Appointment processing completed for job {job_id}")
    except Exception as e:
        logger.error(f"Error processing appointment for job {job_id}: {str(e)}", exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Decision Log Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/decision_log")
async def get_decision_log(job_id: str = Query(None, description="Optional job ID to filter")):
    """Get decision log entries, optionally filtered by job_id"""
    entries = decision_log.get_entries(job_id=job_id)
    return JSONResponse(status_code=200, content={"entries": entries})

# ─────────────────────────────────────────────────────────────────────────────
# Get Appointments by Patient
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/appointments")
async def get_appointments(patient_wallet: str = Query(..., description="Patient wallet address")):
    """
    Get all appointments for a patient.
    
    Returns list of appointments sorted by date (newest first).
    """
    try:
        from appointment_storage import get_appointments_by_patient
        appointments = await get_appointments_by_patient(patient_wallet)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "patient_wallet": patient_wallet,
                "appointments": appointments,
                "count": len(appointments)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving appointments: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve appointments: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "agent": "Appointment Scheduling Agent",
            "timestamp": int(datetime.utcnow().timestamp())
        }
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    # Railway and other platforms provide PORT env var
    port = int(os.environ.get("PORT", os.environ.get("API_PORT", 8001)))
    # Railway requires 0.0.0.0 for external access
    host = os.environ.get("API_HOST", os.environ.get("HOST", "0.0.0.0"))
    
    print("\n" + "=" * 70)
    print("🚀 Starting Appointment Scheduling Agent API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://{host}:{port}/docs")
    print(f"Start Job Endpoint:       http://{host}:{port}/start_job")
    print(f"Availability Check:       http://{host}:{port}/availability")
    print(f"Status Check:             http://{host}:{port}/status")
    print(f"Input Schema:             http://{host}:{port}/input_schema")
    print("\n💡 This agent schedules medical appointments based on user requests")
    print("💡 Uses LLM to understand natural language and find nearby hospitals")
    print("=" * 70 + "\n")
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

