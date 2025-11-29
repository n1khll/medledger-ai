import os
import uvicorn
import uuid
import asyncio
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from masumi.config import Config
from masumi.payment import Payment, Amount
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from crew_definition import ExplainCrew
from shared.logging_config import setup_logging
from utils import canonical_sha256, validate_against_schema, to_epoch_seconds, epoch_or_default
from shared.decision_log import DecisionLog
from pdf_extractor import extract_text_from_pdf
from validation_handler import DoctorValidationHandler

# Configure logging
logger = setup_logging()

# Initialize validation handler (shared instance)
validation_handler = DoctorValidationHandler()

# Load environment variables
load_dotenv(override=True)

# Retrieve API Keys and URLs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY")
NETWORK = os.getenv("NETWORK")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

logger.info("Starting application with configuration:")
logger.info(f"PAYMENT_SERVICE_URL: {PAYMENT_SERVICE_URL}")
if AZURE_OPENAI_ENDPOINT:
    logger.info(f"Azure OpenAI configured: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Azure OpenAI Deployment: {AZURE_OPENAI_DEPLOYMENT}")

# Initialize FastAPI
app = FastAPI(
    title="Explain Agent - MIP-003 Compliant Medical Record Analysis",
    description="API for Explain Agent: deterministic medical record analysis with Masumi payment integration",
    version="1.0.0"
)

# Configure CORS for frontend integration
# In production, replace "*" with your frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
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
# Detect Pydantic version for compatibility
try:
    from pydantic import __version__ as pydantic_version
    pydantic_v2 = pydantic_version.startswith("2.")
except ImportError:
    pydantic_v2 = False

class StartJobRequest(BaseModel):
    identifier_from_purchaser: str = Field(..., alias="identifierFromPurchaser")
    input_data: dict
    
    if pydantic_v2:
        # Pydantic v2 configuration
        model_config = {
            "populate_by_name": True,
            "json_schema_extra": {
                "example": {
                    "identifierFromPurchaser": "demo-run-001",
                    "input_data": {
                        "patient_id": "P-001",
                        "record_text": "Patient has history of hypertension...",
                        "metadata": {"timestamp": 1717000000, "source": "hospitalA"}
                    }
                }
            }
        }
    else:
        # Pydantic v1 configuration
        class Config:
            allow_population_by_field_name = True
            schema_extra = {
                "example": {
                    "identifierFromPurchaser": "demo-run-001",
                    "input_data": {
                        "patient_id": "P-001",
                        "record_text": "Patient has history of hypertension...",
                        "metadata": {"timestamp": 1717000000, "source": "hospitalA"}
                    }
                }
            }

class ProvideInputRequest(BaseModel):
    job_id: str
    input_data: dict

class MockPaymentConfirmRequest(BaseModel):
    job_id: str

# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Task Execution
# ─────────────────────────────────────────────────────────────────────────────
async def execute_crew_task(input_data: dict, job_id: str = None) -> dict:
    """ 
    Execute the Explain Agent task with disease risk prediction and validation.
    Now includes guardrails, risk prediction, and doctor validation workflow.
    """
    logger.info(f"Starting Explain Agent task for job {job_id}")
    
    # Validate input data using jsonschema
    try:
        validate_against_schema(input_data)
    except ValueError as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create the Explain Crew
    crew = ExplainCrew(logger=logger)
    
    # Process the medical record with all enhancements
    patient_id = input_data["patient_id"]
    record_text = input_data["record_text"]
    metadata = input_data.get("metadata", None)
    
    # Enable prediction and guardrails by default
    enable_prediction = input_data.get("enable_prediction", True)
    enable_guardrails = input_data.get("enable_guardrails", True)
    
    logger.info(f"Processing with prediction={enable_prediction}, guardrails={enable_guardrails}")
    
    result = crew.process_record(
        patient_id, 
        record_text, 
        metadata,
        enable_prediction=enable_prediction,
        enable_guardrails=enable_guardrails
    )
    
    # Check if doctor validation is required
    if job_id and validation_handler.should_require_validation(result):
        logger.info(f"Doctor validation required for job {job_id}")
        validation_request = validation_handler.create_validation_request(
            job_id=job_id,
            analysis_result=result,
            patient_info={"patient_id": patient_id}
        )
        result["validation"] = validation_request
    
    logger.info("Explain Agent task completed successfully")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 1) Start Job (MIP-003: /start_job)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_job")
async def start_job(data: StartJobRequest):
    """
    Initiates a job and creates a payment request (MIP-003 compliant).
    If payment is not configured, processes immediately and returns results.
    
    Accepts medical record data from frontend:
    - patient_id: Patient identifier (string)
    - record_text: Medical record text extracted from PDF/image (string)
    - metadata: Optional metadata (object)
    
    Frontend should extract text from PDF/images and send as plain text string.
    """
    logger.info(f"Received start_job request")
    try:
        # Validate input data against schema using jsonschema
        try:
            validate_against_schema(data.input_data)
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {str(e)}. Please ensure record_text is a plain text string extracted from your document."
            )
        
        # Additional validation for frontend integration
        record_text = data.input_data.get("record_text", "")
        if not record_text or not record_text.strip():
            raise HTTPException(
                status_code=400,
                detail="record_text cannot be empty. Please extract text from your medical document and provide it as a plain text string."
            )
        
        # Warn if text is very long (but don't reject - let processing handle it)
        if len(record_text) > 100000:  # 100KB of text
            logger.warning(f"Very long record_text received ({len(record_text)} chars). Processing may take longer.")
        
        job_id = str(uuid.uuid4())
        agent_identifier = os.getenv("AGENT_IDENTIFIER")
        
        # Compute input hash using canonical hashing
        input_hash = canonical_sha256(data.input_data)
        logger.info(f"Computed input hash: {input_hash}")
        
        # Log the input (truncate record text if too long)
        truncated_text = record_text[:100] + "..." if len(record_text) > 100 else record_text
        logger.info(f"Starting job {job_id} for patient {data.input_data.get('patient_id')}")
        logger.info(f"Record text length: {len(record_text)} chars, preview: '{truncated_text}'")

        # Check if payment is configured
        payment_configured = bool(
            os.getenv("PAYMENT_SERVICE_URL") and 
            os.getenv("PAYMENT_API_KEY") and 
            agent_identifier and
            os.getenv("SELLER_VKEY")
        )
        
        payment = None
        blockchain_identifier = None
        payment_info = {}
        
        # Only set up payment if fully configured
        if payment_configured:
            try:
                # Define payment amounts
                payment_amount = os.getenv("PAYMENT_AMOUNT", "10000000")  # Default 10 ADA
                payment_unit = os.getenv("PAYMENT_UNIT", "lovelace") # Default lovelace

                amounts = [Amount(amount=payment_amount, unit=payment_unit)]
                logger.info(f"Using payment amount: {payment_amount} {payment_unit}")
                
                # Create a payment request using Masumi
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
            result_summary=f"Job created for patient {data.input_data.get('patient_id')}"
        )

        # If payment is not required, process immediately and return results
        if not payment:
            logger.info("No payment required, processing immediately with LLM")
            jobs[job_id]["status"] = "running"
            
            try:
                # Process the job with LLM
                result = await execute_crew_task(data.input_data, job_id=job_id)
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
                    result_summary=result.get("summary", "Processing completed")
                )
                
                # Return MIP-003 compliant response (even without payment, all fields required)
                # Include result in response for immediate completion
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
                    "result": result,  # Include the explanation directly
                    "output_hash": output_hash
                }
                
                logger.info(f"Job {job_id} completed successfully")
                return JSONResponse(status_code=200, content=response)
                
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}", exc_info=True)
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to analyze medical record: {str(e)}"
                )
        else:
            # Payment required - set up monitoring and return MIP-003 response
            async def payment_callback(blockchain_identifier: str):
                await handle_payment_status(job_id, blockchain_identifier)

            # Start monitoring the payment status
            payment_instances[job_id] = payment
            logger.info(f"Starting payment status monitoring for job {job_id}")
            asyncio.create_task(payment.start_status_monitoring(payment_callback))
            
            # Return the response in MIP-003 format with exact keys and epoch timestamps
            response = {
                "id": job_id,
                "job_id": job_id,
                "status": "success",  # MIP-003: "success" or "error" (job creation successful, payment pending)
                "blockchainIdentifier": blockchain_identifier,
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
    except KeyError as e:
        logger.error(f"Missing required field in request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Missing required field: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in start_job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2) Process Payment and Execute AI Task
# ─────────────────────────────────────────────────────────────────────────────
async def handle_payment_status(job_id: str, payment_id: str) -> None:
    """ Executes Explain Agent task after payment confirmation """
    try:
        logger.info(f"Payment {payment_id} completed for job {job_id}, executing task...")
        
        # Update job status to running
        jobs[job_id]["status"] = "running"
        logger.info(f"Input data: {jobs[job_id]['input_data']}")

        # Log decision
        decision_log.add_entry(
            job_id=job_id,
            actor="system",
            action="payment_confirmed",
            input_hash=jobs[job_id]["input_hash"],
            result_summary="Payment confirmed, starting processing"
        )

        # Execute the AI task with job_id for validation tracking
        result = await execute_crew_task(jobs[job_id]["input_data"], job_id=job_id)
        logger.info(f"Explain Agent task completed for job {job_id}")
        logger.info(f"Result: {result}")
        
        # Compute output hash using canonical hashing
        output_hash = canonical_sha256(result)
        logger.info(f"Computed output hash: {output_hash}")
        
        # Convert result to string for payment completion
        result_string = str(result)
        
        # Mark payment as completed on Masumi
        await payment_instances[job_id].complete_payment(payment_id, result_string)
        logger.info(f"Payment completed for job {job_id}")

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["payment_status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["output_hash"] = output_hash
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        # Log decision with output
        decision_log.add_entry(
            job_id=job_id,
            actor="agent",
            action="processed",
            input_hash=jobs[job_id]["input_hash"],
            output_hash=output_hash,
            result_summary=result.get("summary", "Processing completed")
        )

        # Stop monitoring payment status
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]
    except Exception as e:
        logger.error(f"Error processing payment {payment_id} for job {job_id}: {str(e)}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        
        # Log failure
        decision_log.add_entry(
            job_id=job_id,
            actor="system",
            action="failed",
            input_hash=jobs[job_id]["input_hash"],
            result_summary=f"Processing failed: {str(e)}"
        )
        
        # Still stop monitoring to prevent repeated failures
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# 3) Check Job and Payment Status (MIP-003: /status)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
async def get_status(job_id: str):
    """ Retrieves the current status of a specific job (MIP-003 compliant) """
    logger.info(f"Checking status for job {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Check latest payment status if payment instance exists
    if job_id in payment_instances:
        try:
            status = await payment_instances[job_id].check_payment_status()
            job["payment_status"] = status.get("data", {}).get("status")
            logger.info(f"Updated payment status for job {job_id}: {job['payment_status']}")
        except ValueError as e:
            logger.warning(f"Error checking payment status: {str(e)}")
            job["payment_status"] = "unknown"
        except Exception as e:
            logger.error(f"Error checking payment status: {str(e)}", exc_info=True)
            job["payment_status"] = "error"

    # Build response with snake_case keys (MIP-003 consistent)
    response = {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job["payment_status"],
        "input_hash": job.get("input_hash"),
        "output_hash": job.get("output_hash"),
        "result": job.get("result"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at")
    }
    
    if job["status"] == "failed":
        response["error"] = job.get("error")

    return JSONResponse(status_code=200, content=response)

# ─────────────────────────────────────────────────────────────────────────────
# 3.5) Upload PDF and Process (Convenience Endpoint)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF medical record file"),
    patient_id: str = Form(..., description="Patient identifier"),
    identifierFromPurchaser: str = Form(default="", description="Optional: User/session identifier. Auto-generated if not provided."),
    source: str = Form(default="pdf_upload", description="Source of the record")
):
    """
    Upload PDF medical record, extract text, and get AI-powered explanation.
    
    Simple flow: Upload PDF → Extract text → LLM analyzes → Return explanation
    
    This endpoint:
    1. Accepts PDF file upload
    2. Extracts text from PDF automatically
    3. Uses LLM to analyze and explain the medical record in plain language
    4. Returns the explanation immediately (if payment not configured)
    
    identifierFromPurchaser: Optional identifier for the user requesting the analysis.
    If not provided, a session-based identifier will be auto-generated.
    For frontend: Can use user ID, session ID, or leave empty for auto-generation.
    """
    logger.info(f"Received PDF upload request for patient {patient_id}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read PDF file
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty PDF file"
            )
        
        logger.info(f"Extracting text from PDF ({len(pdf_bytes)} bytes)")
        
        # Extract text from PDF
        try:
            record_text = extract_text_from_pdf(pdf_bytes)
            logger.info(f"Successfully extracted {len(record_text)} characters from PDF")
        except ValueError as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )
        
        if not record_text or not record_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from PDF. The PDF may be image-based or corrupted."
            )
        
        # Auto-generate identifierFromPurchaser if not provided
        if not identifierFromPurchaser or not identifierFromPurchaser.strip():
            identifierFromPurchaser = f"user_{int(datetime.utcnow().timestamp())}"
            logger.info(f"Auto-generated identifierFromPurchaser: {identifierFromPurchaser}")
        
        # Create MIP-003 compliant input_data
        input_data = {
            "patient_id": patient_id,
            "record_text": record_text,
            "metadata": {
                "timestamp": int(datetime.utcnow().timestamp()),
                "source": source,
                "original_filename": file.filename
            }
        }
        
        # Validate input data
        try:
            validate_against_schema(input_data)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {str(e)}"
            )
        
        job_id = str(uuid.uuid4())
        agent_identifier = os.getenv("AGENT_IDENTIFIER", "")
        
        # Compute input hash
        input_hash = canonical_sha256(input_data)
        logger.info(f"Computed input hash: {input_hash}")
        
        # Check if payment is configured
        payment_configured = bool(
            os.getenv("PAYMENT_SERVICE_URL") and 
            os.getenv("PAYMENT_API_KEY") and 
            agent_identifier and
            os.getenv("SELLER_VKEY")
        )
        
        payment = None
        blockchain_identifier = None
        payment_info = {}
        
        # Only set up payment if fully configured
        if payment_configured:
            try:
                payment_amount = os.getenv("PAYMENT_AMOUNT", "10000000")
                payment_unit = os.getenv("PAYMENT_UNIT", "lovelace")
                
                payment = Payment(
                    agent_identifier=agent_identifier,
                    config=config,
                    identifier_from_purchaser=identifierFromPurchaser,
                    input_data=input_data,
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
            "input_data": input_data,
            "input_hash": input_hash,
            "output_hash": None,
            "result": None,
            "identifier_from_purchaser": identifierFromPurchaser,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Log decision
        decision_log.add_entry(
            job_id=job_id,
            actor="system",
            action="job_created_from_pdf",
            input_hash=input_hash,
            result_summary=f"Job created from PDF upload for patient {patient_id}"
        )
        
        # If payment is not required, process immediately and return results
        if not payment:
            logger.info("No payment required, processing immediately with LLM")
            jobs[job_id]["status"] = "running"
            
            try:
                # Process the job with LLM
                result = await execute_crew_task(input_data)
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
                    result_summary=result.get("summary", "Processing completed")
                )
                
                # Return simplified response with analysis results directly
                response = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": result,  # Include the explanation directly
                    "input_hash": input_hash,
                    "output_hash": output_hash
                }
                
                logger.info(f"PDF analysis completed for patient {patient_id}")
                return JSONResponse(status_code=200, content=response)
                
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}", exc_info=True)
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to analyze medical record: {str(e)}"
                )
        else:
            # Payment required - set up monitoring and return MIP-003 response
            async def payment_callback(blockchain_identifier: str):
                await handle_payment_status(job_id, blockchain_identifier)
            
            payment_instances[job_id] = payment
            logger.info(f"Starting payment status monitoring for job {job_id}")
            asyncio.create_task(payment.start_status_monitoring(payment_callback))
            
            # Return MIP-003 compliant response (payment required)
            response = {
                "id": job_id,
                "job_id": job_id,
                "status": "awaiting_payment",
                "blockchainIdentifier": blockchain_identifier,
                "payByTime": epoch_or_default(payment_info.get("payByTime")),
                "submitResultTime": epoch_or_default(payment_info.get("submitResultTime")),
                "unlockTime": epoch_or_default(payment_info.get("unlockTime")),
                "externalDisputeUnlockTime": epoch_or_default(payment_info.get("externalDisputeUnlockTime")),
                "agentIdentifier": agent_identifier,
                "sellerVKey": os.getenv("SELLER_VKEY", ""),
                "identifierFromPurchaser": identifierFromPurchaser,
                "input_hash": input_hash,
            }
            return JSONResponse(status_code=200, content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 3.6) Get Decision Log (Optional)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/decision_log")
async def get_decision_log(job_id: str = None):
    """
    Retrieve decision log entries for a specific job or all entries.
    
    Args:
        job_id: Optional job ID to filter entries. If omitted, returns all entries.
        
    Returns:
        List of decision log entries
    """
    logger.info(f"Retrieving decision log entries for job_id: {job_id or 'all'}")
    
    entries = decision_log.get_entries(job_id=job_id)
    
    response = {
        "job_id": job_id,
        "count": len(entries),
        "entries": entries
    }
    
    return JSONResponse(status_code=200, content=response)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Check Server Availability (MIP-003: /availability)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/availability")
async def check_availability():
    """ Checks if the server is operational (MIP-003 compliant) """
    return {
        "status": "available", 
        "type": "explain-agent", 
        "message": "Explain Agent operational - Ready to process medical records"
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) Retrieve Input Schema (MIP-003: /input_schema)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/input_schema")
async def input_schema():
    """
    Returns the expected input schema for the /start_job endpoint.
    Fulfills MIP-003 /input_schema endpoint.
    
    Schema for Explain Agent:
    - patient_id (required): Patient identifier (string)
    - record_text (required): Medical record text as plain text string (extracted from PDF/image by frontend)
    - metadata (optional): Additional metadata (timestamp, source, etc.)
    
    Frontend Integration Note:
    - Extract text from PDF/images using libraries like pdf.js, pdf-parse, Tesseract.js, etc.
    - Send the extracted plain text as the record_text string field
    - Do NOT send base64 encoded files - extract text first
    """
    # MIP-003 format: Return input_data array, not JSON Schema
    return {
        "input_data": [
            {
                "id": "patient_id",
                "type": "string",
                "name": "Patient ID",
                "data": {
                    "description": "Unique identifier for the patient"
                },
                "validations": [
                    {"validation": "required"}
                ]
            },
            {
                "id": "record_text",
                "type": "string",
                "name": "Medical Record Text",
                "data": {
                    "description": "Medical record text extracted from PDF/image. Frontend should extract text and send as plain text string. Do NOT send base64 encoded files."
                },
                "validations": [
                    {"validation": "required"}
                ]
            },
            {
                "id": "metadata",
                "type": "none",
                "name": "Metadata",
                "data": {
                    "description": "Optional metadata about the record (timestamp, source, etc.)"
                }
            }
        ]
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6) Provide Additional Input (MIP-003: /provide_input)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/provide_input")
async def provide_input(data: ProvideInputRequest):
    """
    Allows providing additional input for a job that is awaiting input.
    MIP-003 compliant endpoint.
    """
    logger.info(f"Received provide_input request for job {data.job_id}")
    
    if data.job_id not in jobs:
        logger.warning(f"Job {data.job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[data.job_id]
    
    if job["status"] != "awaiting_input":
        logger.warning(f"Job {data.job_id} is not awaiting input (current status: {job['status']})")
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not awaiting input. Current status: {job['status']}"
        )
    
    try:
        # Validate the new input data using jsonschema
        validate_against_schema(data.input_data)
        
        # Update job with new input
        job["input_data"] = data.input_data
        job["input_hash"] = canonical_sha256(data.input_data)
        job["status"] = "running"
        
        logger.info(f"Updated input for job {data.job_id}, starting processing")
        
        # Log decision
        decision_log.add_entry(
            job_id=data.job_id,
            actor="user",
            action="input_provided",
            input_hash=job["input_hash"],
            result_summary="Additional input provided by user"
        )
        
        # Execute the task
        result = await execute_crew_task(job["input_data"])
        
        # Compute output hash
        output_hash = canonical_sha256(result)
        
        # Update job status
        job["status"] = "completed"
        job["result"] = result
        job["output_hash"] = output_hash
        job["completed_at"] = datetime.utcnow().isoformat()
        
        # Log completion
        decision_log.add_entry(
            job_id=data.job_id,
            actor="agent",
            action="processed",
            input_hash=job["input_hash"],
            output_hash=output_hash,
            result_summary=result.get("summary", "Processing completed")
        )
        
        response = {
            "status": "success",
            "job_id": data.job_id,
            "message": "Input provided and processing completed"
        }
        return JSONResponse(status_code=200, content=response)
    
    except ValueError as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Input validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing input for job {data.job_id}: {str(e)}", exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7) Mock Payment Confirmation (Local Dev Only)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/mock_payment_confirm")
async def mock_payment_confirm(data: MockPaymentConfirmRequest):
    """
    Mock endpoint to simulate payment confirmation for local development.
    Accepts job_id directly for simpler testing workflow.
    NOT FOR PRODUCTION USE.
    """
    job_id = data.job_id
    
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    blockchain_identifier = jobs[job_id].get("blockchain_identifier")
    if not blockchain_identifier:
        logger.error(f"No blockchain identifier found for job {job_id}")
        raise HTTPException(status_code=400, detail="No blockchain identifier for job")
    
    logger.info(f"Mock payment confirmation for job {job_id} with blockchain_identifier: {blockchain_identifier}")
    
    # Reuse existing handler that processes payment confirmation
    await handle_payment_status(job_id, blockchain_identifier)
    
    response = {
        "status": "success",
        "job_id": job_id
    }
    return JSONResponse(status_code=200, content=response)

# ─────────────────────────────────────────────────────────────────────────────
# 8) Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Returns the health of the server.
    """
    return {
        "status": "healthy"
    }

# ─────────────────────────────────────────────────────────────────────────────
# 9) Doctor Validation Endpoints (NEW)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/validate_analysis")
async def validate_analysis(
    job_id: str = Form(...),
    doctor_id: str = Form(...),
    decision: str = Form(...),  # "approve", "reject", or "modify"
    comments: Optional[str] = Form(None)
):
    """
    Doctor validates an AI analysis.
    
    Args:
        job_id: Job identifier
        doctor_id: Doctor identifier
        decision: "approve", "reject", or "modify"
        comments: Optional doctor comments
        
    Returns:
        Validation result
    """
    logger.info(f"Received validation request from doctor {doctor_id} for job {job_id}")
    
    try:
        # Submit validation
        validation = validation_handler.submit_validation(
            job_id=job_id,
            doctor_id=doctor_id,
            decision=decision,
            comments=comments
        )
        
        logger.info(f"Validation submitted: {decision}")
        
        # Get final result after validation
        final_result = validation_handler.get_final_result(job_id)
        
        return {
            "status": "success",
            "job_id": job_id,
            "validation_status": validation["status"],
            "doctor_id": doctor_id,
            "decision": decision,
            "final_result": final_result
        }
        
    except Exception as e:
        logger.error(f"Error processing validation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/pending_validations")
async def get_pending_validations():
    """
    Get all pending validations for doctor review.
    
    Returns:
        List of pending validation requests
    """
    logger.info("Fetching pending validations")
    
    try:
        pending = validation_handler.get_pending_validations()
        
        return {
            "status": "success",
            "count": len(pending),
            "pending_validations": pending
        }
        
    except Exception as e:
        logger.error(f"Error fetching pending validations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/validation_status/{job_id}")
async def get_validation_status(job_id: str):
    """
    Get validation status for a specific job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Validation status
    """
    logger.info(f"Fetching validation status for job {job_id}")
    
    try:
        validation = validation_handler.get_validation(job_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail=f"No validation found for job {job_id}")
        
        return {
            "status": "success",
            "job_id": job_id,
            "validation_status": validation["status"],
            "is_validated": validation_handler.is_validated(job_id),
            "critical_findings": validation.get("critical_findings", []),
            "created_at": validation.get("created_at"),
            "validated_at": validation.get("validated_at"),
            "doctor_id": validation.get("doctor_id"),
            "doctor_comments": validation.get("doctor_comments")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching validation status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Main Logic if Called as a Script
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Run the standalone agent flow without the API"""
    import os
    import asyncio
    # Disable execution traces to avoid terminal issues
    os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
    
    print("\n" + "=" * 70)
    print("🚀 Running Explain Agent locally (standalone mode)...")
    print("=" * 70 + "\n")
    
    # Define test input
    input_data = {
        "patient_id": "PT-12345",
        "record_text": "Patient presents with BP 150/95, HR 72 bpm, temp 98.6°F. Currently taking lisinopril 10mg daily and metformin 500mg twice daily. Patient reports occasional headaches.",
        "metadata": {
            "timestamp": 1732896000,
            "source": "clinic_visit"
        }
    }
    
    print(f"Patient ID: {input_data['patient_id']}")
    print(f"Record Text: {input_data['record_text']}")
    print("\nProcessing with Explain Agent...\n")
    
    # Initialize and run the crew
    crew = ExplainCrew(verbose=True)
    result = crew.process_record(
        input_data["patient_id"],
        input_data["record_text"],
        input_data.get("metadata")
    )
    
    # Display the result
    print("\n" + "=" * 70)
    print("✅ Explain Agent Output:")
    print("=" * 70 + "\n")
    print(f"Summary: {result['summary']}")
    print(f"\nKey Findings:")
    for finding in result['key_findings']:
        print(f"  - {finding}")
    print(f"\nConfidence Estimate: {result['confidence_estimate']}")
    print("\n" + "=" * 70 + "\n")
    
    # Compute and display hashes
    input_hash = canonical_sha256(input_data)
    output_hash = canonical_sha256(result)
    print(f"Input Hash:  {input_hash}")
    print(f"Output Hash: {output_hash}")
    print("\n" + "=" * 70 + "\n")
    
    # Ensure terminal is properly reset
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "standalone":
        # Run standalone mode (for testing)
        main()
    else:
        # Default: Run API mode
        # Railway and other platforms provide PORT env var
        port = int(os.environ.get("PORT", os.environ.get("API_PORT", 8000)))
        # Set host from environment variable, default to localhost for security.
        # Use host=0.0.0.0 to allow external connections (e.g., in Docker or production).
        # Railway requires 0.0.0.0 for external access
        host = os.environ.get("API_HOST", os.environ.get("HOST", "0.0.0.0"))

        print("\n" + "=" * 70)
        print("🚀 Starting Explain Agent API Server...")
        print("=" * 70)
        print(f"API Documentation:        http://{host}:{port}/docs")
        print(f"PDF Upload Endpoint:      http://{host}:{port}/upload_pdf")
        print(f"Start Job Endpoint:       http://{host}:{port}/start_job")
        print(f"Availability Check:       http://{host}:{port}/availability")
        print(f"Status Check:             http://{host}:{port}/status")
        print(f"Input Schema:             http://{host}:{port}/input_schema")
        print(f"\n💡 Upload PDF medical reports via /upload_pdf endpoint")
        print(f"💡 Or use /start_job with extracted text")
        print("=" * 70 + "\n")

        uvicorn.run(app, host=host, port=port, log_level="info")
