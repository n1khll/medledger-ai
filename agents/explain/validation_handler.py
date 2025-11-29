"""Doctor-in-the-loop validation handler for critical medical predictions"""
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)


class ValidationStatus(Enum):
    """Status of a validation request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class DoctorValidationHandler:
    """
    Handles doctor validation for critical findings.
    Implements human-in-the-loop workflow for high-risk predictions.
    """
    
    def __init__(self):
        """Initialize the validation handler"""
        self.logger = logger
        self.pending_validations = {}  # job_id -> validation_data
    
    def should_require_validation(self, analysis_result: Dict[str, Any]) -> bool:
        """
        Determine if doctor validation is required based on analysis results.
        
        Triggers validation if:
        - High-risk or critical predictions
        - Low confidence scores
        - Validation failures
        - Explicit flags from agents
        
        Args:
            analysis_result: Complete analysis result including risk predictions
            
        Returns:
            True if validation is required, False otherwise
        """
        try:
            # Check if explicitly flagged for review
            if analysis_result.get("requires_doctor_review", False):
                self.logger.info("Doctor review explicitly required")
                return True
            
            # Check overall confidence
            confidence = analysis_result.get("confidence_estimate", 1.0)
            if confidence < 0.70:
                self.logger.info(f"Low confidence ({confidence}) - requiring doctor review")
                return True
            
            # Check risk predictions
            risk_prediction = analysis_result.get("risk_prediction")
            if risk_prediction:
                # Check overall risk level
                overall_risk = risk_prediction.get("overall_risk", "").upper()
                if overall_risk in ["HIGH", "CRITICAL"]:
                    self.logger.info(f"Overall risk is {overall_risk} - requiring doctor review")
                    return True
                
                # Check if any assessment requires review
                if risk_prediction.get("requires_doctor_review", False):
                    self.logger.info("Risk prediction flagged for doctor review")
                    return True
                
                # Check individual risk assessments
                risk_assessments = risk_prediction.get("risk_assessments", [])
                for risk in risk_assessments:
                    risk_level = risk.get("risk_level", "").upper()
                    if risk_level in ["HIGH", "CRITICAL"]:
                        self.logger.info(f"{risk.get('disease')} has {risk_level} risk - requiring doctor review")
                        return True
                    
                    # Check confidence per assessment
                    assessment_confidence = risk.get("confidence", 1.0)
                    if assessment_confidence < 0.70:
                        self.logger.info(f"{risk.get('disease')} has low confidence ({assessment_confidence}) - requiring doctor review")
                        return True
            
            # Check if validation failed
            validation_result = analysis_result.get("validation_result")
            if validation_result and not validation_result.get("is_valid", True):
                self.logger.info("Validation failed - requiring doctor review")
                return True
            
            self.logger.info("No doctor review required")
            return False
            
        except Exception as e:
            self.logger.error(f"Error determining validation requirement: {str(e)}")
            # Fail safe - require validation on error
            return True
    
    def create_validation_request(self, job_id: str, analysis_result: Dict[str, Any], 
                                 patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a validation request for doctor review.
        
        Args:
            job_id: Unique job identifier
            analysis_result: Complete analysis result
            patient_info: Optional patient information
            
        Returns:
            Validation request dictionary
        """
        try:
            # Extract critical findings
            critical_findings = self._extract_critical_findings(analysis_result)
            
            # Create validation request
            validation_data = {
                "job_id": job_id,
                "status": ValidationStatus.PENDING.value,
                "analysis_result": analysis_result,
                "patient_info": patient_info,
                "critical_findings": critical_findings,
                "created_at": datetime.utcnow().isoformat(),
                "doctor_id": None,
                "doctor_comments": None,
                "doctor_decision": None,
                "validated_at": None
            }
            
            # Store in pending validations
            self.pending_validations[job_id] = validation_data
            
            self.logger.info(f"Created validation request for job {job_id}")
            self.logger.info(f"Critical findings: {len(critical_findings)}")
            
            # Return validation request for API response
            return {
                "requires_validation": True,
                "validation_id": job_id,
                "message": "This analysis requires doctor review before finalizing.",
                "critical_findings": critical_findings,
                "pending_review": True
            }
            
        except Exception as e:
            self.logger.error(f"Error creating validation request: {str(e)}")
            return {
                "requires_validation": True,
                "validation_id": job_id,
                "message": "Error creating validation request. Doctor review required.",
                "error": str(e),
                "pending_review": True
            }
    
    def _extract_critical_findings(self, analysis_result: Dict) -> List[str]:
        """
        Extract critical findings that need doctor review.
        
        Args:
            analysis_result: Complete analysis result
            
        Returns:
            List of critical finding descriptions
        """
        findings = []
        
        try:
            # Extract from risk predictions
            risk_prediction = analysis_result.get("risk_prediction")
            if risk_prediction:
                risk_assessments = risk_prediction.get("risk_assessments", [])
                
                for risk in risk_assessments:
                    risk_level = risk.get("risk_level", "").upper()
                    
                    if risk_level in ["HIGH", "CRITICAL"]:
                        disease = risk.get("disease", "Unknown condition")
                        risk_score = risk.get("risk_score", 0)
                        indicators = risk.get("indicators", [])
                        
                        finding = f"{disease}: {risk_level} risk (Score: {risk_score:.2f})"
                        if indicators:
                            finding += f" - Key indicators: {', '.join(indicators[:3])}"
                        
                        findings.append(finding)
            
            # Extract from key findings if confidence is low
            confidence = analysis_result.get("confidence_estimate", 1.0)
            if confidence < 0.70:
                findings.append(f"Low analysis confidence: {confidence:.2f} - Additional review recommended")
            
            # Extract from validation issues
            validation_result = analysis_result.get("validation_result")
            if validation_result and not validation_result.get("is_valid", True):
                issues = validation_result.get("issues", [])
                for issue in issues[:3]:  # Limit to first 3 issues
                    findings.append(f"Validation concern: {issue}")
            
            self.logger.info(f"Extracted {len(findings)} critical findings")
            
        except Exception as e:
            self.logger.error(f"Error extracting critical findings: {str(e)}")
            findings.append("Error extracting findings - review recommended")
        
        return findings
    
    def submit_validation(self, job_id: str, doctor_id: str, 
                         decision: str, comments: Optional[str] = None,
                         modified_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit doctor validation decision.
        
        Args:
            job_id: Job identifier
            doctor_id: Doctor identifier
            decision: "approve", "reject", or "modify"
            comments: Optional doctor comments
            modified_result: Optional modified analysis (for "modify" decision)
            
        Returns:
            Updated validation data
        """
        try:
            if job_id not in self.pending_validations:
                raise ValueError(f"No pending validation found for job {job_id}")
            
            validation = self.pending_validations[job_id]
            
            # Update validation status
            if decision.lower() == "approve":
                validation["status"] = ValidationStatus.APPROVED.value
                self.logger.info(f"Job {job_id} approved by doctor {doctor_id}")
            elif decision.lower() == "reject":
                validation["status"] = ValidationStatus.REJECTED.value
                self.logger.info(f"Job {job_id} rejected by doctor {doctor_id}")
            elif decision.lower() == "modify":
                validation["status"] = ValidationStatus.MODIFIED.value
                if modified_result:
                    validation["modified_result"] = modified_result
                self.logger.info(f"Job {job_id} modified by doctor {doctor_id}")
            else:
                raise ValueError(f"Invalid decision: {decision}")
            
            # Update validation data
            validation["doctor_id"] = doctor_id
            validation["doctor_comments"] = comments
            validation["doctor_decision"] = decision.lower()
            validation["validated_at"] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Validation submitted for job {job_id}")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error submitting validation: {str(e)}")
            raise
    
    def get_pending_validations(self) -> List[Dict[str, Any]]:
        """
        Get all pending validations.
        
        Returns:
            List of pending validation requests
        """
        pending = []
        
        for job_id, validation in self.pending_validations.items():
            if validation["status"] == ValidationStatus.PENDING.value:
                # Return summary (not full analysis result)
                pending.append({
                    "job_id": job_id,
                    "created_at": validation["created_at"],
                    "critical_findings": validation["critical_findings"],
                    "patient_info": validation.get("patient_info", {})
                })
        
        self.logger.info(f"Found {len(pending)} pending validations")
        return pending
    
    def get_validation(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get validation data for a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Validation data or None
        """
        return self.pending_validations.get(job_id)
    
    def is_validated(self, job_id: str) -> bool:
        """
        Check if a job has been validated.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if validated (approved/rejected/modified), False if pending or not found
        """
        validation = self.pending_validations.get(job_id)
        if not validation:
            return False
        
        status = validation["status"]
        return status in [ValidationStatus.APPROVED.value, 
                         ValidationStatus.REJECTED.value, 
                         ValidationStatus.MODIFIED.value]
    
    def get_final_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the final result after validation.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Final result with validation status
        """
        validation = self.pending_validations.get(job_id)
        if not validation:
            return None
        
        # Get base result
        result = validation["analysis_result"].copy()
        
        # Add validation information
        result["validation_status"] = validation["status"]
        result["validated_by"] = validation.get("doctor_id")
        result["doctor_comments"] = validation.get("doctor_comments")
        result["validated_at"] = validation.get("validated_at")
        
        # If modified, use modified result
        if validation["status"] == ValidationStatus.MODIFIED.value:
            modified = validation.get("modified_result")
            if modified:
                result.update(modified)
        
        # If rejected, add rejection notice
        if validation["status"] == ValidationStatus.REJECTED.value:
            result["rejected"] = True
            result["rejection_reason"] = validation.get("doctor_comments", "Analysis rejected by doctor")
        
        return result

