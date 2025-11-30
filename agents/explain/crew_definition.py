"""Explain Agent Crew Definition - LLM-powered medical record analysis with disease risk prediction"""
import os
import json
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# Try to import OpenAI LLM
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import our new components
from guardrails import MedicalGuardrails
from disease_predictor import DiseaseRiskPredictor
from medical_rules import MedicalRuleValidator

# Load environment variables
load_dotenv(override=True)


class ExplainCrew:
    """
    LLM-powered Explain Agent for medical record analysis.
    Uses OpenAI to intelligently analyze and explain medical records.
    
    Configured with OpenAI using environment variables:
    - OPENAI_API_KEY (required)
    - OPENAI_MODEL (optional, defaults to gpt-4o-mini)
    """
    
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        
        # Configure OpenAI
        self.openai_configured = self._configure_openai()
        
        if not self.openai_configured:
            self.logger.warning("OpenAI not configured. LLM features will not work.")
            self.logger.warning("Please set OPENAI_API_KEY (and optionally OPENAI_MODEL)")
        
        self.logger.info("ExplainCrew initialized")
    
    def _configure_openai(self) -> bool:
        """
        Configure OpenAI for LangChain and CrewAI compatibility.
        
        Returns:
            True if OpenAI is configured, False otherwise
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if openai_api_key:
            # Set environment variable for LangChain
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
            self.logger.info(f"OpenAI configured")
            self.logger.info(f"Model: {openai_model}")
            self.logger.info("Using LangChain ChatOpenAI")
            return True
        else:
            return False
    
    def _create_llm(self):
        """Create and return an OpenAI LLM instance."""
        if not self.openai_configured:
            raise ValueError("OpenAI not configured. Cannot create LLM instance.")
        
        return self._create_openai_llm()
    
    def _create_openai_llm(self):
        """Create and return an OpenAI LLM instance."""
        if not OPENAI_AVAILABLE:
            raise ValueError(
                "langchain-openai package not installed. "
                "Install with: pip install langchain-openai"
            )
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not openai_api_key:
            raise ValueError("Missing required OpenAI configuration: OPENAI_API_KEY")
        
        try:
            llm = ChatOpenAI(
                model=openai_model,
                api_key=openai_api_key,
                temperature=0.7,
                verbose=self.verbose
            )
            
            self.logger.info(f"Created LangChain ChatOpenAI instance")
            self.logger.info(f"  Model: {openai_model}")
            
            return llm
        except Exception as e:
            self.logger.error(f"Failed to create OpenAI LLM: {str(e)}")
            raise ValueError(
                f"Failed to initialize OpenAI LLM. "
                f"Error: {str(e)}"
            )
    
    def _create_crew(self, patient_id: str, record_text: str):
        """Create the CrewAI crew with agent and task for a specific medical record."""
        if not self.openai_configured:
            raise ValueError("OpenAI not configured. Cannot create LLM-powered agent.")
        
        # Create the LLM instance explicitly
        llm = self._create_llm()
        
        # Create the Medical Record Analyst Agent with explicit LLM
        medical_analyst = Agent(
            role="Medical Record Analyst",
            goal="Analyze medical records thoroughly and provide clear, accurate explanations of patient conditions, vital signs, medications, and clinical findings in plain language that patients can understand.",
            backstory="""You are an experienced medical analyst with deep expertise in interpreting clinical documentation. 
            You have years of experience reviewing medical records, understanding medical terminology, and translating 
            complex clinical information into clear, understandable explanations. You are meticulous, accurate, and 
            always prioritize patient safety and understanding. You excel at identifying key clinical findings, 
            understanding medication implications, and explaining medical conditions in accessible language.""",
            verbose=self.verbose,
            allow_delegation=False,
            llm=llm  # Explicitly pass the LLM instance
        )
        
        # Format task description with actual record text
        task_description = f"""Analyze the following medical record and provide a comprehensive explanation.

Your analysis should:
1. Extract and explain all vital signs (blood pressure, heart rate, temperature, etc.)
2. Identify all medications mentioned and explain their purpose
3. Identify any abnormal findings or concerns
4. Provide a clear summary of the patient's condition
5. Explain any clinical findings in plain language

Return your analysis as a JSON object with the following structure:
{{
    "summary": "A clear, comprehensive summary of the medical record analysis",
    "key_findings": [
        "Finding 1: Detailed explanation",
        "Finding 2: Detailed explanation",
        ...
    ],
    "confidence_estimate": <calculate dynamically between 0.0 and 1.0>
}}

The summary should be 2-3 sentences explaining the overall patient condition.
Key findings should be a list of important observations, each with clear explanations.

IMPORTANT - Confidence Estimate Calculation:
You MUST calculate confidence_estimate dynamically for THIS specific record. Do NOT use a fixed value.
Evaluate the record based on:
- Clarity and readability: How clear and well-formatted is the record? (0.0 = unreadable/garbled, 1.0 = perfectly clear)
- Completeness: How complete is the information? Missing vital signs, medications, or key findings = lower confidence
- Structure: How well-organized is the record? Scattered or disorganized information = lower confidence
- Data quality: Are values reasonable and consistent? Inconsistent or suspicious data = lower confidence

Return a specific decimal value (e.g., 0.87, 0.92, 0.75, 0.65) based on your assessment of THIS record.
The value should reflect your confidence in the accuracy and completeness of your analysis for this specific medical record.

Medical Record:
{record_text}

Patient ID: {patient_id}
"""
        
        # Create the analysis task
        analysis_task = Task(
            description=task_description,
            agent=medical_analyst,
            expected_output="JSON object with summary, key_findings array, and confidence_estimate"
        )
        
        # Create the crew
        crew = Crew(
            agents=[medical_analyst],
            tasks=[analysis_task],
            verbose=self.verbose
        )
        
        return crew
    
    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            Dictionary with summary, key_findings, and confidence_estimate
        """
        try:
            # Try to extract JSON from the response
            # LLM might return JSON wrapped in markdown code blocks
            output = llm_output.strip()
            
            # Remove markdown code blocks if present
            if output.startswith("```json"):
                output = output[7:]
            elif output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
            
            # Parse JSON
            parsed = json.loads(output)
            
            # Validate structure
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a dictionary")
            
            # Ensure required fields
            result = {
                "summary": parsed.get("summary", "Medical record analysis completed."),
                "key_findings": parsed.get("key_findings", []),
                "confidence_estimate": float(parsed.get("confidence_estimate", 0.85))
            }
            
            # Validate key_findings is a list
            if not isinstance(result["key_findings"], list):
                result["key_findings"] = [str(result["key_findings"])]
            
            # Ensure confidence is between 0 and 1
            result["confidence_estimate"] = max(0.0, min(1.0, result["confidence_estimate"]))
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            self.logger.error(f"Raw output: {llm_output[:500]}")
            # Fallback: create structured response from raw text
            return {
                "summary": f"Medical record analysis completed. {llm_output[:200]}...",
                "key_findings": [llm_output] if llm_output else ["Analysis completed but response format was unexpected."],
                "confidence_estimate": 0.70
            }
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "summary": "Medical record analysis completed with some parsing issues.",
                "key_findings": ["Analysis completed but response parsing encountered an error."],
                "confidence_estimate": 0.60
            }
    
    def process_record(self, patient_id: str, record_text: str, metadata: Dict[str, Any] = None,
                      enable_prediction: bool = True, enable_guardrails: bool = True) -> Dict[str, Any]:
        """
        Process a medical record using LLM with guardrails, risk prediction, and validation.
        
        Args:
            patient_id: Patient identifier
            record_text: Medical record text
            metadata: Optional metadata dictionary
            enable_prediction: Enable disease risk prediction (default: True)
            enable_guardrails: Enable input/output guardrails (default: True)
            
        Returns:
            Dictionary with analysis, risk predictions, and validation results
        """
        self.logger.info(f"Processing record for patient {patient_id} using LLM")
        self.logger.info(f"Prediction enabled: {enable_prediction}, Guardrails enabled: {enable_guardrails}")
        
        if not self.openai_configured:
            raise ValueError(
                "OpenAI not configured. Please set OPENAI_API_KEY (and optionally OPENAI_MODEL)"
            )
        
        try:
            # Create LLM instance
            llm = self._create_llm()
            
            # Step 1: Input Guardrails (if enabled)
            checked_record_text = record_text
            if enable_guardrails:
                self.logger.info("Running input guardrails")
                guardrails = MedicalGuardrails(llm)
                is_safe, result = guardrails.check_input(record_text)
                
                if not is_safe:
                    self.logger.warning(f"Input validation failed: {result}")
                    raise ValueError(f"Input validation failed: {result}")
                
                checked_record_text = result
                self.logger.info("Input guardrails passed")
            
            # Step 2: Basic Medical Record Analysis
            self.logger.info("Starting basic medical record analysis")
            crew = self._create_crew(patient_id, checked_record_text)
            
            self.logger.info("Executing LLM-powered analysis...")
            result = crew.kickoff()
            
            # Extract the output (CrewAI returns a CrewOutput object)
            if hasattr(result, 'raw'):
                llm_output = result.raw
            elif hasattr(result, 'content'):
                llm_output = result.content
            else:
                llm_output = str(result)
            
            self.logger.info(f"LLM analysis completed. Output length: {len(llm_output)}")
            
            # Parse the LLM response into structured format
            parsed_result = self._parse_llm_response(llm_output)
            
            self.logger.info(f"Basic analysis complete. Confidence: {parsed_result['confidence_estimate']}")
            self.logger.info(f"Found {len(parsed_result['key_findings'])} key findings")
            
            # Step 3: Disease Risk Prediction (if enabled)
            risk_prediction = None
            validation_result = None
            
            if enable_prediction:
                self.logger.info("Starting disease risk prediction")
                predictor = DiseaseRiskPredictor(llm, verbose=self.verbose)
                
                # Extract patient info from metadata if available
                patient_info = None
                if metadata:
                    patient_info = {
                        "age": metadata.get("age"),
                        "gender": metadata.get("gender")
                    }
                
                risk_prediction = predictor.predict_risks(checked_record_text, patient_info)
                self.logger.info(f"Risk prediction complete. Overall risk: {risk_prediction.get('overall_risk')}")
                
                # Step 4: Rule-based validation of predictions
                if risk_prediction and risk_prediction.get("risk_assessments"):
                    self.logger.info("Validating risk predictions against medical rules")
                    validator = MedicalRuleValidator()
                    
                    all_valid = True
                    all_issues = []
                    
                    for assessment in risk_prediction["risk_assessments"]:
                        is_valid, issues = validator.validate_risk_prediction(checked_record_text, assessment)
                        if not is_valid:
                            all_valid = False
                            all_issues.extend(issues)
                            self.logger.warning(f"Validation failed for {assessment.get('disease')}: {issues}")
                    
                    validation_result = {
                        "is_valid": all_valid,
                        "issues": all_issues
                    }
                    
                    if not all_valid:
                        self.logger.warning("Risk prediction validation failed - flagging for review")
                        risk_prediction["requires_doctor_review"] = True
            
            # Step 5: Combine results
            final_result = {
                **parsed_result,
                "risk_prediction": risk_prediction,
                "validation_result": validation_result
            }
            
            # Step 6: Output Guardrails (if enabled)
            if enable_guardrails:
                self.logger.info("Running output guardrails")
                # Convert result to string for guardrails check
                output_str = json.dumps(final_result, indent=2)
                sanitized_output = guardrails.check_output(output_str, checked_record_text)
                
                # Try to parse back if it's still valid JSON
                try:
                    sanitized_result = json.loads(sanitized_output)
                    final_result = sanitized_result
                except json.JSONDecodeError:
                    # If sanitized output is not JSON, add it as a note
                    final_result["guardrails_note"] = sanitized_output
                
                self.logger.info("Output guardrails applied")
            
            self.logger.info("Processing complete")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error during LLM processing: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process medical record with LLM: {str(e)}")
