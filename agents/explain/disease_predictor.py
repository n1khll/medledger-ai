"""Disease Risk Predictor Agent with anti-hallucination techniques"""
import json
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from crewai import Agent, Task, Crew

logger = get_logger(__name__)


class DiseaseRiskPredictor:
    """
    Predicts early disease risks from medical records using LLM with anti-hallucination safeguards.
    """
    
    def __init__(self, llm, verbose=True):
        """
        Initialize the disease risk predictor.
        
        Args:
            llm: Language model instance
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.verbose = verbose
        self.logger = logger
    
    def predict_risks(self, record_text: str, patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze medical record for early disease risk prediction.
        
        Args:
            record_text: Medical record text
            patient_info: Optional patient demographics (age, gender, etc.)
            
        Returns:
            Dictionary with risk assessments
        """
        self.logger.info("Starting disease risk prediction")
        
        try:
            # Create the risk predictor agent with anti-hallucination prompts
            predictor_agent = Agent(
                role="Medical Risk Assessment Specialist",
                goal="Analyze medical records to identify early warning signs and predict disease risks based ONLY on evidence found in the record",
                backstory="""You are an expert in preventive medicine and early disease detection with 20 years of experience.
                You analyze medical records to identify risk factors and early warning signs.
                You are EXTREMELY CAREFUL to only use information explicitly stated in the medical record.
                You NEVER invent or assume values that aren't in the record.
                You provide risk assessments (NOT diagnoses) and always recommend professional medical consultation.
                Your predictions are evidence-based and you cite exact quotes from the record to support each assessment.""",
                verbose=self.verbose,
                allow_delegation=False,
                llm=self.llm
            )
            
            # Build patient context
            patient_context = ""
            if patient_info:
                patient_context = f"""
Patient Demographics (if available):
- Age: {patient_info.get('age', 'Not specified')}
- Gender: {patient_info.get('gender', 'Not specified')}
"""
            
            # Create task with anti-hallucination constraints and few-shot examples
            task_description = self._build_prediction_prompt(record_text, patient_context)
            
            prediction_task = Task(
                description=task_description,
                agent=predictor_agent,
                expected_output="JSON object with risk_assessments array, overall_risk, and data_quality"
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[predictor_agent],
                tasks=[prediction_task],
                verbose=self.verbose
            )
            
            self.logger.info("Executing risk prediction analysis")
            result = crew.kickoff()
            
            # Extract output
            if hasattr(result, 'raw'):
                output = result.raw
            elif hasattr(result, 'content'):
                output = result.content
            else:
                output = str(result)
            
            self.logger.info("Risk prediction completed")
            
            # Parse the result
            parsed_result = self._parse_prediction(output)
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error during risk prediction: {str(e)}", exc_info=True)
            return {
                "risk_assessments": [],
                "overall_risk": "UNKNOWN",
                "data_quality": "ERROR",
                "error": str(e),
                "requires_doctor_review": True
            }
    
    def _build_prediction_prompt(self, record_text: str, patient_context: str) -> str:
        """Build the prediction prompt with anti-hallucination constraints and few-shot examples"""
        
        prompt = f"""Analyze the following medical record for early disease risk prediction.

{patient_context}

Medical Record:
{record_text}

CRITICAL RULES TO PREVENT HALLUCINATION:
1. Use ONLY values explicitly stated in the medical record above
2. If a value is missing, you MUST state "INSUFFICIENT DATA" - do NOT guess or invent values
3. For each risk prediction, you MUST list exact quotes from the record
4. Never claim a value exists if it's not in the record
5. If you're uncertain, mark confidence as LOW and flag for doctor review

FEW-SHOT EXAMPLES (Learn from these correct patterns):

Example 1 - Correct Analysis with Complete Data:
Record: "Patient: 52 years, BP: 148/92, Glucose: 118, HbA1c: 6.2%, Family history: diabetes"
Correct Output:
{{
  "risk_assessments": [
    {{
      "disease": "Type 2 Diabetes",
      "risk_level": "HIGH",
      "risk_score": 0.75,
      "evidence_quotes": ["Glucose: 118", "HbA1c: 6.2%", "Family history: diabetes"],
      "indicators": ["Pre-diabetic glucose (118 mg/dL)", "Pre-diabetic HbA1c (6.2%)", "Family history present"],
      "missing_data": ["BMI", "Cholesterol"],
      "recommendations": ["Consult endocrinologist", "HbA1c monitoring every 3 months", "Lifestyle modifications"],
      "confidence": 0.85,
      "requires_doctor_review": true
    }},
    {{
      "disease": "Hypertension",
      "risk_level": "MODERATE",
      "risk_score": 0.65,
      "evidence_quotes": ["BP: 148/92"],
      "indicators": ["Stage 1 Hypertension (148/92 mmHg)"],
      "missing_data": [],
      "recommendations": ["Monitor BP regularly", "Lifestyle changes"],
      "confidence": 0.90,
      "requires_doctor_review": false
    }}
  ],
  "overall_risk": "HIGH",
  "data_quality": "COMPLETE"
}}

Example 2 - Correct Analysis with Normal Values:
Record: "Patient: 30 years, BP: 118/75, Glucose: 92"
Correct Output:
{{
  "risk_assessments": [
    {{
      "disease": "Type 2 Diabetes",
      "risk_level": "LOW",
      "risk_score": 0.15,
      "evidence_quotes": ["Glucose: 92"],
      "indicators": ["Normal glucose (92 mg/dL)"],
      "missing_data": ["HbA1c", "Family history", "BMI"],
      "recommendations": ["Continue healthy lifestyle", "Regular checkups"],
      "confidence": 0.70,
      "requires_doctor_review": false
    }}
  ],
  "overall_risk": "LOW",
  "data_quality": "INCOMPLETE"
}}

Example 3 - Correct Analysis with Missing Data:
Record: "Patient: 45 years, Family history: diabetes, hypertension"
Correct Output:
{{
  "risk_assessments": [
    {{
      "disease": "Type 2 Diabetes",
      "risk_level": "MODERATE",
      "risk_score": 0.40,
      "evidence_quotes": ["Family history: diabetes"],
      "indicators": ["Family history of diabetes"],
      "missing_data": ["Glucose", "HbA1c", "BMI", "BP"],
      "recommendations": ["Get glucose and HbA1c tested", "Consult healthcare provider"],
      "confidence": 0.50,
      "requires_doctor_review": true
    }}
  ],
  "overall_risk": "MODERATE",
  "data_quality": "INSUFFICIENT"
}}

YOUR TASK:
Analyze the medical record above following these exact patterns.

For diseases to assess, focus on:
- Type 2 Diabetes (check: glucose, HbA1c, family history, age, BMI)
- Hypertension (check: BP values, age, family history)
- Heart Disease (check: cholesterol, BP, age, smoking status, family history)

For EACH disease with evidence in the record, provide:
1. disease: Name of the disease
2. risk_level: "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
3. risk_score: Number between 0.0 and 1.0
4. evidence_quotes: EXACT quotes from the record (must exist in record!)
5. indicators: List of specific risk factors found
6. missing_data: List of what data is missing for complete assessment
7. recommendations: Preventive actions
8. confidence: 0.0-1.0 based on data completeness
9. requires_doctor_review: true if risk is HIGH/CRITICAL or confidence < 0.70

Risk Score Guidelines:
- 0.0-0.3: LOW risk (normal values, no risk factors)
- 0.3-0.6: MODERATE risk (borderline values, some risk factors)
- 0.6-0.8: HIGH risk (abnormal values, multiple risk factors)
- 0.8-1.0: CRITICAL risk (severely abnormal values, urgent attention needed)

Return as JSON:
{{
  "risk_assessments": [/* array of assessments */],
  "overall_risk": "LOW|MODERATE|HIGH|CRITICAL",
  "data_quality": "COMPLETE|INCOMPLETE|INSUFFICIENT"
}}

IMPORTANT:
- If you don't find evidence for a disease, don't include it
- Always include evidence_quotes that can be verified in the record
- If data is missing, lower your confidence and flag for doctor review
- Use "INSUFFICIENT DATA" in missing_data when critical values are absent

Now analyze the record above and provide your assessment:"""
        
        return prompt
    
    def _parse_prediction(self, output: str) -> Dict[str, Any]:
        """Parse LLM prediction output"""
        
        try:
            # Clean output
            output = output.strip()
            
            # Remove markdown code blocks
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
                raise ValueError("Output is not a dictionary")
            
            # Ensure required fields
            result = {
                "risk_assessments": parsed.get("risk_assessments", []),
                "overall_risk": parsed.get("overall_risk", "UNKNOWN"),
                "data_quality": parsed.get("data_quality", "UNKNOWN"),
                "requires_doctor_review": False
            }
            
            # Check if any assessment requires doctor review
            for assessment in result["risk_assessments"]:
                if assessment.get("requires_doctor_review", False):
                    result["requires_doctor_review"] = True
                    break
                
                # Auto-flag HIGH/CRITICAL risks for review
                risk_level = assessment.get("risk_level", "").upper()
                if risk_level in ["HIGH", "CRITICAL"]:
                    assessment["requires_doctor_review"] = True
                    result["requires_doctor_review"] = True
                
                # Auto-flag low confidence for review
                confidence = assessment.get("confidence", 1.0)
                if confidence < 0.70:
                    assessment["requires_doctor_review"] = True
                    result["requires_doctor_review"] = True
            
            self.logger.info(f"Parsed {len(result['risk_assessments'])} risk assessments")
            self.logger.info(f"Overall risk: {result['overall_risk']}, Requires review: {result['requires_doctor_review']}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse prediction JSON: {str(e)}")
            self.logger.error(f"Raw output: {output[:500]}")
            return {
                "risk_assessments": [],
                "overall_risk": "UNKNOWN",
                "data_quality": "ERROR",
                "error": "Failed to parse LLM output",
                "requires_doctor_review": True
            }
        except Exception as e:
            self.logger.error(f"Error parsing prediction: {str(e)}")
            return {
                "risk_assessments": [],
                "overall_risk": "UNKNOWN",
                "data_quality": "ERROR",
                "error": str(e),
                "requires_doctor_review": True
            }

