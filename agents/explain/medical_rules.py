"""Medical reference ranges and rule-based validators to prevent hallucinations"""
import re
from typing import Dict, Tuple, Optional, List, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

# Medical reference ranges (based on standard clinical guidelines)
MEDICAL_RANGES = {
    # Glucose (mg/dL)
    "glucose": {
        "normal": (70, 99),
        "pre_diabetic": (100, 125),
        "diabetic": (126, float('inf')),
        "unit": "mg/dL"
    },
    "glucose_fasting": {
        "normal": (70, 99),
        "pre_diabetic": (100, 125),
        "diabetic": (126, float('inf')),
        "unit": "mg/dL"
    },
    # HbA1c (%)
    "hba1c": {
        "normal": (0, 5.6),
        "pre_diabetic": (5.7, 6.4),
        "diabetic": (6.5, float('inf')),
        "unit": "%"
    },
    # Blood Pressure Systolic (mmHg)
    "bp_systolic": {
        "normal": (90, 119),
        "elevated": (120, 129),
        "stage1_hypertension": (130, 139),
        "stage2_hypertension": (140, float('inf')),
        "unit": "mmHg"
    },
    # Blood Pressure Diastolic (mmHg)
    "bp_diastolic": {
        "normal": (60, 79),
        "stage1_hypertension": (80, 89),
        "stage2_hypertension": (90, float('inf')),
        "unit": "mmHg"
    },
    # Total Cholesterol (mg/dL)
    "cholesterol_total": {
        "desirable": (0, 199),
        "borderline_high": (200, 239),
        "high": (240, float('inf')),
        "unit": "mg/dL"
    },
    # LDL Cholesterol (mg/dL)
    "ldl": {
        "optimal": (0, 99),
        "near_optimal": (100, 129),
        "borderline_high": (130, 159),
        "high": (160, 189),
        "very_high": (190, float('inf')),
        "unit": "mg/dL"
    },
    # HDL Cholesterol (mg/dL)
    "hdl": {
        "low_risk": (60, float('inf')),
        "moderate_risk": (40, 59),
        "high_risk": (0, 39),
        "unit": "mg/dL"
    },
    # Heart Rate (bpm)
    "heart_rate": {
        "normal": (60, 100),
        "bradycardia": (0, 59),
        "tachycardia": (101, float('inf')),
        "unit": "bpm"
    },
    # Body Temperature (°F)
    "temperature": {
        "normal": (97.0, 99.0),
        "fever": (99.1, float('inf')),
        "hypothermia": (0, 96.9),
        "unit": "°F"
    },
    # BMI
    "bmi": {
        "underweight": (0, 18.4),
        "normal": (18.5, 24.9),
        "overweight": (25.0, 29.9),
        "obese": (30.0, float('inf')),
        "unit": "kg/m²"
    }
}


class MedicalRuleValidator:
    """Rule-based validator to prevent LLM hallucinations"""
    
    def __init__(self):
        self.logger = logger
    
    def extract_value_from_text(self, text: str, value_name: str) -> Optional[float]:
        """
        Extract a medical value from text using pattern matching.
        
        Args:
            text: Text to search
            value_name: Type of value to extract (e.g., "glucose", "bp_systolic")
            
        Returns:
            Extracted numeric value or None
        """
        text_lower = text.lower()
        
        # Patterns for different value types
        patterns = {
            "glucose": [
                r'glucose[:\s]+(\d+\.?\d*)',
                r'fasting glucose[:\s]+(\d+\.?\d*)',
                r'blood glucose[:\s]+(\d+\.?\d*)',
                r'bg[:\s]+(\d+\.?\d*)'
            ],
            "glucose_fasting": [
                r'fasting glucose[:\s]+(\d+\.?\d*)',
                r'fasting bg[:\s]+(\d+\.?\d*)'
            ],
            "hba1c": [
                r'hba1c[:\s]+(\d+\.?\d*)',
                r'a1c[:\s]+(\d+\.?\d*)',
                r'hemoglobin a1c[:\s]+(\d+\.?\d*)'
            ],
            "bp_systolic": [
                r'bp[:\s]+(\d+)/\d+',
                r'blood pressure[:\s]+(\d+)/\d+',
                r'systolic[:\s]+(\d+)'
            ],
            "bp_diastolic": [
                r'bp[:\s]+\d+/(\d+)',
                r'blood pressure[:\s]+\d+/(\d+)',
                r'diastolic[:\s]+(\d+)'
            ],
            "cholesterol_total": [
                r'total cholesterol[:\s]+(\d+\.?\d*)',
                r'cholesterol[:\s]+(\d+\.?\d*)',
                r't\.?chol[:\s]+(\d+\.?\d*)'
            ],
            "ldl": [
                r'ldl[:\s]+(\d+\.?\d*)',
                r'ldl cholesterol[:\s]+(\d+\.?\d*)'
            ],
            "hdl": [
                r'hdl[:\s]+(\d+\.?\d*)',
                r'hdl cholesterol[:\s]+(\d+\.?\d*)'
            ],
            "heart_rate": [
                r'heart rate[:\s]+(\d+\.?\d*)',
                r'hr[:\s]+(\d+\.?\d*)',
                r'pulse[:\s]+(\d+\.?\d*)'
            ],
            "temperature": [
                r'temp[erature]*[:\s]+(\d+\.?\d*)',
                r't[:\s]+(\d+\.?\d*)\s*[°f]'
            ],
            "bmi": [
                r'bmi[:\s]+(\d+\.?\d*)',
                r'body mass index[:\s]+(\d+\.?\d*)'
            ]
        }
        
        if value_name not in patterns:
            return None
        
        for pattern in patterns[value_name]:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = float(match.group(1))
                    self.logger.debug(f"Extracted {value_name}: {value}")
                    return value
                except ValueError:
                    continue
        
        return None
    
    def categorize_value(self, value_name: str, value: float) -> Tuple[str, str]:
        """
        Categorize a medical value based on reference ranges.
        
        Args:
            value_name: Type of value (e.g., "glucose")
            value: Numeric value
            
        Returns:
            Tuple of (category, description)
        """
        if value_name not in MEDICAL_RANGES:
            return "unknown", "No reference range available"
        
        ranges = MEDICAL_RANGES[value_name]
        
        for category, (min_val, max_val) in ranges.items():
            if category == "unit":
                continue
            if min_val <= value <= max_val:
                return category, f"{category.replace('_', ' ').title()}"
        
        return "unknown", "Outside typical ranges"
    
    def validate_risk_prediction(self, record_text: str, risk_prediction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a risk prediction against the actual medical record.
        
        Args:
            record_text: Original medical record text
            risk_prediction: Risk prediction from LLM
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        is_valid = True
        
        try:
            disease = risk_prediction.get("disease", "Unknown")
            risk_score = risk_prediction.get("risk_score", 0)
            indicators = risk_prediction.get("indicators", [])
            evidence_quotes = risk_prediction.get("evidence_quotes", [])
            
            self.logger.info(f"Validating {disease} risk prediction")
            
            # Check 1: Verify evidence quotes exist in record
            for quote in evidence_quotes:
                if quote not in record_text:
                    issues.append(f"Evidence quote not found in record: '{quote}'")
                    is_valid = False
                    self.logger.warning(f"Evidence mismatch: '{quote}' not in record")
            
            # Check 2: For diabetes risk, verify glucose/HbA1c values
            if "diabetes" in disease.lower():
                glucose = self.extract_value_from_text(record_text, "glucose")
                hba1c = self.extract_value_from_text(record_text, "hba1c")
                
                if glucose is not None:
                    glucose_category, _ = self.categorize_value("glucose", glucose)
                    
                    # If glucose is normal but risk is HIGH, flag it
                    if glucose_category == "normal" and risk_score > 0.6:
                        issues.append(f"Diabetes risk score ({risk_score}) seems high for normal glucose ({glucose} mg/dL)")
                        is_valid = False
                        self.logger.warning(f"Risk-value mismatch: High diabetes risk with normal glucose")
                    
                    # If glucose is diabetic but risk is LOW, flag it
                    elif glucose_category == "diabetic" and risk_score < 0.4:
                        issues.append(f"Diabetes risk score ({risk_score}) seems low for diabetic glucose ({glucose} mg/dL)")
                        is_valid = False
                
                if hba1c is not None:
                    hba1c_category, _ = self.categorize_value("hba1c", hba1c)
                    
                    if hba1c_category == "normal" and risk_score > 0.6:
                        issues.append(f"Diabetes risk score ({risk_score}) seems high for normal HbA1c ({hba1c}%)")
                        is_valid = False
            
            # Check 3: For hypertension risk, verify BP values
            if "hypertension" in disease.lower():
                bp_sys = self.extract_value_from_text(record_text, "bp_systolic")
                bp_dia = self.extract_value_from_text(record_text, "bp_diastolic")
                
                if bp_sys is not None:
                    bp_category, _ = self.categorize_value("bp_systolic", bp_sys)
                    
                    if bp_category == "normal" and risk_score > 0.6:
                        issues.append(f"Hypertension risk score ({risk_score}) seems high for normal BP ({bp_sys}/{bp_dia or '?'} mmHg)")
                        is_valid = False
                    
                    elif bp_category in ["stage1_hypertension", "stage2_hypertension"] and risk_score < 0.4:
                        issues.append(f"Hypertension risk score ({risk_score}) seems low for elevated BP ({bp_sys}/{bp_dia or '?'} mmHg)")
                        is_valid = False
            
            # Check 4: Risk score should match risk level
            risk_level = risk_prediction.get("risk_level", "").upper()
            if risk_level == "LOW" and risk_score > 0.4:
                issues.append(f"Risk level 'LOW' doesn't match risk score {risk_score}")
                is_valid = False
            elif risk_level == "HIGH" and risk_score < 0.6:
                issues.append(f"Risk level 'HIGH' doesn't match risk score {risk_score}")
                is_valid = False
            elif risk_level == "CRITICAL" and risk_score < 0.8:
                issues.append(f"Risk level 'CRITICAL' doesn't match risk score {risk_score}")
                is_valid = False
            
            if is_valid:
                self.logger.info(f"Validation passed for {disease}")
            else:
                self.logger.warning(f"Validation failed for {disease}: {', '.join(issues)}")
            
            return is_valid, issues
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
    
    def extract_all_values(self, record_text: str) -> Dict[str, Optional[float]]:
        """
        Extract all medical values from record text.
        
        Args:
            record_text: Medical record text
            
        Returns:
            Dictionary of value_name -> value
        """
        values = {}
        
        for value_name in MEDICAL_RANGES.keys():
            value = self.extract_value_from_text(record_text, value_name)
            if value is not None:
                values[value_name] = value
                category, _ = self.categorize_value(value_name, value)
                self.logger.info(f"Found {value_name}: {value} ({category})")
        
        return values
    
    def get_abnormal_values(self, record_text: str) -> Dict[str, Dict]:
        """
        Get all abnormal values from the record.
        
        Args:
            record_text: Medical record text
            
        Returns:
            Dictionary of abnormal values with their categories
        """
        all_values = self.extract_all_values(record_text)
        abnormal = {}
        
        for value_name, value in all_values.items():
            category, description = self.categorize_value(value_name, value)
            
            # Check if abnormal (not in normal range)
            if category != "normal" and category != "unknown" and category != "desirable" and category != "optimal":
                abnormal[value_name] = {
                    "value": value,
                    "category": category,
                    "description": description,
                    "unit": MEDICAL_RANGES[value_name].get("unit", "")
                }
        
        return abnormal

