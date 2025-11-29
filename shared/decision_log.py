"""Decision log module for tracking agent decisions and actions"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List
from shared.logging_config import get_logger

logger = get_logger(__name__)


class DecisionLog:
    """
    Decision log for tracking agent actions and decisions.
    
    Each entry includes:
    - job_id: Unique job identifier
    - timestamp: Unix timestamp
    - actor: Who performed the action (agent, system, user)
    - action: What action was performed
    - input_hash: Hash of the input data
    - output_hash: Hash of the output data (if applicable)
    - result_summary: Human-readable summary of the result
    """
    
    def __init__(self, log_file: str = "logs/decision_log.jsonl"):
        """
        Initialize the decision log.
        
        Args:
            log_file: Path to the log file (JSONL format)
        """
        self.log_file = log_file
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger.info(f"Decision log initialized: {log_file}")
    
    def add_entry(
        self,
        job_id: str,
        actor: str,
        action: str,
        input_hash: str,
        output_hash: str = None,
        result_summary: str = None,
        additional_data: Dict[str, Any] = None
    ) -> None:
        """
        Add an entry to the decision log.
        
        Args:
            job_id: Unique job identifier
            actor: Who performed the action (agent, system, user)
            action: What action was performed
            input_hash: Hash of the input data
            output_hash: Hash of the output data (optional)
            result_summary: Human-readable summary of the result (optional)
            additional_data: Additional data to include in the log entry
        """
        entry = {
            "job_id": job_id,
            "timestamp": int(datetime.utcnow().timestamp()),
            "actor": actor,
            "action": action,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "result_summary": result_summary
        }
        
        # Add any additional data
        if additional_data:
            entry.update(additional_data)
        
        # Write to log file (JSONL format - one JSON object per line)
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            logger.info(f"Decision log entry added: job_id={job_id}, action={action}")
        except Exception as e:
            logger.error(f"Failed to write to decision log: {str(e)}")
    
    def get_entries(self, job_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve entries from the decision log.
        
        Args:
            job_id: Filter by job_id (optional)
            
        Returns:
            List of log entries
        """
        entries = []
        
        if not os.path.exists(self.log_file):
            return entries
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if job_id is None or entry.get("job_id") == job_id:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in decision log: {line}")
        except Exception as e:
            logger.error(f"Failed to read decision log: {str(e)}")
        
        return entries

