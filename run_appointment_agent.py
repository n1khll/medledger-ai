"""Run the Appointment Scheduling Agent"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agents" / "appointment"))

if __name__ == "__main__":
    # Import app directly
    from agents.appointment.appointment_main import app
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting Appointment Scheduling Agent API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://127.0.0.1:8001/docs")
    print(f"Start Job Endpoint:       http://127.0.0.1:8001/start_job")
    print(f"Status Check:             http://127.0.0.1:8001/status")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info"
    )

