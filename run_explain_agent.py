"""Run the Explain Agent (Report Summary Agent)"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agents" / "explain"))

if __name__ == "__main__":
    # Import app directly
    from agents.explain.main import app
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting Explain Agent (Report Summary) API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://127.0.0.1:8000/docs")
    print(f"PDF Upload Endpoint:      http://127.0.0.1:8000/upload_pdf")
    print(f"Start Job Endpoint:       http://127.0.0.1:8000/start_job")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

