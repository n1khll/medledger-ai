"""Run the Appointment Scheduling Agent"""

import sys

from pathlib import Path

import os

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

    print(f"API Documentation:        http://0.0.0.0:$PORT/docs")

    print(f"Start Job Endpoint:       http://0.0.0.0:$PORT/start_job")

    print(f"Status Check:             http://0.0.0.0:$PORT/status")

    print("=" * 70 + "\n")

    host = "0.0.0.0"

    port = int(os.environ.get("PORT", 8001))

    uvicorn.run(

        app,

        host=host,

        port=port,

        reload=False,

        log_level="info"

    )
