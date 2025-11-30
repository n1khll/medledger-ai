"""Run the Medical Q&A Agent"""

import sys

from pathlib import Path

import os

# Add project root to path

project_root = Path(__file__).parent

sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "agents" / "medical_qa"))

if __name__ == "__main__":

    # Import app directly

    from agents.medical_qa.main import app

    import uvicorn

    host = "0.0.0.0"  # Bind to all interfaces (needed for Render/Docker)
    port = int(os.environ.get("PORT", 8002))
    
    # Display URL uses localhost for browser access
    display_host = "localhost"

    print("\n" + "=" * 70)
    print("🚀 Starting Medical Q&A Agent API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://{display_host}:{port}/docs")
    print(f"Ask Question Endpoint:     http://{display_host}:{port}/ask")
    print(f"Upload Document:           http://{display_host}:{port}/upload_document")
    print(f"Health Check:              http://{display_host}:{port}/health")
    print("=" * 70 + "\n")

    uvicorn.run(

        app,

        host=host,

        port=port,

        reload=False,

        log_level="info"

    )


import sys

from pathlib import Path

import os

# Add project root to path

project_root = Path(__file__).parent

sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "agents" / "medical_qa"))

if __name__ == "__main__":

    # Import app directly

    from agents.medical_qa.main import app

    import uvicorn

    host = "0.0.0.0"  # Bind to all interfaces (needed for Render/Docker)
    port = int(os.environ.get("PORT", 8002))
    
    # Display URL uses localhost for browser access
    display_host = "localhost"

    print("\n" + "=" * 70)
    print("🚀 Starting Medical Q&A Agent API Server...")
    print("=" * 70)
    print(f"API Documentation:        http://{display_host}:{port}/docs")
    print(f"Ask Question Endpoint:     http://{display_host}:{port}/ask")
    print(f"Upload Document:           http://{display_host}:{port}/upload_document")
    print(f"Health Check:              http://{display_host}:{port}/health")
    print("=" * 70 + "\n")

    uvicorn.run(

        app,

        host=host,

        port=port,

        reload=False,

        log_level="info"

    )

