# medledger-ai

## Start commands (for Render or local)

Explain agent:

uvicorn agents.explain.main:app --host 0.0.0.0 --port $PORT

Appointment agent:

uvicorn agents.appointment.appointment_main:app --host 0.0.0.0 --port $PORT