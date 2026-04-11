#!/bin/bash

# Start OPA in the background
echo "Starting OPA server..."
./opa run --server --log-level=debug ./src/policy/ &

# Wait a moment for OPA to initialize
sleep 2

# Start the FastAPI application
echo "Starting Sutradhara API..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
./.venv/bin/python3 -m uvicorn src.gateway.main:app --host 0.0.0.0 --port 8001
