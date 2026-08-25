#!/bin/bash

# Start OPA in the background
echo "Starting OPA server..."
# EXTRA_POLICY_PATH: optional extra bundle root(s) for a tenant's own
# policy folder (e.g. myPatasala's policy/opa/), loaded alongside
# ./src/policy/ without sutradhara needing to know that tenant by name.
./opa run --server --log-level=debug ./src/policy/ ${EXTRA_POLICY_PATH:-} &

# Wait a moment for OPA to initialize
sleep 2

# Start the FastAPI application
echo "Starting Sutradhara API..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
./.venv/bin/python3 -m uvicorn src.gateway.main:app --host 0.0.0.0 --port 8001
