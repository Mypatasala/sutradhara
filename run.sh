#!/bin/bash

PORT=8001
APP_MODULE="src.gateway.main:app"
VENV_PYTHON="./.venv/bin/python3"
OPA_PORT=8181

echo "Checking for processes on port $PORT..."
PID=$(lsof -ti:$PORT)
if [ -n "$PID" ]; then
    echo "Killing process $PID running on port $PORT..."
    kill -9 "$PID"
    sleep 1
else
    echo "No process found on port $PORT."
fi

# Start Ollama if not already running
if ! pgrep -x ollama > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
else
    echo "Ollama already running."
fi

# Start OPA only if it isn't already listening
if ! lsof -ti:$OPA_PORT > /dev/null 2>&1; then
    echo "Starting OPA policy server on port $OPA_PORT..."
    # EXTRA_POLICY_PATH: optional extra bundle root(s) for a tenant's own
    # policy folder (e.g. myPatasala's policy/opa/), loaded alongside
    # ./src/policy/ without sutradhara needing to know that tenant by name.
    ./opa run --server --log-level=error --ignore='*_test.rego' ./src/policy/ ${EXTRA_POLICY_PATH:-} &
    OPA_PID=$!
    sleep 2
    echo "OPA started (PID: $OPA_PID)"
else
    echo "OPA already running on port $OPA_PORT."
fi

echo "Starting application on port $PORT..."
export PYTHONPATH="$PYTHONPATH:$(pwd)"
$VENV_PYTHON -m uvicorn $APP_MODULE --host 0.0.0.0 --port $PORT --reload
