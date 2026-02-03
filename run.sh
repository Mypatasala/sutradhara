#!/bin/bash

# Configuration
PORT=8001
APP_MODULE="src.gateway.main:app"
VENV_PYTHON="./.venv/bin/python3"

echo "Checking for processes on port $PORT..."

# Find PID of process listening on the port
PID=$(lsof -ti:$PORT)

if [ -n "$PID" ]; then
    echo "Killing process $PID running on port $PORT..."
    kill -9 $PID
    sleep 1
else
    echo "No process found on port $PORT."
fi

echo "Starting application on port $PORT..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run uvicorn
$VENV_PYTHON -m uvicorn $APP_MODULE --host 0.0.0.0 --port $PORT --reload
