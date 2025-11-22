#!/bin/bash

# SecondEye Agents Server Startup Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include the agents directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one from .env.example"
    echo "Copy .env.example to .env and add your OpenAI API key"
    exit 1
fi

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the server
echo "Starting SecondEye Agents API server..."
echo "PYTHONPATH: $PYTHONPATH"
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
