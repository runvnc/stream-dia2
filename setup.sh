#!/bin/bash
# Setup for stream-dia2
# Initializes git submodule and runs uv sync

set -e

echo "=== stream-dia2 Setup ==="

# Initialize submodule if not already done
if [ ! -f "dia2_repo/pyproject.toml" ]; then
    echo "Initializing dia2 submodule..."
    git submodule update --init --recursive
fi

# Sync dependencies
echo "Running uv sync..."
uv sync

echo ""
echo "=== Setup complete! ==="
echo "Run the server with: uv run uvicorn realtime_dia2_server:app --host 0.0.0.0 --port 8000"
