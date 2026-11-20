#!/bin/bash
# ==============================================================================
# Alzheimer Multimodal WebApp Launcher
# Launches the Python backend and web viewer on http://localhost:8050
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

PORT="${PORT:-8050}"
HOST="${HOST:-0.0.0.0}"

echo "=================================================================="
echo " Starting Alzheimer Multimodal WebApp"
echo " Project root: ${PROJECT_DIR}"
echo " Serving on:   http://localhost:${PORT}"
echo "=================================================================="

# Check Python environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 not found."
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} --version) (${PYTHON_BIN})"

# Start server
exec "${PYTHON_BIN}" webapp/server.py --host "${HOST}" --port "${PORT}" "$@"
