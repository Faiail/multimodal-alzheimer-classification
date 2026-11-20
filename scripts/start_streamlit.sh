#!/bin/bash
# ==============================================================================
# Alzheimer Multimodal Streamlit WebApp Launcher
# Launches Streamlit on http://localhost:8501
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"

echo "=================================================================="
echo " Starting Alzheimer Multimodal Streamlit WebApp"
echo " Project root: ${PROJECT_DIR}"
echo " Serving on:   http://localhost:${PORT}"
echo "=================================================================="

# Check Python and Streamlit environment
if [ -f ".venv/bin/streamlit" ]; then
    STREAMLIT_BIN=".venv/bin/streamlit"
elif command -v streamlit &>/dev/null; then
    STREAMLIT_BIN="streamlit"
else
    echo "Error: Streamlit binary not found in .venv."
    exit 1
fi

echo "Using Streamlit: $(${STREAMLIT_BIN} --version) (${STREAMLIT_BIN})"

# Start Streamlit application
exec "${STREAMLIT_BIN}" run webapp/streamlit_app.py \
    --server.port "${PORT}" \
    --server.address "${HOST}" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base "dark" \
    --theme.primaryColor "#3b82f6" \
    --theme.backgroundColor "#0b0f19" \
    --theme.secondaryBackgroundColor "#111827" \
    --theme.textColor "#f3f4f6" \
    "$@"
