#!/bin/bash
# ==============================================================================
# Correct all_features_multimodal.csv using ADNIMERGE dynamic labels
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Running dataset correction with ${PYTHON_BIN}..."
"${PYTHON_BIN}" scripts/preprocess/correct_all_features_multimodal.py "$@"
