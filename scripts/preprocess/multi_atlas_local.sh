#!/bin/bash

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Starting Multi-Atlas Standardization (8 Atlases, 988 ROIs)"
echo " Reference: data/processed/AAL3v1_1mm_197x233x189.nii.gz"
echo "=========================================================="

python main.py preprocess \
    --parameters configs/preprocess/multi_atlas.yaml \
    --cls MultiAtlasProcessor

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Multi-atlas preprocessing failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

echo "=========================================================="
echo " Multi-Atlas Preprocessing Completed Successfully!"
echo "=========================================================="
