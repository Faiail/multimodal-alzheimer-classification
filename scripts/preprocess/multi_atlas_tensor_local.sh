#!/bin/bash

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Starting Multi-Atlas Tensor Extraction (988 ROIs)"
echo " SafeTensors: data/processed/multi_atlas_safetensors"
echo "=========================================================="

python main.py preprocess \
    --parameters configs/preprocess/multi_atlas_tensor.yaml \
    --cls MultiAtlasTensorProcessor

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Multi-atlas tensor extraction failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

echo "=========================================================="
echo " Multi-Atlas Tensor Extraction Completed Successfully!"
echo "=========================================================="
