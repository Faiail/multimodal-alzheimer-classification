#!/bin/bash

mkdir -p outs/preprocess

# Check if Slurm (sbatch) is available in current environment
if command -v sbatch >/dev/null 2>&1; then
    sbatch --job-name multi_atlas_tensor \
           -o outs/preprocess/multi_atlas_tensor.out \
           -e outs/preprocess/multi_atlas_tensor.out \
           slurms/launch_cpu preprocess --parameters configs/preprocess/multi_atlas_tensor.yaml --cls MultiAtlasTensorProcessor
else
    echo "Slurm 'sbatch' not found. Executing locally via scripts/preprocess/multi_atlas_tensor_local.sh..."
    exec ./scripts/preprocess/multi_atlas_tensor_local.sh "$@"
fi
