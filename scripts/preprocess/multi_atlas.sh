#!/bin/bash

mkdir -p outs/preprocess

# Check if Slurm (sbatch) is available in the current environment
if command -v sbatch >/dev/null 2>&1; then
    sbatch --job-name multi_atlas \
           -o outs/preprocess/multi_atlas.out \
           -e outs/preprocess/multi_atlas.out \
           slurms/launch_cpu preprocess --parameters configs/preprocess/multi_atlas.yaml --cls MultiAtlasProcessor
else
    echo "Slurm 'sbatch' not found. Executing locally via scripts/preprocess/multi_atlas_local.sh..."
    exec ./scripts/preprocess/multi_atlas_local.sh "$@"
fi
