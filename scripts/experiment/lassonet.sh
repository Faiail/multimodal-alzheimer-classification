#!/bin/bash

mkdir -p outs/experiment/lassonet

TASKS=("cn_ad" "cn_mci" "mci_ad" "cn_mci_ad")

if [ -n "$1" ]; then
    TASKS=("$1")
fi

for TASK in "${TASKS[@]}"; do
    echo "Submitting lassonet experiment for task: ${TASK} (folds 0-4)..."
    sbatch --array=0-4 \
           --job-name "lassonet" \
           -o "outs/experiment/lassonet/${TASK}-%a.out" \
           -e "outs/experiment/lassonet/${TASK}-%a.out" \
           ./slurms/launch_experiment --parameters="configs/experiment/lassonet/${TASK}.yaml" --cls LassoNetRun
done
