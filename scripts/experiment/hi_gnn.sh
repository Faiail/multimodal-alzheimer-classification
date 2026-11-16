#!/bin/bash

mkdir -p outs/experiment/hi_gnn

TASKS=("cn_ad" "cn_mci" "mci_ad" "cn_mci_ad")

if [ -n "$1" ]; then
    TASKS=("$1")
fi

for TASK in "${TASKS[@]}"; do
    echo "Submitting hi_gnn experiment for task: ${TASK} (folds 0-4)..."
    sbatch --array=0-4 \
           --job-name "hi_gnn" \
           -o "outs/experiment/hi_gnn/${TASK}-%a.out" \
           -e "outs/experiment/hi_gnn/${TASK}-%a.out" \
           ./slurms/launch_experiment --parameters="configs/experiment/hi_gnn/${TASK}.yaml" --cls HiGNNRun
done
