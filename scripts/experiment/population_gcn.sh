#!/bin/bash

mkdir -p outs/experiment/population_gcn

TASKS=("cn_ad" "cn_mci" "mci_ad" "cn_mci_ad")

if [ -n "$1" ]; then
    TASKS=("$1")
fi

for TASK in "${TASKS[@]}"; do
    echo "Submitting population_gcn experiment for task: ${TASK} (folds 0-4)..."
    sbatch --array=0-4 \
           --job-name "pop_gcn" \
           -o "outs/experiment/population_gcn/${TASK}-%a.out" \
           -e "outs/experiment/population_gcn/${TASK}-%a.out" \
           ./slurms/launch_experiment --parameters="configs/experiment/population_gcn/${TASK}.yaml" --cls PopulationRun
done
