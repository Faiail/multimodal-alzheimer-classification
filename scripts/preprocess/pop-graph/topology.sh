#!/bin/bash

mkdir -p outs/preprocess

TASKS=("cn_ad" "cn_mci" "cn_mci_ad" "mci_ad")

# If a specific task is provided as argument, run only that task
if [ -n "$1" ]; then
    TASKS=("$1")
fi

for TASK in "${TASKS[@]}"; do
    echo "Submitting population topology for task: ${TASK}"
    sbatch --job-name "top_${TASK}" \
           -o "outs/preprocess/topology_${TASK}.out" \
           -e "outs/preprocess/topology_${TASK}.out" \
           slurms/launch_cpu preprocess --parameters "configs/preprocess/population_topology/${TASK}.yaml" --cls PopulationTopologyProcessor
done


