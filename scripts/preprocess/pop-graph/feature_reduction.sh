#!/bin/bash

mkdir -p outs/preprocess/population_feature

TASKS=("cn_ad" "cn_mci" "cn_mci_ad" "mci_ad")

# If a specific task is provided as argument, run only that task
if [ -n "$1" ]; then
    TASKS=("$1")
fi

SPECIFIC_FOLD="$2"

for TASK in "${TASKS[@]}"; do
    echo "Submitting feature reduction job for task: ${TASK}..."
    sbatch --job-name "feat_${TASK}" \
           -o "outs/preprocess/population_feature/${TASK}.out" \
           -e "outs/preprocess/population_feature/${TASK}.out" \
           slurms/launch_pop_feature "${TASK}" "${SPECIFIC_FOLD}"
done
