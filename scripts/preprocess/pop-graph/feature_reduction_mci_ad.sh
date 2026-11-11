#!/bin/bash

mkdir -p outs/preprocess/population_feature

TASK="mci_ad"
SPECIFIC_FOLD="$1"

echo "Submitting feature reduction job for task: ${TASK}..."
sbatch --job-name "feat_${TASK}" \
       -o "outs/preprocess/population_feature/${TASK}.out" \
       -e "outs/preprocess/population_feature/${TASK}.out" \
       slurms/launch_pop_feature "${TASK}" "${SPECIFIC_FOLD}"
