#!/bin/bash

TASKS=("cn_ad" "cn_mci" "mci_ad" "cn_mci_ad")
FOLDS=(0 1 2 3 4)

# Filter by task if provided as first argument
if [ -n "$1" ]; then
    TASKS=("$1")
fi

# Filter by fold if provided as second argument
if [ -n "$2" ]; then
    FOLDS=("$2")
fi

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Starting Local Execution for BrainGNN"
echo " Tasks: ${TASKS[*]}"
echo " Folds: ${FOLDS[*]}"
echo "=========================================================="

for TASK in "${TASKS[@]}"; do
    CONFIG="configs/experiment/brain_gnn/${TASK}.yaml"
    
    if [ ! -f "$CONFIG" ]; then
        echo "Warning: Configuration file ${CONFIG} not found. Skipping."
        continue
    fi

    for FOLD in "${FOLDS[@]}"; do
        echo ""
        echo ">>> Running Task: ${TASK} | Fold: ${FOLD} <<<"

        python main.py experiment \
            --parameters "$CONFIG" \
            --cls BrainGNNRun \
            --fold "$FOLD"

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Error: Execution failed on ${TASK} fold ${FOLD} with exit code ${EXIT_CODE}"
            exit $EXIT_CODE
        fi
    done
done

echo ""
echo "=========================================================="
echo " All BrainGNN local runs completed successfully!"
echo "=========================================================="
