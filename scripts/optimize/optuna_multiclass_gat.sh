#!/bin/bash

mkdir -p outs/optimize/
sbatch --job-name optuna_gat -o outs/optimize/optuna_gat.out -e outs/optimize/optuna_gat.out ./slurms/optimize/launch --parameters="configs/optimize/optuna_multiclass_gat.yaml" --cls="OptunaOptimizer"

