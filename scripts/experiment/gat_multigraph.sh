#!/bin/bash

mkdir -p outs/experiment/multiclass/
sbatch --job-name gat_multigraph -o outs/experiment/multiclass/gat_multigraph.out -e outs/experiment/multiclass/gat_multigraph.out ./slurms/experiment/launch --parameters="configs/experiment/multiclass/gat_multigraph.yaml" --cls="AlzheimerGNNRun"

