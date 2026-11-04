#!/bin/bash

mkdir -p outs/explain/
sbatch --job-name explain_single -o outs/explain/explain_single.out -e outs/explain/explain_single.out ./slurms/explain/launch --parameters="configs/explain/explain_single.yaml" --cls="ExplainerRun"

