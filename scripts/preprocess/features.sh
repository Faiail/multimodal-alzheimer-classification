#!/bin/bash

mkdir -p outs/preprocess/
sbatch --job-name features -o outs/preprocess/features.out -e outs/preprocess/features.out ./slurms/preprocess/launch --parameters="configs/preprocess/processing.yaml" --features

