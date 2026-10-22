#!/bin/bash

mkdir -p outs/preprocess/
sbatch --job-name smri -o outs/preprocess/smri.out -e outs/preprocess/smri.out ./slurms/preprocess/launch --parameters="configs/preprocess/processing.yaml" --smri

