#!/bin/bash

mkdir -p outs/preprocess/
sbatch --job-name pet -o outs/preprocess/pet.out -e outs/preprocess/pet.out ./slurms/preprocess/launch --parameters="configs/preprocess/processing.yaml" --pet

