#!/bin/bash

mkdir -p outs/preprocess

sbatch --job-name brain_graphs \
       -o outs/preprocess/brain_graphs.out \
       -e outs/preprocess/brain_graphs.out \
       slurms/launch_cpu preprocess --parameters configs/preprocess/brain_graphs.yaml --cls BrainGNNProcessor
