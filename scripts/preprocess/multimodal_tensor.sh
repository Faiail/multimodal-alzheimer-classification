#!/bin/bash

mkdir -p outs/preprocess

sbatch --job-name multimodal_tensor -o outs/preprocess/multimodal_tensor.out -e outs/preprocess/multimodal_tensor.out slurms/launch_cpu preprocess --parameters configs/preprocess/multimodal_tensor.yaml --cls MultimodalTensorProcessor
