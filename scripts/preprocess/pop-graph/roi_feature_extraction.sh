#!/bin/bash

mkdir -p outs/preprocess

sbatch --job-name roi -o outs/preprocess/roi_feature_extraction.out -e outs/preprocess/roi_feature_extraction.out slurms/launch_cpu preprocess --parameters configs/preprocess/roi_feature.yaml --cls ROIFeatureProcessor


