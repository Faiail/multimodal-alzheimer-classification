#!/bin/bash

mkdir -p outs/preprocess

sbatch --job-name coreg -o outs/preprocess/co_registration.out -e outs/preprocess/co_registration.out slurms/launch_cpu preprocess --parameters configs/preprocess/pet_coregistration.yaml --cls CoregistrationProcessor
