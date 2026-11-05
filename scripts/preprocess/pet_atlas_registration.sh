#!/bin/bash

mkdir -p outs/preprocess

sbatch --job-name pet_atlas -o outs/preprocess/pet_atlas.out -e outs/preprocess/pet_atlas.out slurms/launch_cpu preprocess --parameters configs/preprocess/pet_atlas_registration.yaml --cls PETAtlasRegistrationProcessor


