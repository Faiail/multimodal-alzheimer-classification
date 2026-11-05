#!/bin/bash

mkdir -p outs/preprocess

sbatch --job-name mrireg -o outs/preprocess/mri_atlas_registration.out -e outs/preprocess/mri_atlas_registration.out slurms/launch_cpu preprocess --parameters configs/preprocess/mri_template_registration.yaml --cls MRIAtlasRegistrationProcessor


