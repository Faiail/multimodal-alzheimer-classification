#!/bin/bash

mkdir -p outs/experiment/vit
sbatch --array=0-4 --job-name vit -o outs/experiment/vit/cn_ad-%a.out -e outs/experiment/vit/cn_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/vit/cn_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name vit -o outs/experiment/vit/cn_mci_ad-%a.out -e outs/experiment/vit/cn_mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/vit/cn_mci_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name vit -o outs/experiment/vit/cn_mci-%a.out -e outs/experiment/vit/cn_mci-%a.out ./slurms/launch_experiment --parameters="configs/experiment/vit/cn_mci.yaml" --cls VisualRun

sbatch --array=0-4 --job-name vit -o outs/experiment/vit/mci_ad-%a.out -e outs/experiment/vit/mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/vit/mci_ad.yaml" --cls VisualRun
