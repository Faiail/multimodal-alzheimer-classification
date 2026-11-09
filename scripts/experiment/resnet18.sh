#!/bin/bash

mkdir -p outs/experiment/resnet18
sbatch --array=0-4 --job-name r18 -o outs/experiment/resnet18/cn_ad-%a.out -e outs/experiment/resnet18/cn_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet18/cn_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name r18 -o outs/experiment/resnet18/cn_mci_ad-%a.out -e outs/experiment/resnet18/cn_mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet18/cn_mci_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name r18 -o outs/experiment/resnet18/cn_mci-%a.out -e outs/experiment/resnet18/cn_mci-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet18/cn_mci.yaml" --cls VisualRun

sbatch --array=0-4 --job-name r18 -o outs/experiment/resnet18/mci_ad-%a.out -e outs/experiment/resnet18/mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet18/mci_ad.yaml" --cls VisualRun

