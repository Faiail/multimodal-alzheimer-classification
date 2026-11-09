#!/bin/bash

mkdir -p outs/experiment/resnet50
sbatch --array=0-4 --job-name r50 -o outs/experiment/resnet50/cn_ad-%a.out -e outs/experiment/resnet50/cn_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet50/cn_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name r50 -o outs/experiment/resnet50/cn_mci_ad-%a.out -e outs/experiment/resnet50/cn_mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet50/cn_mci_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name r50 -o outs/experiment/resnet50/cn_mci-%a.out -e outs/experiment/resnet50/cn_mci-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet50/cn_mci.yaml" --cls VisualRun

sbatch --array=0-4 --job-name r50 -o outs/experiment/resnet50/mci_ad-%a.out -e outs/experiment/resnet50/mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/resnet50/mci_ad.yaml" --cls VisualRun
