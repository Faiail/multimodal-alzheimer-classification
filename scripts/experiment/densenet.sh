#!/bin/bash

mkdir -p outs/experiment/densenet
sbatch --array=0-4 --job-name dn121 -o outs/experiment/densenet/cn_ad-%a.out -e outs/experiment/densenet/cn_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/densenet/cn_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name dn121 -o outs/experiment/densenet/cn_mci_ad-%a.out -e outs/experiment/densenet/cn_mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/densenet/cn_mci_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name dn121 -o outs/experiment/densenet/cn_mci-%a.out -e outs/experiment/densenet/cn_mci-%a.out ./slurms/launch_experiment --parameters="configs/experiment/densenet/cn_mci.yaml" --cls VisualRun

sbatch --array=0-4 --job-name dn121 -o outs/experiment/densenet/mci_ad-%a.out -e outs/experiment/densenet/mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/densenet/mci_ad.yaml" --cls VisualRun
