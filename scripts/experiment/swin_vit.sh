#!/bin/bash

mkdir -p outs/experiment/swin_vit
sbatch --array=0-4 --job-name swin -o outs/experiment/swin_vit/cn_ad-%a.out -e outs/experiment/swin_vit/cn_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/swin_vit/cn_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name swin -o outs/experiment/swin_vit/cn_mci_ad-%a.out -e outs/experiment/swin_vit/cn_mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/swin_vit/cn_mci_ad.yaml" --cls VisualRun

sbatch --array=0-4 --job-name swin -o outs/experiment/swin_vit/cn_mci-%a.out -e outs/experiment/swin_vit/cn_mci-%a.out ./slurms/launch_experiment --parameters="configs/experiment/swin_vit/cn_mci.yaml" --cls VisualRun

sbatch --array=0-4 --job-name swin -o outs/experiment/swin_vit/mci_ad-%a.out -e outs/experiment/swin_vit/mci_ad-%a.out ./slurms/launch_experiment --parameters="configs/experiment/swin_vit/mci_ad.yaml" --cls VisualRun
