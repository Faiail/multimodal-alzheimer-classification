#!/bin/bash

mkdir -p outs/test/
sbatch --job-name smoke_test -o outs/test/smoke_test.out -e outs/test/smoke_test.out ./slurms/test/launch --parameters="configs/test/smoke_test.yaml" --cls="SmokeTestRun"

