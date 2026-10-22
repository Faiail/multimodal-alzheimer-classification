#!/bin/bash

mkdir -p outs/experiment/benchmark/
sbatch --job-name multiclass_bench -o outs/experiment/benchmark/multiclass_bench.out -e outs/experiment/benchmark/multiclass_bench.out ./slurms/experiment/launch --parameters="configs/experiment/benchmark/multiclass_benchmark.yaml" --cls="AlzheimerBenchmarkRun"

