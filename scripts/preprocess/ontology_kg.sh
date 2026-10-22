#!/bin/bash

mkdir -p outs/preprocess/
sbatch --job-name ontology -o outs/preprocess/ontology.out -e outs/preprocess/ontology.out ./slurms/preprocess/launch --parameters="configs/preprocess/ontology_kg.yaml" --ontology

