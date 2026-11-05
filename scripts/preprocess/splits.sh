#!/bin/bash

uv run main.py preprocess --parameters configs/preprocess/splits/cn_ad.yaml --cls SplitProcessor
uv run main.py preprocess --parameters configs/preprocess/splits/cn_mci_ad.yaml --cls SplitProcessor
uv run main.py preprocess --parameters configs/preprocess/splits/cn_mci.yaml --cls SplitProcessor
uv run main.py preprocess --parameters configs/preprocess/splits/mci_ad.yaml --cls SplitProcessor
