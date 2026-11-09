#!/bin/bash

echo "CN vs AD"
uv run main.py preprocess --parameters configs/preprocess/splits/cn_ad.yaml --cls SplitProcessor
echo "CN vs MCI vs AD"
uv run main.py preprocess --parameters configs/preprocess/splits/cn_mci_ad.yaml --cls SplitProcessor
echo "CN vs MCI"
uv run main.py preprocess --parameters configs/preprocess/splits/cn_mci.yaml --cls SplitProcessor
echo "MCI vs AD"
uv run main.py preprocess --parameters configs/preprocess/splits/mci_ad.yaml --cls SplitProcessor
