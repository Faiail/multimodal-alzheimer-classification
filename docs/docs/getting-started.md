Getting started
===============

1. Install dependencies using `uv sync` or `make requirements`.
2. Ensure the ADNI preprocessed imaging data is mounted under `/media/BarraCuda/dcirilli2/`.
3. Run preprocessing pipelines via `python main.py preprocess --parameters configs/preprocess/<step>.yaml --<flag>`.
4. Launch experiments via `python main.py experiment --parameters configs/experiment/<task>/<model>.yaml --cls AlzheimerGNNRun`.
5. Run hyperparameter optimization via `python main.py optimize --parameters configs/optimize/<config>.yaml --cls OptunaOptimizer`.
6. Generate Integrated Gradients and Attention explanations via `python main.py explain --parameters configs/explain/<config>.yaml --cls ExplainerRun`.

