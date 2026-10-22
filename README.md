# multimodal-alzheimer-classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Multimodal Alzheimer's Disease and MCI Progression Classification using Heterogeneous Graph Neural Networks (`GCN`, `GAT`, `DualGraph`), Kolmogorov-Arnold Network (`KAN`) layers, and Foundational Model of Anatomy (`FMA`) Ontology Knowledge Graph constraints.

## Project Organization

```
├── LICENSE            <- MIT Open-source license
├── Makefile           <- Makefile with convenience commands (`make requirements`, `make test`, `make lint`)
├── README.md          <- Top-level README for developers using this project
├── main.py            <- Unified Click CLI entrypoint (`preprocess`, `experiment`, `optimize`, `explain`, `test`)
├── configs
│   ├── AAL3           <- AAL3v1 MNI brain parcellation atlas files (166 regions)
│   ├── FMA            <- Foundational Model of Anatomy (FMA) ontology artifacts
│   ├── preprocess     <- YAML configurations for neuroimaging & ontology preprocessing
│   ├── experiment     <- YAML configurations for single-model CV and multi-model benchmarks
│   ├── optimize       <- YAML configurations for Optuna TPE and Grid hyperparameter search
│   ├── explain        <- YAML configurations for Integrated Gradients and Attention explainability
│   └── test           <- YAML configurations for synthetic end-to-end smoke testing
│
├── data               <- Local data hierarchy (linked to `/media/BarraCuda/dcirilli2/`)
│   ├── external       <- Third-party atlases and ontology files (AAL3, FMA)
│   ├── interim        <- Intermediate N4 bias-corrected MRI, SPM segmentations, and co-registered PET
│   ├── processed      <- Extracted ROI feature tables and graph-ready datasets
│   └── raw            <- Original DICOM and raw NIfTI dumps
│
├── docs               <- MkDocs documentation project
├── logs               <- Pipeline execution and conversion tracking logs
├── models             <- Trained model checkpoints, fold metrics, and evaluation summaries
├── notebooks          <- Jupyter notebooks for interactive analysis
├── pyproject.toml     <- Project configuration and dependencies (`uv` / PEP 621)
├── references         <- Data dictionaries, manuals, and explanatory materials
├── reports            <- Generated analysis, tables, and explainability summaries
│   └── figures        <- Generated graphics and brain heatmaps
├── requirements.txt   <- Reproducible Python dependencies
├── scripts            <- Batch launcher shell scripts (`preprocess`, `experiment`, `optimize`, `explain`, `test`)
├── slurms             <- SLURM (`launch`, `exe`) scripts for cluster execution
└── src                <- Source code package
    ├── cli.py         <- Click CLI group and command definitions
    ├── data           <- PyTorch Geometric datasets (`AlzheimerGraphDataset`, `HybridAlzheimerDataset`, `FMAKnowledgeGraph`)
    ├── early_stop     <- `EarlyStopping` callback and checkpoint persistence
    ├── experiment     <- Experiment runners (`Run`, `GNNRun`, `AlzheimerGNNRun`, `AlzheimerBenchmarkRun`, `SmokeTestRun`)
    ├── explainability <- Integrated Gradients, GAT attention extraction, and fold-safe checkpoint reconstruction
    ├── losses         <- Custom loss functions (`FocalLoss`, `CrossEntropyLoss`)
    ├── metrics        <- Binary and multiclass classification metric accumulators
    ├── models         <- GNN+KAN architectures (`GCN_KAN_Classifier`, `GAT_KAN_Classifier`, `DualGraph_KAN_Classifier`)
    ├── optimization   <- Hyperparameter tuning (`OptunaOptimizer`, `GridOptimizer`)
    ├── preprocess     <- Neuroimaging (N4, SPM25, FSL FLIRT, SUVR) & FMA ontology (`ont/`) processors
    ├── scheduler      <- Learning rate schedulers (`HFCosineScheduler`, `CosineAnnealingWarmRestarts`)
    ├── utils          <- Configuration loading (`ruamel.yaml`), `ParameterKeys`, seeding, and path resolution
    └── visualization  <- Brain connectome, 3D glass brain heatmaps, and FMA explanation figures
```

--------

## Data Location

Preprocessed neuroimaging data is stored under `/media/BarraCuda/dcirilli2/` (automatically linked under `data/`):
- `/media/BarraCuda/dcirilli2/nifti`: Converted sMRI (`MPRAGE`) and FDG-PET NIfTI volumes
- `/media/BarraCuda/dcirilli2/n4`: ANTs N4 bias-field corrected sMRI volumes
- `/media/BarraCuda/dcirilli2/segmentation`: SPM25 tissue probability maps (`wc1`, `wc2`, `wc3`) in MNI space
- `/media/BarraCuda/dcirilli2/pet_coregistered`: FSL FLIRT 6-DOF PET-to-sMRI co-registered volumes
- `/media/BarraCuda/dcirilli2/swin_npy`: Preprocessed 3D NumPy volumes and manifest

## CLI Usage

### 1. Preprocessing (`preprocess`)
```bash
# Extract AAL3 166-region MNI centroids
python main.py preprocess --parameters configs/preprocess/extract_coords.yaml --coords

# Build or validate the compact FMA Knowledge Graph from fma.owl
python main.py preprocess --parameters configs/preprocess/ontology_mapping.yaml --ontology
python main.py preprocess --parameters configs/preprocess/ontology_kg.yaml --ontology
python main.py preprocess --parameters configs/preprocess/ontology_validate.yaml --ontology

# Run neuroimaging preprocessing stages (DICOM -> NIfTI -> sMRI N4+SPM -> PET FLIRT -> ROI Features)
python main.py preprocess --parameters configs/preprocess/processing.yaml --dicom
python main.py preprocess --parameters configs/preprocess/processing.yaml --smri
python main.py preprocess --parameters configs/preprocess/processing.yaml --pet
python main.py preprocess --parameters configs/preprocess/processing.yaml --features

# Create clinical and diagnostic cohorts (MCI conversion, CN vs AD, Multiclass CN-MCI-AD)
python main.py preprocess --parameters configs/preprocess/dataset_clinical.yaml --clinical
python main.py preprocess --parameters configs/preprocess/dataset_cn_ad.yaml --cn_ad
python main.py preprocess --parameters configs/preprocess/dataset_multiclass.yaml --multiclass
```

### 2. Training & Benchmarking (`experiment`)
```bash
# Train 5-fold StratifiedGroupKFold GAT+KAN with heterogeneous multigraph + FMA ontology
python main.py experiment --parameters configs/experiment/multiclass/gat_multigraph.yaml --cls AlzheimerGNNRun

# Run full multi-model (XGBoost, GCN, GAT, DualGraph) x multi-topology benchmark
python main.py experiment --parameters configs/experiment/benchmark/multiclass_benchmark.yaml --cls AlzheimerBenchmarkRun
```

### 3. Hyperparameter Optimization (`optimize`)
```bash
python main.py optimize --parameters configs/optimize/optuna_multiclass_gat.yaml --cls OptunaOptimizer
```

### 4. Explainability (`explain`)
```bash
# Single-patient Integrated Gradients + GAT Attention extraction
python main.py explain --parameters configs/explain/explain_single.yaml --cls ExplainerRun

# Cohort-level batch Integrated Gradients across saved fold checkpoints
python main.py explain --parameters configs/explain/explain_batch.yaml --cls BatchExplainerRun
```

### 5. Verification & Smoke Testing (`test`)
```bash
python main.py test --parameters configs/test/smoke_test.yaml --cls SmokeTestRun
```
