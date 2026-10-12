"""Rebuild a model and its held-out fold from a saved training checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.data import Batch

from data.batch_utils import fma_dataset_kwargs
from data.dataset import AlzheimerGraphDataset, HybridAlzheimerDataset
from models.model import (
    DualGraph_KAN_Classifier,
    GAT_KAN_Classifier,
    GCN_KAN_Classifier,
)


def load_checkpoint(path: str | Path, device: torch.device):
    return torch.load(Path(path), map_location=device, weights_only=False)


def apply_normalizer_state(
    frame: pd.DataFrame, state: dict | None
) -> pd.DataFrame:
    """Apply the imputation and scaling values stored in a checkpoint."""
    transformed = frame.copy()
    if state is None:
        return transformed
    for group_name in ('imaging', 'clinical'):
        group = state.get(group_name)
        if not group:
            continue
        columns = group['columns']
        # Some clinical columns enter the raw frame as integers. Cast before
        # assigning standardized values to avoid dtype-dependent pandas errors.
        for column in columns:
            transformed[column] = transformed[column].astype(float)
        values = transformed.loc[:, columns].apply(pd.to_numeric, errors='coerce')
        values = values.fillna(pd.Series(group['fill_values']))
        mean = np.asarray(group['mean'], dtype=np.float64)
        scale = np.asarray(group['scale'], dtype=np.float64)
        scaled = (values.to_numpy() - mean) / scale
        for index, column in enumerate(columns):
            transformed[column] = scaled[:, index]
    return transformed


def held_out_frame(frame: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    split = parameters['data_split']
    seed = int(parameters['training']['random_seed'])
    fold = int(parameters['identity']['fold'])
    splitter = StratifiedGroupKFold(
        n_splits=int(split['outer_n_splits']),
        shuffle=True,
        random_state=seed,
    )
    folds = splitter.split(frame, frame['label'], frame[split['outer_group']])
    for index, (_, test_indices) in enumerate(folds, start=1):
        if index == fold:
            return frame.iloc[test_indices].reset_index(drop=True)
    raise ValueError(f'checkpoint requests fold {fold}, which does not exist')


def build_dataset(frame: pd.DataFrame, checkpoint: dict, project_root: Path):
    parameters = checkpoint['parameters']
    graph = parameters['graph']
    source_cfg = parameters['source_configuration']
    model_params = parameters['model']
    kwargs = {
        'sigma': graph['sigma'],
        'k_neighbors': graph['k_neighbors'],
        'graph_type': parameters['identity']['graph_type'],
        'graph_mix_alpha': graph['graph_mix_alpha'],
        'node_semantics_mode': graph['node_semantics_mode'],
        'precomputed_stats': checkpoint.get('zhang_stats'),
        'codeviation_threshold': graph['codeviation_threshold'],
        'codeviation_kernel_sigma': graph['codeviation_kernel_sigma'],
    }
    fma_cfg = {'model': {
        'fma_graph_path': graph.get('fma_graph_path'),
        'fma_embedding_backend': graph.get('fma_embedding_backend'),
        'fma_embedding_model': graph.get('fma_embedding_model'),
        'fma_embedding_dim': graph.get('fma_embedding_dim'),
        'fma_embedding_cache_dir': graph.get('fma_embedding_cache_dir'),
    }}
    kwargs.update(fma_dataset_kwargs(fma_cfg))
    coords = project_root / parameters['paths']['coordinates']
    if Path(parameters['paths']['coordinates']).is_absolute():
        coords = Path(parameters['paths']['coordinates'])
    if model_params['n_clinical_features']:
        kwargs['clinical_cols'] = model_params['clinical_columns']
        dataset_class = HybridAlzheimerDataset
    else:
        dataset_class = AlzheimerGraphDataset
    return dataset_class(frame, coords, **kwargs)


def build_model(checkpoint: dict, dataset, device: torch.device):
    params = checkpoint['parameters']['model']
    identity = checkpoint['parameters']['identity']
    common = dict(
        in_features=params['in_features'],
        n_clinical_features=params['n_clinical_features'],
        hidden_dim=params['hidden_dim'],
        n_classes=params['n_classes'],
        dropout=params['dropout'],
        use_kan_backbone=params['use_kan_backbone'],
        use_kan_head=params['use_kan_head'],
        n_layers=params['n_layers'],
        graph_type=params['graph_type'],
        hetero_metadata=dataset.metadata() if dataset.is_heterogeneous else None,
        hetero_input_dims=(
            {node_type: int(dataset[0][node_type].x.size(-1))
             for node_type in dataset.metadata()[0]}
            if dataset.is_heterogeneous else None
        ),
    )
    classes = {
        'GCN': GCN_KAN_Classifier,
        'GAT': GAT_KAN_Classifier,
        'DualGraph': DualGraph_KAN_Classifier,
    }
    model = classes[identity['model']](**common).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def prepare_sample(dataset, index: int, device: torch.device):
    sample = dataset[index]
    if dataset.is_heterogeneous:
        graph = Batch.from_data_list([sample]).to(device)
        clinical = getattr(graph, 'x_clin', None)
        return graph, None, clinical
    if len(sample) == 4:
        graph, adjacency, clinical, _ = sample
        return (
            graph.unsqueeze(0).to(device),
            adjacency.unsqueeze(0).to(device),
            clinical.unsqueeze(0).to(device),
        )
    graph, adjacency, _ = sample
    return graph.unsqueeze(0).to(device), adjacency.unsqueeze(0).to(device), None
