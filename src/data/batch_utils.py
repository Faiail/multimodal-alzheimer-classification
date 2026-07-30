import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeometricDataLoader


def fma_dataset_kwargs(cfg):
    """Common FMA graph/embedding options shared by every training entrypoint."""
    model_cfg = cfg['model']
    return {
        'fma_graph_path': model_cfg.get('fma_graph_path'),
        'fma_embedding_backend': model_cfg.get('fma_embedding_backend', 'pubmedbert'),
        'fma_embedding_model': model_cfg.get(
            'fma_embedding_model', 'NeuML/pubmedbert-base-embeddings'
        ),
        'fma_embedding_dim': int(model_cfg.get('fma_embedding_dim', 128)),
        'fma_embedding_cache_dir': model_cfg.get('fma_embedding_cache_dir'),
    }


def build_graph_dataloader(dataset, **kwargs):
    """Use PyG batching whenever the dataset is heterogeneous."""
    loader_cls = GeometricDataLoader if dataset.is_heterogeneous else torch.utils.data.DataLoader
    return loader_cls(dataset, **kwargs)


def get_node_feature_dim(dataset):
    """Return the feature width for either tuple samples or HeteroData samples."""
    sample = dataset[0]
    if isinstance(sample, HeteroData):
        return sample[dataset.NODE_TYPE].x.size(-1)
    return sample[0].shape[1]


def get_node_feature_dims(dataset):
    """Return every node type's input width for heterogeneous model encoders."""
    sample = dataset[0]
    if not isinstance(sample, HeteroData):
        return None
    return {node_type: sample[node_type].x.size(-1) for node_type in sample.node_types}


def move_graph_batch_to_device(batch, device, graph_type='static', has_clinical=False):
    """Move a batch to the device.

    Multigraph batches are a single PyG HeteroData/Batch with one node type and
    two relation types. Homogeneous graph modes retain the original dense format.
    """
    if isinstance(batch, HeteroData):
        if not batch.node_types:
            raise ValueError("Received an empty HeteroData batch")
        graph = batch.to(device)
        y = graph.y.view(-1)
        if has_clinical:
            return graph, None, graph.x_clin, y
        return graph, None, y

    if has_clinical:
        x_graph, adj, x_clin, y = batch
        return x_graph.to(device), adj.to(device), x_clin.to(device), y.to(device)

    x_graph, adj, y = batch
    return x_graph.to(device), adj.to(device), y.to(device)
