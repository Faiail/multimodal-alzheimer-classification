"""Fast synthetic smoke test for every graph topology, with and without FMA."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.batch_utils import (
    build_graph_dataloader,
    get_node_feature_dim,
    get_node_feature_dims,
    move_graph_batch_to_device,
)
from src.data.dataset import AlzheimerGraphDataset, HybridAlzheimerDataset
from src.data.preprocessing import FoldFeatureNormalizer, imaging_feature_columns
from src.models.model import (DualGraph_KAN_Classifier, GAT_KAN_Classifier,
                              GCN_KAN_Classifier)

GRAPH_TYPES = ("static", "zhang", "hybrid", "knn", "multigraph")
CLINICAL_COLS = [
    "AGE", "Gender_Male", "PTEDUCAT", "APOE4", "MMSE",
    "ABETA_final", "TAU_final", "PTAU_final", "CSF_Available",
]
ROI_IDS = (1, 2, 131, 132, 999)  # four mapped, one deliberately unmapped


def synthetic_inputs(root: Path):
    coords_path = root / "coords.csv"
    pd.DataFrame({
        "label_id": ROI_IDS,
        "x": [float(index * 10) for index in range(len(ROI_IDS))],
        "y": [0.0] * len(ROI_IDS),
        "z": [0.0] * len(ROI_IDS),
    }).to_csv(coords_path, index=False)
    rows = []
    for sample_index in range(4):
        row = {
            "label": sample_index % 3,
            **{column: float(sample_index + 1) for column in CLINICAL_COLS},
        }
        for modality_index, prefix in enumerate((
            "GM_vol_Region_", "WM_vol_Region_",
            "CSF_vol_Region_", "PET_suvr_Region_",
        )):
            for roi_index, roi_id in enumerate(ROI_IDS):
                scale = float((modality_index + 1) * (roi_index + 1))
                row[f"{prefix}{roi_id}"] = roi_id / 10.0 + scale * (sample_index + 1)
        rows.append(row)
    return coords_path, pd.DataFrame(rows)


def normalize_and_validate(frame, coords_path):
    train_raw = frame.iloc[:3].reset_index(drop=True)
    held_out_raw = frame.iloc[3:].reset_index(drop=True)
    raw_reference = AlzheimerGraphDataset(
        train_raw, coords_path, graph_type="zhang", precompute_adjacency=False
    )
    raw_stats = {"mean": raw_reference.global_mean, "std": raw_reference.global_std}
    raw_graph = AlzheimerGraphDataset(
        frame,
        coords_path,
        graph_type="zhang",
        precomputed_stats=raw_stats,
        codeviation_threshold=0.9,
    )
    raw_adjacency = raw_graph.adj_matrices[0]
    assert torch.is_tensor(raw_adjacency)
    assert torch.count_nonzero(raw_adjacency).item() == len(ROI_IDS) ** 2
    raw_features = np.concatenate([
        raw_graph.x_gm[0].numpy(), raw_graph.x_wm[0].numpy(),
        raw_graph.x_csf[0].numpy(), raw_graph.x_pet[0].numpy(),
    ])
    affinity = raw_graph._build_zhang_affinity(raw_features)
    legacy_scale = raw_graph._symmetric_degree_scale(affinity)
    legacy_degree = np.diag(legacy_scale)
    legacy_adjacency = torch.tensor(
        legacy_degree.dot(affinity).dot(legacy_degree), dtype=torch.float32
    )
    assert torch.allclose(raw_adjacency, legacy_adjacency, atol=2e-7, rtol=1e-6)

    optimized_graph = AlzheimerGraphDataset(
        frame,
        coords_path,
        graph_type="multigraph",
        precomputed_stats=raw_stats,
        codeviation_threshold=0.9,
    )
    expected_edges = raw_graph._dense_to_edges(raw_adjacency)
    optimized_edges = optimized_graph.adj_matrices[0]["codeviation"]
    assert torch.equal(expected_edges[0], optimized_edges[0])
    assert torch.allclose(expected_edges[1], optimized_edges[1], atol=2e-7, rtol=1e-6)

    zhang_fma = AlzheimerGraphDataset(
        frame,
        coords_path,
        graph_type="zhang",
        node_semantics_mode="fma_ontology",
        precomputed_stats=raw_stats,
        codeviation_threshold=0.9,
        fma_embedding_backend="hash",
        fma_embedding_dim=24,
        fma_embedding_cache_dir=coords_path.parent / "cache",
    )
    stored_zhang_edges = zhang_fma.adj_matrices[0]
    assert isinstance(stored_zhang_edges, tuple)
    assert torch.equal(expected_edges[0], stored_zhang_edges[0])
    assert torch.allclose(expected_edges[1], stored_zhang_edges[1], atol=2e-7, rtol=1e-6)
    zhang_fma_sample = zhang_fma[0]
    zhang_edge_type = ("brain_region", "codeviation", "brain_region")
    assert zhang_edge_type in zhang_fma_sample.edge_types
    assert zhang_fma_sample[zhang_edge_type].edge_index is stored_zhang_edges[0]
    assert zhang_fma_sample[zhang_edge_type].edge_weight is stored_zhang_edges[1]
    mapping_edges = zhang_fma_sample[
        "brain_region", "maps_to", "fma_entity"
    ].edge_index
    roi_to_entity = {
        int(source): int(target)
        for source, target in mapping_edges.t().tolist()
    }
    roi_positions = {roi_id: index for index, roi_id in enumerate(ROI_IDS)}
    assert roi_positions[131] in roi_to_entity
    assert roi_positions[132] in roi_to_entity
    assert roi_to_entity[roi_positions[131]] == roi_to_entity[roi_positions[132]]
    assert (
        zhang_fma.fma_graph.entity_ids[roi_to_entity[roi_positions[131]]]
        == "0327261"
    )

    knn_dense = AlzheimerGraphDataset(
        frame,
        coords_path,
        graph_type="knn",
        k_neighbors=2,
    )
    dense_knn_adjacency = knn_dense.adj_matrices[0]
    assert torch.is_tensor(dense_knn_adjacency)
    expected_knn_edges = knn_dense._dense_to_edges(dense_knn_adjacency)
    knn_fma = AlzheimerGraphDataset(
        frame,
        coords_path,
        graph_type="knn",
        k_neighbors=2,
        node_semantics_mode="fma_ontology",
        fma_embedding_backend="hash",
        fma_embedding_dim=24,
        fma_embedding_cache_dir=coords_path.parent / "cache",
    )
    stored_knn_edges = knn_fma.adj_matrices[0]
    assert isinstance(stored_knn_edges, tuple)
    assert torch.equal(expected_knn_edges[0], stored_knn_edges[0])
    assert torch.allclose(
        expected_knn_edges[1], stored_knn_edges[1], atol=2e-7, rtol=1e-6
    )
    knn_fma_sample = knn_fma[0]
    knn_edge_type = ("brain_region", "feature_similarity", "brain_region")
    assert knn_edge_type in knn_fma_sample.edge_types
    assert knn_fma_sample[knn_edge_type].edge_index is stored_knn_edges[0]
    assert knn_fma_sample[knn_edge_type].edge_weight is stored_knn_edges[1]

    cached_proximity = optimized_graph.adj_matrices[0]["proximity"]
    for relations in optimized_graph.adj_matrices[1:]:
        assert relations["proximity"][0] is cached_proximity[0]
        assert relations["proximity"][1] is cached_proximity[1]

    imaging_cols = imaging_feature_columns(frame)
    normalizer = FoldFeatureNormalizer(imaging_cols, CLINICAL_COLS)
    train, held_out = normalizer.fit_transform_many(train_raw, held_out_raw)

    train_imaging = train[imaging_cols].to_numpy()
    train_clinical = train[CLINICAL_COLS].to_numpy()
    assert np.allclose(train_imaging.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train_imaging.std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(train_clinical.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train_clinical.std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(held_out[imaging_cols].to_numpy(), np.sqrt(6.0), atol=1e-6)
    assert np.allclose(held_out[CLINICAL_COLS].to_numpy(), np.sqrt(6.0), atol=1e-6)

    for prefix in ("GM_vol_Region_", "WM_vol_Region_", "CSF_vol_Region_", "PET_suvr_Region_"):
        columns = [column for column in imaging_cols if column.startswith(prefix)]
        assert len(columns) == len(ROI_IDS)
        assert np.allclose(train[columns].to_numpy().std(axis=0), 1.0, atol=1e-6)

    reference = AlzheimerGraphDataset(
        train, coords_path, graph_type="multigraph", precompute_adjacency=False
    )
    stats = {"mean": reference.global_mean, "std": reference.global_std}
    normalized = pd.concat([train, held_out], ignore_index=True)
    graph_dataset = AlzheimerGraphDataset(
        normalized,
        coords_path,
        graph_type="multigraph",
        precomputed_stats=stats,
        codeviation_threshold=0.9,
    )
    sample = graph_dataset[0]
    assert sample["brain_region"].x.shape == (len(ROI_IDS), 4)
    assert set(sample.edge_types) == {
        ("brain_region", "proximity", "brain_region"),
        ("brain_region", "codeviation", "brain_region"),
    }
    codeviation_edges = sample[
        "brain_region", "codeviation", "brain_region"
    ].edge_index
    assert codeviation_edges.size(1) == len(ROI_IDS) ** 2
    assert torch.isfinite(
        sample["brain_region", "codeviation", "brain_region"].edge_weight
    ).all()
    print(
        "PASS fold-safe normalization, dense/direct edge equivalence, "
        "cached proximity, sparse Zhang+FMA, sparse KNN+FMA and multigraph relations"
    )
    return normalized


def exercise_dataset(frame, coords_path, root, graph_type, semantics, backend):
    dataset = HybridAlzheimerDataset(
        frame,
        coords_path,
        CLINICAL_COLS,
        sigma=15.0,
        k_neighbors=2,
        graph_type=graph_type,
        node_semantics_mode=semantics,
        fma_embedding_backend=backend,
        fma_embedding_dim=24,
        fma_embedding_cache_dir=root / "cache",
    )
    sample = dataset[0]
    if semantics == "fma_ontology":
        assert sample["brain_region"].x.size(1) == 4
        assert sample["brain_region", "maps_to", "fma_entity"].edge_index.size(1) == 4
        assert ("fma_entity", "part_of", "fma_entity") in sample.edge_types
        assert ("fma_entity", "is_a", "fma_entity") in sample.edge_types

    batch = next(iter(build_graph_dataloader(dataset, batch_size=2, shuffle=False)))
    x_graph, adjacency, clinical, labels = move_graph_batch_to_device(
        batch, torch.device("cpu"), graph_type=graph_type, has_clinical=True
    )
    metadata = dataset.metadata() if dataset.is_heterogeneous else None
    input_dims = get_node_feature_dims(dataset)
    for classifier in (GCN_KAN_Classifier, GAT_KAN_Classifier, DualGraph_KAN_Classifier):
        model = classifier(
            get_node_feature_dim(dataset),
            len(CLINICAL_COLS),
            8,
            n_classes=3,
            dropout=0.1,
            use_kan_backbone=False,
            use_kan_head=False,
            n_layers=2,
            graph_type=graph_type,
            hetero_metadata=metadata,
            hetero_input_dims=input_dims,
        )
        output = model(x_graph, adjacency, clinical)
        torch.nn.functional.cross_entropy(output, labels).backward()
        assert output.shape == (2, 3)
    print(f"PASS semantics={semantics:<12} graph={graph_type:<10}")
    return dataset, (x_graph, adjacency, clinical, labels)



def exercise_imaging_dataset(frame, coords_path, root, graph_type, backend):
    dataset = AlzheimerGraphDataset(
        frame,
        coords_path,
        sigma=15.0,
        k_neighbors=2,
        graph_type=graph_type,
        node_semantics_mode="fma_ontology",
        fma_embedding_backend=backend,
        fma_embedding_dim=24,
        fma_embedding_cache_dir=root / "cache",
    )
    batch = next(iter(build_graph_dataloader(dataset, batch_size=2, shuffle=False)))
    x_graph, adjacency, labels = move_graph_batch_to_device(
        batch, torch.device("cpu"), graph_type=graph_type, has_clinical=False
    )
    model = GCN_KAN_Classifier(
        get_node_feature_dim(dataset),
        0,
        8,
        n_classes=3,
        dropout=0.1,
        use_kan_backbone=False,
        use_kan_head=False,
        n_layers=2,
        graph_type=graph_type,
        hetero_metadata=dataset.metadata(),
        hetero_input_dims=get_node_feature_dims(dataset),
    )
    output = model(x_graph, adjacency, None)
    torch.nn.functional.cross_entropy(output, labels).backward()
    assert output.shape == (2, 3)
    print(f"PASS modality=imaging graph={graph_type:<10} with FMA")

def run_smoke_test(backend: str = "hash"):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        coords_path, frame = synthetic_inputs(root)
        frame = normalize_and_validate(frame, coords_path)
        for semantics in ("none", "fma_ontology"):
            for graph_type in GRAPH_TYPES:
                exercise_dataset(
                    frame, coords_path, root, graph_type, semantics, backend
                )
        for graph_type in GRAPH_TYPES:
            exercise_imaging_dataset(frame, coords_path, root, graph_type, backend)

        spatial = HybridAlzheimerDataset(
            frame,
            coords_path,
            CLINICAL_COLS,
            graph_type="static",
            node_semantics_mode="fma_spatial",
            fma_embedding_backend=backend,
            fma_embedding_dim=24,
            fma_embedding_cache_dir=root / "cache",
        )
        assert spatial[0]["brain_region"].x.size(1) == 9
        print("PASS semantics=fma_spatial ROI features=4 imaging+5 spatial")

        dataset, inputs = exercise_dataset(
            frame, coords_path, root, "multigraph", "fma_ontology", backend
        )
        x_graph, adjacency, clinical, labels = inputs
        for classifier in (GCN_KAN_Classifier, GAT_KAN_Classifier, DualGraph_KAN_Classifier):
            model = classifier(
                get_node_feature_dim(dataset),
                len(CLINICAL_COLS),
                8,
                n_classes=3,
                dropout=0.1,
                use_kan_backbone=True,
                use_kan_head=False,
                n_layers=2,
                graph_type="multigraph",
                hetero_metadata=dataset.metadata(),
                hetero_input_dims=get_node_feature_dims(dataset),
            )
            output = model(x_graph, adjacency, clinical)
            torch.nn.functional.cross_entropy(output, labels).backward()
        print("PASS FullKAN GCN/GAT/DualGraph with multigraph+FMA")
    return {"status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pubmedbert",
        action="store_true",
        help="Use/download the production biomedical embeddings instead of hash vectors.",
    )
    args = parser.parse_args()
    backend = "pubmedbert" if args.pubmedbert else "hash"
    run_smoke_test(backend=backend)


if __name__ == "__main__":
    main()
