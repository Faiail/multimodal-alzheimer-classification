"""Figures used to document graph construction in the methodology chapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import sys
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.fma_graph import FMAKnowledgeGraph
from explain.checkpoint import apply_normalizer_state
from train_multiclass import prepare_multiclass_data
from visualization.explanation_figures import (
    _mni_axial_slice,
    find_deformation,
    find_patient_mri,
    warp_mri_to_mni,
)
from nilearn import datasets


CLASS_NAMES = ("CN", "MCI", "AD")
IMAGING_PREFIXES = ("GM_vol_Region_", "WM_vol_Region_", "CSF_vol_Region_", "PET_suvr_Region_")


def _symmetric_normalize(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1) + 1e-6
    scale = np.power(degree, -0.5)
    return adjacency * scale[:, None] * scale[None, :]


def spatial_adjacency(coordinates: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Reproduce the static dataset spatial-proximity adjacency."""
    distances = cdist(coordinates, coordinates, metric="euclidean")
    adjacency = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    adjacency[adjacency < 0.1] = 0.0
    np.fill_diagonal(adjacency, 1.0)
    return _symmetric_normalize(adjacency)


def codeviation_adjacency(
    features: np.ndarray,
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
    threshold: float = 0.9,
    kernel_sigma: float = np.sqrt(2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the dataset's reference-based co-deviation graph."""
    n_regions = features.shape[0]
    flat = np.concatenate([features[:, index] for index in range(4)])
    delta = (flat - reference_mean) / np.maximum(reference_std, 1e-6)
    delta_nodes = delta.reshape(4, n_regions).T
    distances = cdist(delta_nodes, delta_nodes, metric="euclidean")
    adjacency = np.exp(-(distances ** 2) / (2.0 * kernel_sigma ** 2))
    adjacency[adjacency < threshold] = 0.0
    np.fill_diagonal(adjacency, 1.0)
    return _symmetric_normalize(adjacency), delta_nodes


def _coordinates_2d(coordinates: np.ndarray) -> np.ndarray:
    """Use the MNI x/y projection used by axial atlas visualisations."""
    return np.asarray(coordinates, dtype=float)[:, :2]


def _edge_segments(adjacency: np.ndarray, coordinates_2d: np.ndarray):
    upper = np.transpose(np.nonzero(np.triu(adjacency, k=1)))
    if not len(upper):
        return upper, [], np.asarray([])
    segments = [coordinates_2d[[source, target]] for source, target in upper]
    weights = np.asarray([adjacency[source, target] for source, target in upper])
    return upper, segments, weights


def _draw_graph_panel(
    axis,
    adjacency: np.ndarray,
    coordinates_2d: np.ndarray,
    title: str,
    edge_colour: str,
    node_values: np.ndarray | None = None,
    node_label: str | None = None,
    background=None,
    background_label: str | None = None,
):
    image_extent = None
    if background is not None:
        section, image_extent = _mni_axial_slice(background, world_z=0.0)
        nonzero = section[np.isfinite(section) & (section != 0)]
        low, high = (
            np.percentile(nonzero, [1, 99.5])
            if nonzero.size else (0.0, 1.0)
        )
        axis.imshow(
            section, cmap="gray", vmin=low, vmax=high,
            origin="lower", extent=image_extent, zorder=0,
        )
        axis.set_facecolor("black")

    upper, segments, weights = _edge_segments(adjacency, coordinates_2d)
    if len(weights):
        low = float(np.percentile(weights, 5))
        high = float(np.percentile(weights, 99))
        if high <= low:
            low, high = float(weights.min()), float(weights.max())
        scaled = np.ones_like(weights) if high <= low else np.clip((weights - low) / (high - low), 0, 1)
        scaled = np.sqrt(scaled)
        line_collection = LineCollection(
            segments,
            colors=[(*mcolors.to_rgb(edge_colour), 0.20 + 0.55 * value) for value in scaled],
            linewidths=0.45 + 2.5 * scaled,
            zorder=1,
        )
        axis.add_collection(line_collection)

    if node_values is None:
        axis.scatter(
            coordinates_2d[:, 0], coordinates_2d[:, 1],
            s=21, color="#5B8DB8", edgecolors="white", linewidths=0.35,
            zorder=2,
        )
    else:
        values = np.asarray(node_values, dtype=float)
        maximum = values.max(initial=0.0)
        values = values / maximum if maximum > 0 else np.zeros_like(values)
        axis.scatter(
            coordinates_2d[:, 0], coordinates_2d[:, 1],
            c=values, cmap="viridis", vmin=0, vmax=1,
            s=18 + 38 * values, edgecolors="white", linewidths=0.35,
            zorder=2,
        )

    axis.set_title(title, fontsize=12, pad=10)
    axis.set_xlabel("MNI x (mm)")
    axis.set_ylabel("MNI y (mm)")
    axis.set_aspect("equal", adjustable="box")
    if background is not None:
        # The equal-aspect MRI image is shorter than the allocated subplot
        # slot; anchor it upward to reduce the gap to the figure legend.
        axis.set_anchor((0.5, 0.82))
    if background is None:
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.set_xlim(coordinates_2d[:, 0].min() - 8, coordinates_2d[:, 0].max() + 8)
        axis.set_ylim(coordinates_2d[:, 1].min() - 8, coordinates_2d[:, 1].max() + 8)
    else:
        axis.set_xlim(image_extent[0], image_extent[1])
        axis.set_ylim(image_extent[2], image_extent[3])
        axis.text(0.03, 0.96, "L", transform=axis.transAxes, va="top", color="white")
        axis.text(0.97, 0.96, "R", transform=axis.transAxes, ha="right", va="top", color="white")
    if node_label:
        axis.text(
            0.02, 0.02, node_label, transform=axis.transAxes,
            fontsize=8, color="white" if background is not None else "black",
            bbox={"facecolor": "black" if background is not None else "white",
                  "alpha": 0.68 if background is not None else 0.78,
                  "edgecolor": "none"},
        )
    if background_label:
        axis.text(
            0.98, 0.02, background_label, transform=axis.transAxes,
            fontsize=7, color="white", ha="right",
            bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
        )
    return len(upper)


def _fold_training_and_test(frame: pd.DataFrame, parameters: dict):
    split = parameters["data_split"]
    fold = int(parameters["identity"]["fold"])
    seed = int(parameters["training"]["random_seed"])
    splitter = StratifiedGroupKFold(
        n_splits=int(split["outer_n_splits"]),
        shuffle=True,
        random_state=seed,
    )
    for index, (train_indices, test_indices) in enumerate(
        splitter.split(frame, frame["label"], frame[split["outer_group"]]), start=1
    ):
        if index == fold:
            outer_train = frame.iloc[train_indices].reset_index(drop=True)
            outer_test = frame.iloc[test_indices].reset_index(drop=True)
            inner_indices, _ = train_test_split(
                np.arange(len(outer_train)),
                test_size=float(split["inner_validation_fraction"]),
                stratify=outer_train["label"],
                random_state=seed,
            )
            return outer_train, outer_test, outer_train.iloc[inner_indices].reset_index(drop=True)
    raise ValueError(f"Checkpoint requests fold {fold}, which does not exist")


def _imaging_matrix(frame: pd.DataFrame, label_ids: np.ndarray) -> np.ndarray:
    groups = []
    for prefix in IMAGING_PREFIXES:
        columns = [f"{prefix}{int(label)}" for label in label_ids]
        groups.append(frame.loc[:, columns].to_numpy(dtype=float))
    return np.stack(groups, axis=2)


def _reference_statistics(
    checkpoint: dict,
    inner_train: pd.DataFrame,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    saved = checkpoint.get("zhang_stats")
    if saved is not None:
        return np.asarray(saved["mean"], dtype=float), np.asarray(saved["std"], dtype=float)
    matrix = _imaging_matrix(inner_train, label_ids)
    flat = np.concatenate([matrix[:, :, index] for index in range(4)], axis=1)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-6


def _load_mri_background(record: dict, mri_root: str | Path | None, segmentation_root: str | Path | None):
    """Load the subject MRI warped to MNI, with the same template fallback as explanations."""
    if mri_root is None:
        return datasets.load_mni152_template(resolution=1), "MNI template"
    mri_path = find_patient_mri(record, mri_root)
    deformation_path = find_deformation(mri_path, segmentation_root) if mri_path else None
    if mri_path is not None and deformation_path is not None:
        try:
            return warp_mri_to_mni(mri_path, deformation_path), "patient T1 MRI, normalized to MNI"
        except Exception:
            pass
    return datasets.load_mni152_template(resolution=1), "MNI template (patient MRI unavailable)"


def _load_methodology_record(checkpoint_path: str | Path, sample_index: int):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frame = prepare_multiclass_data()
    outer_train, outer_test, inner_train = _fold_training_and_test(
        frame, checkpoint["parameters"]
    )
    del outer_train
    test = apply_normalizer_state(outer_test, checkpoint.get("normalizer_state"))
    inner = apply_normalizer_state(inner_train, checkpoint.get("normalizer_state"))

    coords_path = PROJECT_ROOT / checkpoint["parameters"]["paths"]["coordinates"]
    if Path(checkpoint["parameters"]["paths"]["coordinates"]).is_absolute():
        coords_path = Path(checkpoint["parameters"]["paths"]["coordinates"])
    coordinates_frame = pd.read_csv(coords_path)
    available = {
        int(column.rsplit("_", 1)[-1])
        for column in test.columns
        if column.startswith("GM_vol_Region_")
    }
    coordinates_frame = coordinates_frame[
        coordinates_frame["label_id"].isin(available)
    ].sort_values("label_id").reset_index(drop=True)
    label_ids = coordinates_frame["label_id"].to_numpy(dtype=int)
    coordinates = coordinates_frame[["x", "y", "z"]].to_numpy(dtype=float)
    if sample_index < 0 or sample_index >= len(test):
        raise IndexError(f"sample_index={sample_index} outside held-out fold of size {len(test)}")
    sample = test.iloc[[sample_index]]
    features = _imaging_matrix(sample, label_ids)[0]
    mean, std = _reference_statistics(checkpoint, inner, label_ids)
    graph = checkpoint["parameters"]["graph"]
    spatial = spatial_adjacency(coordinates, float(graph.get("sigma", 15.0)))
    codeviation, delta_nodes = codeviation_adjacency(
        features,
        mean,
        std,
        threshold=float(graph.get("codeviation_threshold", 0.9)),
        kernel_sigma=float(graph.get("codeviation_kernel_sigma", np.sqrt(2.0))),
    )
    record_id = str(sample.iloc[0].get("PTID", sample.iloc[0].get("subject_id", sample_index)))
    class_name = CLASS_NAMES[int(sample.iloc[0]["label"])]
    return {
        "checkpoint": checkpoint,
        "record_id": record_id,
        "class_name": class_name,
        "row": sample.iloc[0].to_dict(),
        "label_ids": label_ids,
        "coordinates": coordinates,
        "spatial": spatial,
        "codeviation": codeviation,
        "delta_nodes": delta_nodes,
    }


def plot_spatial_vs_codeviation(
    checkpoint_path: str | Path,
    output_path: str | Path,
    sample_index: int = 0,
    mri_root: str | Path | None = None,
    segmentation_root: str | Path | None = None,
    content: dict | None = None,
):
    """Create spatial and co-deviation graphs over the selected subject MRI."""
    content = content or _load_methodology_record(checkpoint_path, sample_index)
    coordinates_2d = _coordinates_2d(content["coordinates"])
    background, background_label = _load_mri_background(
        content["row"], mri_root, segmentation_root
    )
    spatial_count = int(np.count_nonzero(np.triu(content["spatial"], k=1)))
    codeviation_count = int(np.count_nonzero(np.triu(content["codeviation"], k=1)))
    graph = content["checkpoint"]["parameters"]["graph"]
    # A taller canvas gives the axial slices more room while keeping a
    # dedicated band for the legend and a clear footer below the x labels.
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 9.0), constrained_layout=False)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.07, top=0.87, wspace=0.45)
    _draw_graph_panel(
        axes[0], content["spatial"], coordinates_2d,
        "Static spatial-proximity graph over T1 MRI",
        "#2C6EAA",
        node_label=f"165 active AAL3 ROIs\n{spatial_count} undirected edges",
        background=background,
        background_label=background_label,
    )
    _draw_graph_panel(
        axes[1], content["codeviation"], coordinates_2d,
        f"Subject-specific co-deviation over T1 MRI ({content['class_name']}, {content['record_id']})",
        "#D97732",
        node_values=np.linalg.norm(content["delta_nodes"], axis=1),
        node_label=(
            f"165 active AAL3 ROIs\n{codeviation_count} undirected edges\n"
            f"threshold={graph.get('codeviation_threshold', 0.9):.2f}"
        ),
        background=background,
        background_label=background_label,
    )
    figure.suptitle(
        "Spatial proximity versus reference-based co-deviation",
        fontsize=15, y=0.985,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color="#2C6EAA", lw=2.5, label="Spatial affinity"),
            Line2D([0], [0], color="#D97732", lw=2.5, label="Co-deviation affinity"),
        ],
        loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.text(
        0.5, 0.154,
        "Edges are projected onto the MNI axial T1 slice; line width and opacity encode normalized affinity.",
        ha="center", fontsize=8.5,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _normalise_positions(positions: np.ndarray, x_range, y_range):
    lower = positions.min(axis=0, keepdims=True)
    upper = positions.max(axis=0, keepdims=True)
    unit = (positions - lower) / np.maximum(upper - lower, 1e-9)
    return np.column_stack([
        x_range[0] + (x_range[1] - x_range[0]) * unit[:, 0],
        y_range[0] + (y_range[1] - y_range[0]) * unit[:, 1],
    ])


def _fma_from_checkpoint(checkpoint: dict, roi_label_ids: np.ndarray):
    graph = checkpoint["parameters"]["graph"]
    graph_path = graph.get("fma_graph_path")
    embedding_backend = graph.get("fma_embedding_backend") or "pubmedbert"
    embedding_model = graph.get("fma_embedding_model")
    embedding_dim = int(graph.get("fma_embedding_dim") or 128)
    cache_dir = graph.get("fma_embedding_cache_dir")
    return FMAKnowledgeGraph(
        roi_label_ids,
        graph_path=graph_path,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model or "NeuML/pubmedbert-base-embeddings",
        embedding_dim=embedding_dim,
        cache_dir=cache_dir,
    )


def _edge_array(edge_tensor) -> np.ndarray:
    return edge_tensor.detach().cpu().numpy().T if hasattr(edge_tensor, "detach") else np.asarray(edge_tensor).T


def plot_fma_knowledge_graph(
    checkpoint_path: str | Path,
    output_path: str | Path,
    roi_label_ids: np.ndarray | None = None,
    sample_content: dict | None = None,
    mri_root: str | Path | None = None,
    segmentation_root: str | Path | None = None,
):
    """Plot the ROI/FMA topology, feature schema, and ROI nodes over T1 MRI."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if roi_label_ids is None:
        # The active atlas set is deterministic for this project; recover it from the
        # coordinate table rather than requiring a patient sample export.
        coords_path = PROJECT_ROOT / checkpoint["parameters"]["paths"]["coordinates"]
        coords = pd.read_csv(coords_path)
        roi_label_ids = coords["label_id"].to_numpy(dtype=int)
        roi_label_ids = np.asarray([
            rid for rid in roi_label_ids
            if f"GM_vol_Region_{int(rid)}" in pd.read_csv(
                PROJECT_ROOT / "data" / "04_final" / "all_features.csv", nrows=1
            ).columns
        ])
    roi_label_ids = np.asarray(roi_label_ids, dtype=int)
    fma = _fma_from_checkpoint(checkpoint, roi_label_ids)
    background = None
    background_label = None
    if sample_content is not None:
        background, background_label = _load_mri_background(
            sample_content["row"], mri_root, segmentation_root
        )
    coords_path = PROJECT_ROOT / checkpoint["parameters"]["paths"]["coordinates"]
    coords_frame = pd.read_csv(coords_path).set_index("label_id").loc[roi_label_ids]
    coordinates = coords_frame[["x", "y", "z"]].to_numpy(dtype=float)
    roi_xy = _normalise_positions(_coordinates_2d(coordinates), (0.05, 0.37), (0.10, 0.90))

    hierarchy = nx.Graph()
    hierarchy.add_nodes_from(range(len(fma.entity_ids)))
    for relation in ("is_a", "part_of"):
        edges = _edge_array(fma.edges[relation])
        hierarchy.add_edges_from((int(source), int(target)) for source, target in edges)
    raw_fma = np.asarray(list(nx.spring_layout(hierarchy, seed=42, iterations=250).values()))
    fma_xy = _normalise_positions(raw_fma, (0.46, 0.73), (0.10, 0.90))

    figure = plt.figure(figsize=(18, 9))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.45, 3.5, 1.3), wspace=0.04)
    mri_axis = figure.add_subplot(grid[0, 0])
    axis = figure.add_subplot(grid[0, 1])
    schema_axis = figure.add_subplot(grid[0, 2])
    axis.set_title(
        "FMA entities and anatomical relations",
        fontsize=13,
    )

    mapping = _edge_array(fma.roi_to_entity)

    relation_styles = {"is_a": ("#D55E00", "-"), "part_of": ("#009E73", "--")}
    for relation, (colour, linestyle) in relation_styles.items():
        edges = _edge_array(fma.edges[relation])
        if not len(edges):
            continue
        segments = [fma_xy[[source, target]] for source, target in edges]
        axis.add_collection(LineCollection(
            segments, colors=colour, linewidths=0.85, linestyles=linestyle,
            alpha=0.58, zorder=1,
        ))

    fma_degree = np.zeros(len(fma.entity_ids), dtype=float)
    for relation in ("is_a", "part_of"):
        edges = _edge_array(fma.edges[relation])
        if len(edges):
            fma_degree += np.bincount(edges.ravel(), minlength=len(fma.entity_ids))
    degree_max = fma_degree.max(initial=1.0)

    if background is not None:
        section, image_extent = _mni_axial_slice(background, world_z=0.0)
        nonzero = section[np.isfinite(section) & (section != 0)]
        low, high = (
            np.percentile(nonzero, [1, 99.5])
            if nonzero.size else (0.0, 1.0)
        )
        mri_axis.imshow(
            section, cmap="gray", vmin=low, vmax=high,
            origin="lower", extent=image_extent,
        )
        mapped_scores = np.zeros(len(roi_label_ids), dtype=float)
        if mapping.size:
            mapped_scores[mapping[:, 0]] = fma_degree[mapping[:, 1]]
        maximum = mapped_scores.max(initial=0.0)
        if maximum > 0:
            mapped_scores /= maximum
        mri_axis.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=mapped_scores, cmap="inferno", vmin=0, vmax=1,
            s=22 + 42 * mapped_scores, edgecolors="#202020", linewidths=0.45,
            zorder=3,
        )
        for roi_index in np.argsort(mapped_scores)[::-1][:8]:
            if mapped_scores[roi_index] <= 0:
                continue
            mri_axis.annotate(
                str(int(roi_label_ids[roi_index])),
                coordinates[roi_index, :2], xytext=(3, 3),
                textcoords="offset points", fontsize=6, color="white",
            )
        mri_axis.set_xlim(image_extent[0], image_extent[1])
        mri_axis.set_ylim(image_extent[2], image_extent[3])
        mri_axis.set_aspect("equal", adjustable="box")
        mri_axis.text(0.03, 0.96, "L", transform=mri_axis.transAxes, va="top", color="white")
        mri_axis.text(0.97, 0.96, "R", transform=mri_axis.transAxes, ha="right", va="top", color="white")
        mri_axis.text(
            0.03, 0.02,
            "Node colour/size = mapped FMA degree",
            transform=mri_axis.transAxes, fontsize=7, color="white",
            bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
        )
        mri_axis.text(
            0.97, 0.02, background_label, transform=mri_axis.transAxes,
            fontsize=7, color="white", ha="right",
            bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
        )
        mri_axis.set_title("Patient-specific ROI nodes over T1 MRI", fontsize=11, pad=10)
        mri_axis.set_xlabel("MNI x (mm)")
        mri_axis.set_ylabel("MNI y (mm)")
    else:
        mri_axis.axis("off")
        mri_axis.text(0.5, 0.5, "MRI overlay unavailable", ha="center", va="center")

    # Connect the actual ROI points on the MRI panel directly to the FMA
    # entities they map to.  The connection uses each axes' data coordinates,
    # so it remains anchored to the MRI anatomy rather than to an abstract
    # ROI layout in the centre panel.
    if background is not None and mapping.size:
        for source, target in mapping:
            connection = ConnectionPatch(
                xyA=coordinates[int(source), :2],
                xyB=fma_xy[int(target)],
                coordsA="data", coordsB="data",
                axesA=mri_axis, axesB=axis,
                color="#0072B2", linewidth=0.58, alpha=0.22,
                zorder=1.5, connectionstyle="arc3,rad=0.0",
            )
            connection.set_clip_on(False)
            axis.add_artist(connection)

    axis.scatter(
        fma_xy[:, 0], fma_xy[:, 1], s=20 + 40 * fma_degree / degree_max,
        c=fma_degree, cmap="viridis", edgecolors="#202020", linewidths=0.35,
        label="fma_entity", zorder=3,
    )

    # Put the most connected entities in a dedicated label column.  Leader
    # lines preserve the node association without stacking text on the dense
    # ontology layout.
    top_nodes = np.argsort(fma_degree)[::-1][:10]
    top_nodes = sorted(top_nodes, key=lambda node: fma_xy[node, 1], reverse=True)
    label_y = np.linspace(0.84, 0.18, len(top_nodes))
    axis.text(0.77, 0.93, "Highest-degree FMA entities", fontsize=8,
              color="#6B420D", ha="left", va="center")
    for node, target_y in zip(top_nodes, label_y):
        label = str(fma.entity_labels[node])
        if len(label) > 23:
            label = label[:20] + "..."
        axis.annotate(
            label, xy=fma_xy[node], xycoords="data",
            xytext=(0.77, target_y), textcoords="data",
            ha="left", va="center", fontsize=6.7,
            bbox={"facecolor": "white", "alpha": 0.90,
                  "edgecolor": "#AAAAAA", "linewidth": 0.25, "pad": 0.25},
            arrowprops={"arrowstyle": "-", "color": "#555555",
                        "linewidth": 0.60, "alpha": 0.85,
                        "connectionstyle": "arc3,rad=0.12"},
            zorder=5,
        )

    axis.text(0.46, 0.94, "Patient-independent FMA ontology", transform=axis.transAxes, color="#8A5411", fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    node_legend = axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3976A8",
                   markersize=6, label="brain_region (MRI panel)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#35B779",
                   markersize=6, label="fma_entity"),
        ],
        loc="lower left", frameon=True, fontsize=7,
    )
    axis.add_artist(node_legend)

    relation_handles = [
        Line2D([0], [0], color="#0072B2", lw=2.5, label="MRI ROI → FMA entity"),
        Line2D([0], [0], color="#D55E00", lw=2.5, label="is_a / has_subclass"),
        Line2D([0], [0], color="#009E73", lw=2.5, linestyle="--", label="part_of / has_part"),
    ]
    axis.legend(handles=relation_handles, loc="lower left", bbox_to_anchor=(0.02, 0.20),
                frameon=True, fontsize=7)

    schema_axis.axis("off")
    schema_axis.set_xlim(0, 1)
    schema_axis.set_ylim(0, 1)
    schema_axis.text(0.5, 0.97, "Feature and relation schema", ha="center", va="top", fontsize=12)

    def box(y, colour, title, body):
        patch = FancyBboxPatch(
            (0.10, y), 0.80, 0.16, boxstyle="round,pad=0.02",
            facecolor=colour, edgecolor="#555555", linewidth=0.8, alpha=0.9,
        )
        schema_axis.add_patch(patch)
        schema_axis.text(0.50, y + 0.115, title, ha="center", va="center", fontsize=10, weight="bold")
        schema_axis.text(0.50, y + 0.045, body, ha="center", va="center", fontsize=8)

    box(0.72, "#DCEAF5", "brain_region", "165 nodes · imaging x_i ∈ R^4\n(+5 spatial descriptors in fma_spatial)")
    box(0.40, "#F8E6C4", "fma_entity", f"{len(fma.entity_ids)} nodes · x_u ∈ R^{fma.x.shape[1]}\nlabel + definition → PubMedBERT")
    schema_axis.add_patch(FancyArrowPatch((0.50, 0.71), (0.50, 0.57), arrowstyle="-|>", mutation_scale=12, color="#0072B2"))
    schema_axis.add_patch(FancyArrowPatch((0.46, 0.57), (0.46, 0.71), arrowstyle="-|>", mutation_scale=12, color="#0072B2"))
    schema_axis.text(0.54, 0.64, "maps_to / describes", fontsize=8, va="center")
    schema_axis.text(0.50, 0.30, "FMA hierarchy", ha="center", fontsize=9, weight="bold")
    schema_axis.text(0.50, 0.25, "is_a ↔ has_subclass\npart_of ↔ has_part", ha="center", fontsize=8)
    schema_axis.add_patch(FancyArrowPatch((0.50, 0.38), (0.50, 0.33), arrowstyle="-|>", mutation_scale=12, color="#777777"))
    schema_axis.text(0.50, 0.12, "type-specific projection\n→ shared hidden dimension", ha="center", fontsize=8,
                     bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F0F0F0", "edgecolor": "#999999"})

    figure.suptitle("FMA anatomical knowledge graph used by the heterogeneous model", fontsize=16, y=0.995)
    figure.text(
        0.5, 0.015,
        f"Forward edges shown in the graph; inverse relation types are included by the model. "
        f"Blue lines connect MRI ROI coordinates directly to mapped FMA entities; "
        f"mappings: {len(mapping)} active ROI links · is_a: {len(_edge_array(fma.edges['is_a']))} · "
        f"part_of: {len(_edge_array(fma.edges['part_of']))}.",
        ha="center", fontsize=8.5,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _default_roi_ids(checkpoint: dict) -> np.ndarray:
    coords_path = PROJECT_ROOT / checkpoint["parameters"]["paths"]["coordinates"]
    coordinates = pd.read_csv(coords_path)
    features = pd.read_csv(PROJECT_ROOT / "data" / "04_final" / "all_features.csv", nrows=1)
    available = {
        int(column.rsplit("_", 1)[-1])
        for column in features.columns
        if column.startswith("GM_vol_Region_")
    }
    return coordinates.loc[coordinates["label_id"].isin(available), "label_id"].sort_values().to_numpy(dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Generate graph-construction methodology figures")
    parser.add_argument("checkpoint", type=Path, help="Saved checkpoint used to reconstruct fold-safe graphs")
    parser.add_argument("--sample-index", type=int, default=0, help="Held-out-fold sample used for the patient overlays")
    parser.add_argument(
        "--mri-root", type=Path,
        default=PROJECT_ROOT / "data" / "02_processed" / "nifti",
        help="Directory containing subject MPRAGE NIfTI files",
    )
    parser.add_argument(
        "--segmentation-root", type=Path,
        default=PROJECT_ROOT / "data" / "02_processed" / "segmentation",
        help="Directory containing SPM forward deformation fields",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures" / "methodology")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    content = _load_methodology_record(args.checkpoint, args.sample_index)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = output_dir / "spatial_vs_codeviation.png"
    fma_path = output_dir / "fma_anatomical_knowledge_graph.png"
    plot_spatial_vs_codeviation(
        args.checkpoint, spatial_path, args.sample_index,
        mri_root=args.mri_root, segmentation_root=args.segmentation_root,
        content=content,
    )
    plot_fma_knowledge_graph(
        args.checkpoint, fma_path, _default_roi_ids(checkpoint),
        sample_content=content,
        mri_root=args.mri_root, segmentation_root=args.segmentation_root,
    )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sample_index": args.sample_index,
        "record_id": content["record_id"],
        "class_name": content["class_name"],
        "mri_root": str(args.mri_root),
        "segmentation_root": str(args.segmentation_root),
        "spatial_vs_codeviation": str(spatial_path),
        "fma_anatomical_knowledge_graph": str(fma_path),
    }
    (output_dir / "methodology_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(spatial_path)
    print(fma_path)
    print(output_dir / "methodology_figure_manifest.json")


if __name__ == "__main__":
    main()
