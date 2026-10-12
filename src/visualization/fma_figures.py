"""Figures for typed FMA relation explanations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D


FMA_RELATIONS = (
    "maps_to",
    "describes",
    "is_a",
    "has_subclass",
    "part_of",
    "has_part",
)
FMA_RELATION_COLORS = {
    "maps_to": "#1f77b4",
    "describes": "#6baed6",
    "is_a": "#d62728",
    "has_subclass": "#ff9896",
    "part_of": "#2ca02c",
    "has_part": "#98df8a",
}
CLASS_NAMES = ("CN", "MCI", "AD")


def _class_index(record: dict) -> int:
    if "true_label" in record:
        return int(record["true_label"])
    return CLASS_NAMES.index(record["true_class"])


def _edge_relation_from_key(key: str):
    if not key.startswith("edge_attribution__"):
        return None
    suffix = key.removeprefix("edge_attribution__")
    if suffix.endswith("__edge_index"):
        return None
    parts = suffix.split("__")
    return tuple(parts) if len(parts) == 3 else None


def _attention_relation_from_key(key: str):
    if not key.startswith("attention__"):
        return None
    stem = key.removeprefix("attention__")
    if "__head" in stem:
        stem = stem.split("__head", 1)[0]
    parts = stem.split("__")
    node_types = {"brain_region", "fma_entity"}
    for index in range(len(parts) - 2):
        candidate = tuple(parts[index:index + 3])
        if candidate[0] in node_types and candidate[2] in node_types:
            return candidate
    return None


def _is_fma_relation(relation) -> bool:
    return relation is not None and "fma_entity" in (relation[0], relation[2])


def _relation_node_limit(relation, n_regions: int, n_fma: int, side: int) -> int:
    return n_fma if relation[side] == "fma_entity" else n_regions


def collect_fma_edges(arrays: dict[str, np.ndarray]):
    """Return typed FMA edges as ``relation -> (source, target, score)``.

    Repeated GAT layer/head coefficients are averaged per typed edge. GCN edge
    Integrated-Gradients values are already one vector per relation.
    """
    n_regions = int(np.asarray(arrays.get("roi_label_ids", [])).size)
    n_fma = int(np.asarray(arrays.get("fma_entity_ids", [])).size)
    if n_fma == 0:
        n_fma = int(np.asarray(arrays.get("fma_entity_labels", [])).size)
    values_by_relation = defaultdict(lambda: defaultdict(list))
    kind = None

    for key, values in arrays.items():
        if not key.startswith("edge_attribution__") or key.endswith("__edge_index"):
            continue
        relation = _edge_relation_from_key(key)
        if not _is_fma_relation(relation):
            continue
        edge_key = f"{key}__edge_index"
        if edge_key not in arrays:
            continue
        kind = "GCN edge Integrated Gradients"
        edge_index = np.asarray(arrays[edge_key])
        scores = np.abs(np.asarray(values).reshape(-1))
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            continue
        count = min(edge_index.shape[1], scores.size)
        source_limit = _relation_node_limit(relation, n_regions, n_fma, 0)
        target_limit = _relation_node_limit(relation, n_regions, n_fma, 2)
        for source, target, score in zip(
            edge_index[0, :count], edge_index[1, :count], scores[:count]
        ):
            source, target = int(source), int(target)
            if source < 0 or target < 0:
                continue
            if source >= source_limit or target >= target_limit:
                continue
            values_by_relation[relation][(source, target)].append(float(score))

    if kind is None:
        for key, values in arrays.items():
            if not key.startswith("attention__") or not key.endswith("__coefficients"):
                continue
            relation = _attention_relation_from_key(key)
            if not _is_fma_relation(relation):
                continue
            edge_key = f"{key.removesuffix('__coefficients')}__edge_index"
            if edge_key not in arrays:
                continue
            kind = "GAT attention coefficients"
            edge_index = np.asarray(arrays[edge_key])
            scores = np.asarray(values).reshape(-1)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                continue
            count = min(edge_index.shape[1], scores.size)
            source_limit = _relation_node_limit(relation, n_regions, n_fma, 0)
            target_limit = _relation_node_limit(relation, n_regions, n_fma, 2)
            for source, target, score in zip(
                edge_index[0, :count], edge_index[1, :count], scores[:count]
            ):
                source, target = int(source), int(target)
                if source < 0 or target < 0:
                    continue
                if source >= source_limit or target >= target_limit:
                    continue
                values_by_relation[relation][(source, target)].append(float(score))

    edges = {
        relation: [
            (source, target, float(np.mean(scores)))
            for (source, target), scores in values.items()
        ]
        for relation, values in values_by_relation.items()
    }
    return edges, kind


def _relation_fractions(entry: dict):
    edges, kind = collect_fma_edges(entry["arrays"])
    masses = {
        relation: float(sum(score for _, _, score in values))
        for relation, values in edges.items()
    }
    total = sum(masses.values())
    if total <= 0:
        return {}, kind
    relation_masses = defaultdict(float)
    for relation, value in masses.items():
        relation_masses[relation[1]] += value
    return {relation: value / total for relation, value in relation_masses.items()}, kind


def plot_fma_relation_importance(
    records: list[dict], output_path: str | Path, percentile: float = 90.0
):
    """Plot class-level fractions of captured FMA relation importance."""
    relation_order = [
        relation for relation in FMA_RELATIONS
        if any(relation in _relation_fractions(entry)[0] for entry in records)
    ]
    if not relation_order:
        raise ValueError("No FMA edge explanations were found in the records")

    means, errors, counts = [], [], []
    kinds = []
    for class_index in range(3):
        rows = []
        for entry in records:
            if _class_index(entry["record"]) != class_index:
                continue
            fractions, kind = _relation_fractions(entry)
            if fractions:
                rows.append([fractions.get(relation, 0.0) for relation in relation_order])
                if kind:
                    kinds.append(kind)
        counts.append(len(rows))
        values = np.asarray(rows, dtype=float)
        if not len(values):
            means.append(np.full(len(relation_order), np.nan))
            errors.append(np.zeros((2, len(relation_order))))
            continue
        mean = values.mean(axis=0)
        if len(values) == 1:
            low = high = mean
        else:
            rng = np.random.default_rng(42 + class_index)
            sample_indices = rng.choice(
                len(values), size=(2000, len(values)), replace=True
            )
            bootstrap = values[sample_indices].mean(axis=1)
            low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        means.append(mean)
        errors.append(np.stack((mean - low, high - mean)))

    means = np.asarray(means)
    figure, axis = plt.subplots(figsize=(max(10, 1.4 * len(relation_order)), 5.8))
    x = np.arange(len(relation_order))
    width = 0.24
    colours = ("#4C78A8", "#F58518", "#54A24B")
    for class_index, (name, colour) in enumerate(zip(CLASS_NAMES, colours)):
        axis.bar(
            x + (class_index - 1) * width,
            means[class_index],
            width,
            yerr=errors[class_index],
            capsize=3,
            color=colour,
            label=f"{name} (n={counts[class_index]})",
        )
    labels = [relation.replace("_", " ") for relation in relation_order]
    axis.set_xticks(x, labels, rotation=25, ha="right")
    kind = kinds[0] if kinds else "captured FMA edge"
    axis.set_ylabel(f"Fraction of {kind.lower()}")
    maximum = float(np.nanmax(means))
    upper_limit = 0.2 * (np.floor(maximum / 0.2 + 1e-12) + 1)
    axis.set_ylim(0, max(0.2, upper_limit))
    axis.set_title(f"FMA relation importance by diagnostic class ({kind})")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _ontology_layout(n_fma: int, edges_by_relation: dict):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_fma))
    for relation, edges in edges_by_relation.items():
        if relation[0] == relation[2] == "fma_entity":
            graph.add_edges_from((source, target) for source, target, _ in edges)
    if graph.number_of_edges():
        positions = nx.spring_layout(
            graph, seed=42, iterations=250,
            k=1.8 / max(np.sqrt(max(n_fma, 1)), 1.0),
        )
    else:
        positions = nx.circular_layout(graph)
    coordinates = np.asarray([positions[index] for index in range(n_fma)])
    lower = coordinates.min(axis=0, keepdims=True)
    upper = coordinates.max(axis=0, keepdims=True)
    coordinates = (coordinates - lower) / np.maximum(upper - lower, 1e-12)
    return 0.08 + 0.84 * coordinates


def _retained_edges(edges_by_relation: dict, percentile: float):
    all_edges = [
        (relation, source, target, score)
        for relation, edges in edges_by_relation.items()
        for source, target, score in edges
        if score > 0 and relation[0] == relation[2] == "fma_entity"
    ]
    if not all_edges:
        return []
    scores = np.asarray([edge[3] for edge in all_edges])
    threshold = float(np.percentile(scores, percentile))
    return [edge for edge in all_edges if edge[3] >= threshold]


# The ontology contains inverse predicates for the same semantic relation.
# For the comparison view we collapse each inverse pair into one undirected
# semantic edge after applying the attribution threshold.
FMA_RELATION_FAMILIES = {
    "is_a": "taxonomy",
    "has_subclass": "taxonomy",
    "part_of": "composition",
    "has_part": "composition",
}
FMA_FAMILY_COLORS = {
    "taxonomy": "#c44e52",
    "composition": "#55a868",
}


def _retained_semantic_edges(edges_by_relation: dict, percentile: float):
    """Return thresholded FMA--FMA edges with inverse predicates merged."""
    retained = _retained_edges(edges_by_relation, percentile)
    grouped = defaultdict(list)
    for relation, source, target, score in retained:
        family = FMA_RELATION_FAMILIES.get(relation[1], relation[1])
        endpoints = tuple(sorted((int(source), int(target))))
        grouped[(family, endpoints)].append(float(score))
    return [
        (family, endpoints[0], endpoints[1], float(max(scores)))
        for (family, endpoints), scores in grouped.items()
    ]


def _comparison_layout(n_fma: int, retained_by_patient: list[list[tuple]]):
    """Create a compact common layout with explicit component packing.

    A spring layout of a disconnected graph can put a tiny component far away
    from the main ontology component.  Normalising those extrema makes the
    useful graph appear tiny.  We therefore allocate most of the panel to the
    largest component and pack any smaller components in a narrow side strip.
    """
    graph = nx.Graph()
    active = set()
    for edges in retained_by_patient:
        for _, source, target, _ in edges:
            active.update((source, target))
            graph.add_edge(source, target)
    if not active:
        return np.zeros((n_fma, 2), dtype=float), set()

    graph.add_nodes_from(active)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    coordinates = np.zeros((n_fma, 2), dtype=float)

    def normalise(values, x0, x1, y0, y1):
        values = np.asarray(values, dtype=float)
        lower = values.min(axis=0, keepdims=True)
        upper = values.max(axis=0, keepdims=True)
        values = (values - lower) / np.maximum(upper - lower, 1e-12)
        return np.column_stack((x0 + (x1 - x0) * values[:, 0],
                                y0 + (y1 - y0) * values[:, 1]))

    # The largest connected component receives almost the whole plotting area.
    largest = sorted(components[0])
    main_graph = graph.subgraph(largest).copy()
    main_positions = nx.spring_layout(
        main_graph, seed=42, iterations=600,
        k=1.5 / max(np.sqrt(len(largest)), 1.0), scale=1.0,
    )
    main_values = normalise(
        [main_positions[index] for index in largest], 0.04, 0.86, 0.07, 0.93
    )
    coordinates[np.asarray(largest, dtype=int)] = main_values

    # Smaller components are retained, but no longer determine the axis
    # limits.  They are placed in compact tiles on the right-hand side.
    secondary = components[1:]
    if secondary:
        columns = min(2, len(secondary))
        rows = int(np.ceil(len(secondary) / columns))
        for component_index, component in enumerate(secondary):
            nodes = sorted(component)
            subgraph = graph.subgraph(nodes).copy()
            positions = (nx.spring_layout(
                subgraph, seed=100 + component_index, iterations=250,
                k=1.2 / max(np.sqrt(len(nodes)), 1.0), scale=1.0,
            ) if len(nodes) > 1 else {nodes[0]: np.zeros(2)})
            column = component_index % columns
            row = component_index // columns
            x0 = 0.89 + column * (0.10 / columns)
            x1 = 0.89 + (column + 1) * (0.10 / columns) - 0.008
            y_top = 0.90 - row * (0.80 / rows)
            y_bottom = y_top - (0.65 / rows)
            values = normalise([positions[index] for index in nodes],
                               x0, x1, y_bottom, y_top)
            coordinates[np.asarray(nodes, dtype=int)] = values

    return coordinates, active


def _display_weights(edges):
    values = np.asarray([edge[3] for edge in edges], dtype=float)
    lower = float(np.percentile(values, 5))
    upper = float(np.percentile(values, 99))
    if upper <= lower:
        lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        return np.ones(len(values))
    return np.sqrt(np.clip((values - lower) / (upper - lower), 0.0, 1.0))


def plot_fma_patient_comparison(
    patients: list[dict],
    output_path: str | Path,
    percentile: float = 90.0,
    layout: str = "horizontal",
):
    """Plot compact, common-layout FMA ontology graphs for three patients.

    Only nodes incident to thresholded FMA--FMA edges are displayed.  The
    layout is computed once from the union of the three retained subgraphs,
    so the panels remain directly comparable.
    """
    if len(patients) != 3:
        raise ValueError("patients must contain one CN, one MCI, and one AD record")
    first_arrays = patients[0]["arrays"]
    labels = np.asarray(first_arrays.get("fma_entity_labels", []))
    n_fma = int(labels.size)
    if n_fma == 0:
        ids = np.asarray(first_arrays.get("fma_entity_ids", []))
        n_fma = int(ids.size)
        labels = np.asarray([f"FMA {index}" for index in range(n_fma)])
    if n_fma == 0:
        raise ValueError("FMA entity metadata is missing from the explanations")

    retained_by_patient = []
    kinds = []
    for patient in patients:
        edges_by_relation, kind = collect_fma_edges(patient["arrays"])
        retained_by_patient.append(
            _retained_semantic_edges(edges_by_relation, percentile)
        )
        if kind:
            kinds.append(kind)
    coordinates, active_union = _comparison_layout(n_fma, retained_by_patient)
    first_kind = kinds[0] if kinds else "FMA edge score"

    # Select a page-friendly arrangement.  ``two_rows`` uses a 2x4 grid so
    # the third panel occupies the middle two columns of the lower row.
    if layout == "horizontal":
        figure, axes_grid = plt.subplots(1, 3, figsize=(16, 6.8), squeeze=False)
        axes = list(axes_grid[0])
        figure.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.88,
                               wspace=0.04)
    elif layout == "vertical":
        figure, axes_grid = plt.subplots(3, 1, figsize=(7.0, 12), squeeze=False)
        axes = list(axes_grid[:, 0])
        figure.subplots_adjust(left=0.06, right=0.96, bottom=0.045, top=0.94,
                               hspace=0.14)
    elif layout == "two_rows":
        figure = plt.figure(figsize=(8.5, 7.8))
        grid = figure.add_gridspec(2, 4, hspace=0.22, wspace=0.12)
        axes = [
            figure.add_subplot(grid[0, :2]),
            figure.add_subplot(grid[0, 2:]),
            figure.add_subplot(grid[1, 1:3]),
        ]
        figure.subplots_adjust(left=0.035, right=0.965, bottom=0.06, top=0.92)
    else:
        raise ValueError("layout must be 'horizontal', 'vertical', or 'two_rows'")
    for panel_index, patient in enumerate(patients):
        retained = retained_by_patient[panel_index]
        axis = axes[panel_index]
        axis.set_facecolor("#fbfbfb")
        if retained:
            display_weights = _display_weights(retained)
            for edge, weight in zip(retained, display_weights):
                family, source, target, _ = edge
                axis.plot(
                    coordinates[[source, target], 0],
                    coordinates[[source, target], 1],
                    color=FMA_FAMILY_COLORS.get(family, "#555555"),
                    linewidth=0.7 + 3.0 * float(weight),
                    alpha=0.28 + 0.68 * float(weight),
                    zorder=1,
                )
            node_scores = np.zeros(n_fma, dtype=float)
            for _, source, target, score in retained:
                node_scores[source] += score
                node_scores[target] += score
            maximum = node_scores.max(initial=0.0)
            node_values = node_scores / maximum if maximum > 0 else node_scores
        else:
            node_values = np.zeros(n_fma, dtype=float)

        # Small grey context points preserve the common coordinate system;
        # only patient-active nodes receive colour and size from attribution.
        if active_union:
            context_nodes = np.asarray(sorted(active_union), dtype=int)
            axis.scatter(
                coordinates[context_nodes, 0], coordinates[context_nodes, 1],
                s=12, color="#d0d0d0", edgecolors="none", zorder=2,
            )
        patient_nodes = np.flatnonzero(node_values > 0)
        axis.scatter(
            coordinates[patient_nodes, 0], coordinates[patient_nodes, 1],
            c=node_values[patient_nodes], cmap="turbo", vmin=0, vmax=1,
            s=24 + 80 * node_values[patient_nodes], edgecolors="black",
            linewidths=0.35, zorder=3,
        )

        top_nodes = np.argsort(node_values)[::-1][:6]
        label_offsets = ((4, 4), (4, -10), (-4, 5), (-4, -11),
                         (6, 12), (-6, 13))
        for rank, node in enumerate(top_nodes):
            if node_values[node] <= 0:
                continue
            label = str(labels[node])
            if len(label) > 22:
                label = label[:19] + "..."
            axis.annotate(
                label, coordinates[node], xytext=label_offsets[rank],
                textcoords="offset points",
                fontsize=7.6 if layout != "horizontal" else 6.5, zorder=4,
                bbox={"facecolor": "white", "alpha": 0.72,
                      "edgecolor": "none", "pad": 0.25},
            )
        record = patient["record"]
        class_name = CLASS_NAMES[_class_index(record)]
        axis.set_title(
            f"{class_name} — {record.get('record_id', '')} · "
            f"{len(retained)} edges / {len(patient_nodes)} nodes",
            loc="left", fontsize=10.5 if layout != "horizontal" else 9.5,
        )
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        if not retained:
            axis.text(0.5, 0.5, "No FMA--FMA edges above threshold",
                      ha="center")

    handles = [
        Line2D([0], [0], color=FMA_FAMILY_COLORS[family], lw=2, label=label)
        for family, label in (("taxonomy", "is-a / has-subclass"),
                              ("composition", "part-of / has-part"))
        if any(edge[0] == family
               for edges in retained_by_patient for edge in edges)
    ]
    if handles:
        axes[0].legend(handles=handles, loc="upper right", frameon=True,
                       title="Ontology relation", fontsize=8.5 if layout != "horizontal" else 8)
    title_y = 0.985 if layout != "vertical" else 0.985
    figure.suptitle(
        f"Patient-level FMA ontology edge explanations ({first_kind})",
        fontsize=15, y=title_y,
    )
    figure.text(
        0.5, 0.015,
        f"Top {100 - percentile:.0f}% of positive FMA--FMA attributions shown; "
        "inverse predicates are merged. ROI-to-FMA mappings are included in "
        "the relation-importance chart.",
        ha="center", fontsize=8.5,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)



def has_fma_edge_explanations(entry: dict) -> bool:
    edges, _ = collect_fma_edges(entry["arrays"])
    return bool(edges)
