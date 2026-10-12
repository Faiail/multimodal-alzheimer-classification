"""Plot graph and clinical Integrated Gradients from exported explanations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


CLASS_NAMES = ("CN", "MCI", "AD")
MODALITY_NAMES = ("GM", "WM", "CSF", "PET")


def _class_index(metadata: dict) -> int:
    if "true_label" in metadata:
        return int(metadata["true_label"])
    return CLASS_NAMES.index(metadata["true_class"])


def _feature_labels(arrays: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    graph_labels = list(MODALITY_NAMES)
    clinical_labels = [
        str(value)
        for value in np.asarray(arrays["clinical_feature_names"]).reshape(-1)
    ]
    return graph_labels, clinical_labels


def _read_explanations(explanation_dir: Path):
    records = []
    for npz_path in sorted(explanation_dir.glob("*.npz")):
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            continue
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        arrays = dict(np.load(npz_path, allow_pickle=False))
        required = {"brain_attribution", "clinical_attribution", "clinical_feature_names", "roi_label_ids"}
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"{npz_path.name} is missing fields: {sorted(missing)}")
        records.append((metadata, arrays))
    if not records:
        raise FileNotFoundError(f"No paired explanations found in {explanation_dir}")
    return records


def _aggregate(records):
    graph_labels, clinical_labels = _feature_labels(records[0][1])
    labels = graph_labels + clinical_labels
    categories = ["graph"] * len(graph_labels) + ["clinical"] * len(clinical_labels)
    values = {class_name: [] for class_name in CLASS_NAMES}

    for metadata, arrays in records:
        graph = np.asarray(arrays["brain_attribution"], dtype=float).squeeze(0)
        clinical = np.asarray(arrays["clinical_attribution"], dtype=float).reshape(-1)
        if graph.ndim != 2 or graph.shape[-1] != len(graph_labels):
            raise ValueError("brain_attribution does not match modality metadata")
        if clinical.size != len(clinical_labels):
            raise ValueError("clinical_attribution does not match clinical feature metadata")
        graph_importance = np.abs(graph).sum(axis=0)
        attribution = np.concatenate((graph_importance, np.abs(clinical)))
        values[CLASS_NAMES[_class_index(metadata)]].append(attribution)

    class_means = {}
    for class_name, rows in values.items():
        if not rows:
            class_means[class_name] = np.full(len(labels), np.nan)
        else:
            class_means[class_name] = np.asarray(rows).mean(axis=0)
    matrix = np.column_stack([class_means[class_name] for class_name in CLASS_NAMES])
    overall = np.nanmean(matrix, axis=1)
    table = pd.DataFrame(matrix, columns=CLASS_NAMES)
    table.insert(0, "feature", labels)
    table.insert(1, "category", categories)
    table["overall_mean"] = overall
    table = table.sort_values("overall_mean", ascending=False, ignore_index=True)
    return table, {name: len(rows) for name, rows in values.items()}


def _plot_heatmap(table: pd.DataFrame, output_path: Path):
    values = table.loc[:, CLASS_NAMES].to_numpy(dtype=float)
    height = max(10.0, 0.18 * len(table) + 2.5)
    figure, axis = plt.subplots(figsize=(10, height))
    image = axis.imshow(values, aspect="auto", cmap="magma", interpolation="nearest")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    tick_step = max(1, len(table) // 120)
    tick_indices = np.arange(0, len(table), tick_step)
    axis.set_yticks(tick_indices, table.loc[tick_indices, "feature"], fontsize=6)
    axis.set_xlabel("True diagnostic class")
    axis.set_ylabel("Feature, sorted by mean absolute attribution")
    axis.set_title("All graph and clinical feature importances")
    figure.colorbar(image, ax=axis, label="Mean absolute Integrated Gradients")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_top_features(table: pd.DataFrame, output_path: Path, top_n: int):
    top = table.head(top_n).iloc[::-1]
    y = np.arange(len(top))
    figure, axis = plt.subplots(figsize=(12, max(8, 0.28 * len(top) + 2)))
    width = 0.25
    colours = ("#4C78A8", "#F58518", "#54A24B")
    for index, (class_name, colour) in enumerate(zip(CLASS_NAMES, colours)):
        axis.barh(
            y + (index - 1) * width,
            top[class_name],
            width,
            color=colour,
            label=class_name,
        )
    axis.set_yticks(y, [f"{row.feature} [{row.category}]" for row in top.itertuples()])
    axis.set_xlabel("Mean absolute Integrated Gradients")
    shown = min(top_n, len(table))
    axis.set_title(f"Graph and clinical feature importance ({shown} features)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("explanation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/explanations"))
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")

    checkpoint_path = args.checkpoint
    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        checkpoint_path = candidates[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameters = checkpoint.get("parameters", {})
    model_identity = parameters.get("identity", {})
    if model_identity.get("modality") not in (None, "multimodal"):
        raise ValueError("The selected checkpoint is not multimodal")

    records = _read_explanations(args.explanation_dir)
    table, counts = _aggregate(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_feature_importance.csv"
    heatmap_path = args.output_dir / "all_feature_importance_heatmap.png"
    top_path = args.output_dir / "all_feature_importance.png"
    table.to_csv(csv_path, index=False)
    _plot_heatmap(table, heatmap_path)
    _plot_top_features(table, top_path, args.top_n)
    print(f"Explanations: {len(records)}; classes: {counts}")
    print(csv_path)
    print(heatmap_path)
    print(top_path)


if __name__ == "__main__":
    main()