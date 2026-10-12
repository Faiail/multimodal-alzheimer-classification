"""Publication figures for paper-style graph-model explanations."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import resample_to_img
from scipy.ndimage import map_coordinates
from .fma_figures import _attention_relation_from_key, _edge_relation_from_key


MODALITY_NAMES = ('GM', 'WM', 'CSF', 'PET')
CLASS_NAMES = ('CN', 'MCI', 'AD')


def normalized_modality_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)[:4]
    denominator = values.sum()
    return values / denominator if denominator > 0 else np.zeros_like(values)


def plot_modality_importance(
    records: list[dict],
    output_path: str | Path,
    title: str = 'Imaging-modality importance by diagnostic class',
):
    """Plot mean normalized IG importance with 95% bootstrap intervals."""
    rng = np.random.default_rng(42)
    means, errors, counts = [], [], []
    for class_index in range(3):
        values = np.asarray([
            normalized_modality_importance(record['modality_importance'])
            for record in records if record['true_label'] == class_index
        ])
        counts.append(len(values))
        if len(values) == 0:
            means.append(np.full(4, np.nan))
            errors.append(np.zeros((2, 4)))
            continue
        mean = values.mean(axis=0)
        if len(values) == 1:
            low = high = mean
        else:
            samples = rng.choice(len(values), size=(2000, len(values)), replace=True)
            boot = values[samples].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        means.append(mean)
        errors.append(np.stack((mean - low, high - mean)))

    means = np.asarray(means)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x = np.arange(4)
    width = 0.24
    colours = ('#4C78A8', '#F58518', '#54A24B')
    for class_index, (name, colour) in enumerate(zip(CLASS_NAMES, colours)):
        axis.bar(
            x + (class_index - 1) * width,
            means[class_index],
            width,
            yerr=errors[class_index],
            capsize=3,
            color=colour,
            label=f'{name} (n={counts[class_index]})',
        )
    axis.set_xticks(x, MODALITY_NAMES)
    axis.set_ylabel('Fraction of absolute imaging attribution')
    maximum = float(np.nanmax(means))
    upper_limit = 0.2 * (math.floor(maximum / 0.2 + 1e-12) + 1)
    axis.set_ylim(0, max(0.2, upper_limit))
    axis.set_title(title)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def atlas_importance_image(
    atlas_path: str | Path,
    label_ids: np.ndarray,
    importance: np.ndarray,
):
    atlas = nib.load(str(atlas_path))
    labels = np.asarray(atlas.dataobj)
    volume = np.zeros(labels.shape, dtype=np.float32)
    scores = np.asarray(importance, dtype=float).reshape(-1)
    maximum = scores.max(initial=0.0)
    if maximum > 0:
        scores = scores / maximum
    for label, score in zip(label_ids, scores):
        volume[labels == int(label)] = float(score)
    return nib.Nifti1Image(volume, atlas.affine)


def attention_adjacency(
    arrays: dict[str, np.ndarray],
    n_regions: int,
    percentile: float = 98.0,
) -> np.ndarray:
    """Aggregate brain-to-brain attention over captured layers and heads."""
    dense = []
    sparse = {}
    for key, value in arrays.items():
        if not key.startswith('attention__'):
            continue
        if key.endswith('__edge_index'):
            relation = _attention_relation_from_key(key)
            if relation is not None and (relation[0] != 'brain_region' or relation[2] != 'brain_region'):
                continue
            sparse[key.removesuffix('__edge_index')] = [value, None]
        elif key.endswith('__coefficients'):
            relation = _attention_relation_from_key(key)
            if relation is not None and (relation[0] != 'brain_region' or relation[2] != 'brain_region'):
                continue
            stem = key.removesuffix('__coefficients')
            sparse.setdefault(stem, [None, None])[1] = value
        elif value.ndim == 3 and value.shape[-2:] == (n_regions, n_regions):
            dense.append(value[0])
    matrices = list(dense)
    for edge_index, coefficients in sparse.values():
        if edge_index is None or coefficients is None:
            continue
        edge_index = np.asarray(edge_index)
        coefficients = np.asarray(coefficients).reshape(-1)
        valid = (
            (edge_index[0] < n_regions) & (edge_index[1] < n_regions)
        )
        if not valid.any():
            continue
        matrix = np.zeros((n_regions, n_regions), dtype=float)
        matrix[edge_index[0, valid], edge_index[1, valid]] = coefficients[valid]
        matrices.append(matrix)
    if not matrices:
        return np.zeros((n_regions, n_regions), dtype=float)
    adjacency = np.mean(matrices, axis=0)
    np.fill_diagonal(adjacency, 0.0)
    positive = adjacency[adjacency > 0]
    if positive.size:
        adjacency[adjacency < np.percentile(positive, percentile)] = 0.0
    return np.maximum(adjacency, adjacency.T)


def attributed_edge_adjacency(
    arrays: dict[str, np.ndarray],
    n_regions: int,
    percentile: float = 98.0,
) -> np.ndarray:
    """Combine absolute GCN edge IG values into a brain-region adjacency."""
    matrices = []
    dense = arrays.get('edge_attribution__dense')
    if dense is not None:
        matrix = np.abs(np.asarray(dense).squeeze())
        if matrix.shape == (n_regions, n_regions):
            matrices.append(matrix)
    for key, values in arrays.items():
        if not key.startswith('edge_attribution__') or key.endswith('__edge_index'):
            continue
        relation = _edge_relation_from_key(key)
        if relation is not None and (relation[0] != 'brain_region' or relation[2] != 'brain_region'):
            continue
        suffix = key.removeprefix('edge_attribution__')
        edge_key = f'edge_attribution__{suffix}__edge_index'
        if edge_key not in arrays:
            continue
        edge_index = np.asarray(arrays[edge_key])
        scores = np.abs(np.asarray(values).reshape(-1))
        valid = (edge_index[0] < n_regions) & (edge_index[1] < n_regions)
        matrix = np.zeros((n_regions, n_regions), dtype=float)
        matrix[edge_index[0, valid], edge_index[1, valid]] = scores[valid]
        matrices.append(matrix)
    if not matrices:
        return np.zeros((n_regions, n_regions), dtype=float)
    adjacency = np.mean(matrices, axis=0)
    np.fill_diagonal(adjacency, 0.0)
    positive = adjacency[adjacency > 0]
    if positive.size:
        adjacency[adjacency < np.percentile(positive, percentile)] = 0.0
    return np.maximum(adjacency, adjacency.T)


def find_patient_mri(row: dict, mri_root: str | Path | None) -> Path | None:
    if mri_root is None:
        return None
    root = Path(mri_root)
    identifiers = [
        row.get('mri_subject_id'), row.get('subject_id'), row.get('PTID')
    ]
    for identifier in identifiers:
        if identifier is None or str(identifier).lower() == 'nan':
            continue
        identifier = str(identifier)
        matches = [
            path for path in root.rglob(f'*{identifier}*.nii*')
            if 'MPRAGE' in path.name.upper()
        ]
        if matches:
            return sorted(matches, key=lambda path: (len(path.name), path.name))[0]
    return None


def find_deformation(
    mri_path: Path, segmentation_root: str | Path | None
) -> Path | None:
    if segmentation_root is None:
        return None
    acquisition = mri_path.name.removesuffix('.nii.gz').removesuffix('.nii')
    candidate = (
        Path(segmentation_root) / acquisition /
        f'y_{acquisition}_n4.nii'
    )
    return candidate if candidate.exists() else None


def warp_mri_to_mni(mri_path: Path, deformation_path: Path):
    """Apply an SPM forward deformation field to a native T1 volume."""
    native = nib.load(str(mri_path))
    native_data = np.asarray(native.dataobj, dtype=np.float32)
    if native_data.ndim > 3:
        native_data = native_data[..., 0]
    deformation = nib.load(str(deformation_path))
    world_coordinates = np.asarray(deformation.dataobj, dtype=np.float32).squeeze()
    if world_coordinates.shape[-1] != 3:
        raise ValueError(f'Unexpected deformation shape: {world_coordinates.shape}')
    flat_world = world_coordinates.reshape(-1, 3)
    homogeneous = np.column_stack(
        (flat_world, np.ones(len(flat_world), dtype=np.float32))
    )
    voxel_coordinates = (np.linalg.inv(native.affine) @ homogeneous.T)[:3]
    warped = map_coordinates(
        native_data, voxel_coordinates, order=1,
        mode='constant', cval=0.0, prefilter=False,
    ).reshape(world_coordinates.shape[:3])
    return nib.Nifti1Image(warped.astype(np.float32), deformation.affine)


def _canonical_volume(source) -> tuple[np.ndarray, nib.Nifti1Image]:
    image_object = nib.as_closest_canonical(
        nib.load(str(source)) if not isinstance(source, nib.spatialimages.SpatialImage)
        else source
    )
    data = np.asarray(image_object.dataobj)
    if data.ndim > 3:
        data = data[..., 0]
    return np.nan_to_num(data.astype(np.float32)), image_object


def _central_axial_slice(source, world_z: float | None = None):
    data, image_object = _canonical_volume(source)
    if world_z is None:
        foreground = np.abs(data) > np.percentile(np.abs(data), 55)
        weights = foreground.sum(axis=(0, 1))
        slice_index = (
            int(np.average(np.arange(data.shape[2]), weights=weights))
            if weights.sum() else data.shape[2] // 2
        )
    else:
        inverse = np.linalg.inv(image_object.affine)
        slice_index = int(round((inverse @ [0, 0, world_z, 1])[2]))
        slice_index = int(np.clip(slice_index, 0, data.shape[2] - 1))
    return np.rot90(data[:, :, slice_index])


def _mni_axial_slice(source, world_z: float = 0.0):
    data, image_object = _canonical_volume(source)
    inverse = np.linalg.inv(image_object.affine)
    slice_index = int(round((inverse @ [0, 0, world_z, 1])[2]))
    slice_index = int(np.clip(slice_index, 0, data.shape[2] - 1))
    section = data[:, :, slice_index].T
    x0, y0 = nib.affines.apply_affine(
        image_object.affine, [-0.5, -0.5, slice_index]
    )[:2]
    x1, y1 = nib.affines.apply_affine(
        image_object.affine,
        [data.shape[0] - 0.5, data.shape[1] - 0.5, slice_index],
    )[:2]
    return section, (x0, x1, y0, y1)


def _centred_square(arrays, mask):
    """Crop arrays around one foreground mask and pad to a centred square."""
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return arrays
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    height, width = upper - lower
    margin = max(3, int(round(0.06 * max(height, width))))
    y0 = max(0, lower[0] - margin)
    y1 = min(mask.shape[0], upper[0] + margin)
    x0 = max(0, lower[1] - margin)
    x1 = min(mask.shape[1], upper[1] + margin)
    cropped = [array[y0:y1, x0:x1] for array in arrays]
    height, width = cropped[0].shape
    side = max(height, width)
    pad_y = (side - height)
    pad_x = (side - width)
    padding = (
        (pad_y // 2, pad_y - pad_y // 2),
        (pad_x // 2, pad_x - pad_x // 2),
    )
    return [np.pad(array, padding) for array in cropped]


def _show_anatomical(axis, source):
    section, extent = _mni_axial_slice(source)
    nonzero = section[np.isfinite(section) & (section != 0)]
    low, high = (
        np.percentile(nonzero, [1, 99.5]) if nonzero.size else (0.0, 1.0)
    )
    axis.imshow(
        section, cmap='gray', vmin=low, vmax=high,
        origin='lower', extent=extent,
    )
    axis.set_facecolor('black')
    axis.set_aspect('equal', adjustable='box')
    axis.axis('off')


def _show_atlas(axis, background_image, atlas_path):
    atlas_image = resample_to_img(
        str(atlas_path), background_image, interpolation='nearest',
        force_resample=True, copy_header=True,
    )
    background, extent = _mni_axial_slice(background_image)
    atlas, _ = _mni_axial_slice(atlas_image)
    nonzero = background[background != 0]
    low, high = np.percentile(nonzero, [1, 99.5])
    axis.imshow(
        background, cmap='gray', vmin=low, vmax=high,
        origin='lower', extent=extent,
    )
    masked = np.ma.masked_where(atlas == 0, atlas)
    axis.imshow(
        masked, cmap='nipy_spectral', alpha=0.42, origin='lower',
        interpolation='nearest', extent=extent,
    )
    axis.set_facecolor('black')
    axis.set_aspect('equal', adjustable='box')
    axis.axis('off')


def _show_importance(axis, background_image, importance_image):
    importance_image = resample_to_img(
        importance_image, background_image, interpolation='nearest',
        force_resample=True, copy_header=True,
    )
    background, extent = _mni_axial_slice(background_image)
    importance, _ = _mni_axial_slice(importance_image)
    nonzero = background[background != 0]
    low, high = np.percentile(nonzero, [1, 99.5])
    axis.imshow(
        background, cmap='gray', vmin=low, vmax=high,
        origin='lower', extent=extent,
    )
    masked = np.ma.masked_where(importance <= 0, importance)
    axis.imshow(masked, cmap='turbo', vmin=0, vmax=1, alpha=0.88,
                origin='lower', interpolation='nearest', extent=extent)
    axis.set_facecolor('black')
    axis.set_aspect('equal', adjustable='box')
    axis.axis('off')


def _show_edge_graph(
    axis, adjacency, coordinates, node_values, background_image
):
    background, extent = _mni_axial_slice(background_image)
    nonzero = background[background != 0]
    low, high = (
        np.percentile(nonzero, [1, 99.5]) if nonzero.size else (0.0, 1.0)
    )
    axis.imshow(
        background, cmap='gray', vmin=low, vmax=high,
        origin='lower', extent=extent,
    )
    axis.set_facecolor('black')
    upper = np.transpose(np.nonzero(np.triu(adjacency, k=1)))
    # Scale line widths using the distribution of the edges that survived the
    # attribution threshold. Scaling by the global maximum alone can make
    # almost every line appear hairline-thin when one edge is a strong
    # outlier. A small square-root stretch preserves the ranking while making
    # weaker, but still selected, edges visible.
    if upper.size:
        retained = adjacency[upper[:, 0], upper[:, 1]].astype(float)
        lower = float(np.percentile(retained, 5.0))
        upper_bound = float(np.percentile(retained, 99.0))
        if upper_bound <= lower:
            lower = float(retained.min())
            upper_bound = float(retained.max())
    else:
        lower, upper_bound = 0.0, 1.0
    for source, target in upper:
        strength = float(adjacency[source, target])
        if upper_bound > lower:
            weight = (strength - lower) / (upper_bound - lower)
            weight = float(np.clip(weight, 0.0, 1.0))
        else:
            weight = 1.0
        weight = float(np.sqrt(weight))
        axis.plot(
            coordinates[[source, target], 0],
            coordinates[[source, target], 1],
            color=plt.cm.Blues(0.45 + 0.55 * weight),
            linewidth=1.0 + 3.5 * weight,
            alpha=0.45 + 0.5 * weight,
            zorder=1,
        )
    axis.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=node_values, cmap='turbo', vmin=0, vmax=1,
        s=8 + 34 * node_values, edgecolors='black', linewidths=0.25, zorder=2,
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect('equal', adjustable='box')
    axis.text(0.03, 0.96, 'L', transform=axis.transAxes, va='top')
    axis.text(0.97, 0.96, 'R', transform=axis.transAxes, ha='right', va='top')
    axis.axis('off')


def plot_patient_comparison(
    patients: list[dict],
    atlas_path: str | Path,
    coordinates: np.ndarray,
    output_path: str | Path,
    mri_root: str | Path | None = None,
    segmentation_root: str | Path | None = None,
    edge_percentile: float = 98.0,
):
    """Create MRI, atlas, regional-IG, and GAT-attention panels."""
    if len(patients) not in (1, 3):
        raise ValueError('patients must contain one record or one CN, one MCI, and one AD record')
    row_count = len(patients)
    template = datasets.load_mni152_template(resolution=1)
    figure, axes = plt.subplots(
        row_count, 4, figsize=(16, 4.5 if row_count == 1 else 10.5),
        gridspec_kw={'width_ratios': (1, 1, 1, 1)},
    )
    axes = np.atleast_2d(axes)
    has_attention = any(
        any(key.startswith('attention__') for key in patient['arrays'])
        for patient in patients
    )
    has_gcn_edges = any(
        any(key.startswith('edge_attribution__') for key in patient['arrays'])
        for patient in patients
    )
    edge_title = (
        'GAT edge attention' if has_attention else
        'GCN edge attribution' if has_gcn_edges else
        'Regional importance'
    )
    column_titles = (
        'T1-weighted MRI', 'AAL3 atlas', 'Regional IG importance',
        edge_title,
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=12, pad=8)

    for row_index, patient in enumerate(patients):
        record = patient['record']
        arrays = patient['arrays']
        raw_row = patient.get('raw_row', {})
        mri_path = find_patient_mri(raw_row, mri_root)
        deformation_path = (
            find_deformation(mri_path, segmentation_root) if mri_path else None
        )
        if mri_path and deformation_path:
            background = warp_mri_to_mni(mri_path, deformation_path)
            mri_label = 'patient T1 MRI, normalized to MNI'
        else:
            background = template
            mri_label = 'MNI template (normalized patient MRI unavailable)'
        _show_anatomical(axes[row_index, 0], background)
        axes[row_index, 0].text(
            0.02, 0.03, mri_label, transform=axes[row_index, 0].transAxes,
            color='white', fontsize=7,
        )

        _show_atlas(axes[row_index, 1], background, atlas_path)

        importance_img = atlas_importance_image(
            atlas_path, arrays['roi_label_ids'],
            arrays['region_importance'][0],
        )
        _show_importance(axes[row_index, 2], background, importance_img)

        if any(key.startswith('attention__') for key in arrays):
            adjacency = attention_adjacency(
                arrays, len(arrays['roi_label_ids']), edge_percentile
            )
        else:
            adjacency = attributed_edge_adjacency(
                arrays, len(arrays['roi_label_ids']), edge_percentile
            )
        node_values = arrays['region_importance'][0].astype(float)
        maximum = node_values.max(initial=0.0)
        node_values = node_values / maximum if maximum > 0 else node_values
        _show_edge_graph(
            axes[row_index, 3], adjacency, coordinates, node_values, background
        )
        if not np.count_nonzero(adjacency):
            axes[row_index, 3].text(
                0.5, 0.04, 'No edge attribution: nodes show IG importance',
                ha='center', transform=axes[row_index, 3].transAxes, fontsize=7,
            )
        class_name = CLASS_NAMES[int(record['true_label'])]
        label_y = (
            0.82 - row_index * 0.31 if row_count == 3 else 0.5
        )
        figure.text(
            0.01, label_y,
            f"{class_name}\n{record['record_id']}\nPred: {record['predicted_class']}",
            va='center', fontsize=10,
        )
    figure.suptitle(
        'Patient-level multimodal graph explanations', fontsize=15, y=0.99
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.045, top=0.85,
        wspace=0.10, hspace=0.10,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def load_explanation(path: str | Path) -> dict:
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    arrays = dict(np.load(path, allow_pickle=False))
    return {'record': metadata, 'arrays': arrays}
