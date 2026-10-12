"""Generate a larger, one-row version of the patient comparison figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from visualization.explanation_figures import load_explanation, plot_patient_comparison


CLASS_NAMES = ('CN', 'MCI', 'AD')


def _load_entries(explanation_dir: Path) -> list[dict]:
    entries = []
    for npz_path in sorted(explanation_dir.glob('*.npz')):
        json_path = npz_path.with_suffix('.json')
        if json_path.exists():
            entries.append(load_explanation(npz_path))
    if not entries:
        raise FileNotFoundError(
            f'No paired .json/.npz explanations found in {explanation_dir}'
        )
    return entries


def _select_entry(entries: list[dict], class_name: str, record_id: str | None):
    candidates = [
        entry for entry in entries
        if entry['record']['true_class'] == class_name
    ]
    if record_id is not None:
        candidates = [
            entry for entry in candidates
            if str(entry['record']['record_id']) == record_id
        ]
        if not candidates:
            raise ValueError(
                f'No {class_name} explanation found for record ID {record_id}'
            )
    if not candidates:
        raise ValueError(f'No {class_name} explanation is available')
    candidates.sort(key=lambda entry: (
        entry['record']['predicted_class'] != class_name,
        abs(entry['record']['convergence_delta']),
    ))
    chosen = candidates[0]
    chosen['record']['true_label'] = CLASS_NAMES.index(class_name)
    chosen['raw_row'] = {
        'PTID': chosen['record']['record_id'],
        'subject_id': chosen['record'].get('subject_id'),
        'mri_subject_id': chosen['record'].get('mri_subject_id'),
    }
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description='Generate a large one-row patient comparison figure.'
    )
    parser.add_argument(
        'explanation_dir', type=Path,
        help='Directory containing paired explainer .json and .npz files',
    )
    parser.add_argument('--class-name', choices=CLASS_NAMES, required=True)
    parser.add_argument(
        '--record-id',
        help='Exact record ID; otherwise choose a correctly classified representative',
    )
    parser.add_argument('--mri-root', type=Path)
    parser.add_argument(
        '--segmentation-root', type=Path,
        default=PROJECT_ROOT / 'data' / '02_processed' / 'segmentation',
    )
    parser.add_argument('--edge-percentile', type=float, default=98.0)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'reports' / 'figures' /
        'explanations' / 'patient_comparison_single.png',
    )
    args = parser.parse_args()

    entry = _select_entry(
        _load_entries(args.explanation_dir), args.class_name, args.record_id
    )
    coordinates_frame = pd.read_csv(
        PROJECT_ROOT / 'data' / '03_features' / 'aal3_coordinates.csv'
    )
    label_order = entry['arrays']['roi_label_ids'].astype(int)
    coordinates = (
        coordinates_frame.set_index('label_id')
        .loc[label_order, ['x', 'y', 'z']].to_numpy()
    )
    plot_patient_comparison(
        [entry],
        PROJECT_ROOT / 'configs' / 'AAL3' / 'ROI_MNI_V7_1mm.nii',
        coordinates,
        args.output,
        mri_root=args.mri_root,
        segmentation_root=args.segmentation_root,
        edge_percentile=args.edge_percentile,
    )
    print(args.output)


if __name__ == '__main__':
    main()
