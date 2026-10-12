"""Generate the thesis explanation figures from exported .json/.npz files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from visualization.explanation_figures import (
    load_explanation,
    plot_modality_importance,
    plot_patient_comparison,
)
from visualization.fma_figures import (
    has_fma_edge_explanations,
    plot_fma_patient_comparison,
    plot_fma_relation_importance,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'explanation_dir', type=Path,
        help='Directory containing paired explainer .json and .npz files',
    )
    parser.add_argument(
        '--scope', choices=('all', 'subset'), default='all',
        help='Use all explanations or a reproducible class-balanced subset',
    )
    parser.add_argument('--samples-per-class', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mri-root', type=Path)
    parser.add_argument(
        '--segmentation-root', type=Path,
        default=PROJECT_ROOT / 'data' / '02_processed' / 'segmentation',
    )
    parser.add_argument('--edge-percentile', type=float, default=98.0)
    parser.add_argument(
        '--fma-edge-percentile', type=float, default=75.0,
        help='Percentile used only for the patient-level FMA graph (default: 75)',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=PROJECT_ROOT / 'reports' / 'figures' / 'explanations',
    )
    args = parser.parse_args()

    entries = []
    for npz_path in sorted(args.explanation_dir.glob('*.npz')):
        json_path = npz_path.with_suffix('.json')
        if not json_path.exists():
            continue
        entry = load_explanation(npz_path)
        metadata = entry['record']
        metadata['true_label'] = ('CN', 'MCI', 'AD').index(metadata['true_class'])
        metadata['modality_importance'] = entry['arrays']['modality_importance'][0]
        entries.append(entry)
    if not entries:
        raise FileNotFoundError(
            f'No paired .json/.npz explanations found in {args.explanation_dir}'
        )

    rng = np.random.default_rng(args.seed)
    selected = []
    for class_index in range(3):
        group = [entry for entry in entries if entry['record']['true_label'] == class_index]
        if args.scope == 'subset' and len(group) > args.samples_per_class:
            indices = rng.choice(len(group), args.samples_per_class, replace=False)
            group = [group[index] for index in indices]
        selected.extend(group)
    records = [entry['record'] for entry in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_modality_importance(
        records, args.output_dir / f'modality_importance_{args.scope}.png'
    )

    # Prefer correctly classified examples with low numerical integration error.
    representatives = []
    for class_index, class_name in enumerate(('CN', 'MCI', 'AD')):
        candidates = [
            entry for entry in selected
            if entry['record']['true_label'] == class_index
        ]
        candidates.sort(key=lambda entry: (
            entry['record']['predicted_class'] != class_name,
            abs(entry['record']['convergence_delta']),
        ))
        if not candidates:
            raise ValueError(f'No {class_name} explanation is available')
        chosen = candidates[0]
        chosen['raw_row'] = {
            'PTID': chosen['record']['record_id'],
            'subject_id': chosen['record'].get('subject_id'),
            'mri_subject_id': chosen['record'].get('mri_subject_id'),
        }
        representatives.append(chosen)

    coords_frame = pd.read_csv(
        PROJECT_ROOT / 'data' / '03_features' / 'aal3_coordinates.csv'
    )
    label_order = representatives[0]['arrays']['roi_label_ids'].astype(int)
    coordinates = (
        coords_frame.set_index('label_id')
        .loc[label_order, ['x', 'y', 'z']].to_numpy()
    )
    plot_patient_comparison(
        representatives,
        PROJECT_ROOT / 'configs' / 'AAL3' / 'ROI_MNI_V7_1mm.nii',
        coordinates,
        args.output_dir / 'patient_comparison_CN_MCI_AD.png',
        mri_root=args.mri_root,
        segmentation_root=args.segmentation_root,
        edge_percentile=args.edge_percentile,
    )

    fma_entries = [entry for entry in selected if has_fma_edge_explanations(entry)]
    fma_outputs = {}
    if fma_entries and all(has_fma_edge_explanations(entry) for entry in representatives):
        fma_relation_path = args.output_dir / f'fma_relation_importance_{args.scope}.png'
        fma_patient_path = args.output_dir / 'fma_patient_comparison_CN_MCI_AD.png'
        fma_patient_vertical_path = args.output_dir / 'fma_patient_comparison_CN_MCI_AD_vertical.png'
        fma_patient_two_rows_path = args.output_dir / 'fma_patient_comparison_CN_MCI_AD_two_rows.png'
        fma_patient_percentile = args.fma_edge_percentile
        plot_fma_relation_importance(fma_entries, fma_relation_path, args.edge_percentile)
        plot_fma_patient_comparison(
            representatives, fma_patient_path, fma_patient_percentile, layout='horizontal'
        )
        plot_fma_patient_comparison(
            representatives, fma_patient_vertical_path, fma_patient_percentile, layout='vertical'
        )
        plot_fma_patient_comparison(
            representatives, fma_patient_two_rows_path, fma_patient_percentile, layout='two_rows'
        )
        fma_outputs = {
            'relation_importance': str(fma_relation_path),
            'patient_comparison': str(fma_patient_path),
            'patient_comparison_vertical': str(fma_patient_vertical_path),
            'patient_comparison_two_rows': str(fma_patient_two_rows_path),
        }
    elif fma_entries:
        print('FMA edge explanations are incomplete for the selected representatives; skipping FMA patient graph')
    manifest = {
        'scope': args.scope,
        'samples_per_class': args.samples_per_class,
        'number_of_explanations': len(selected),
        'representatives': [
            entry['record']['record_id'] for entry in representatives
        ],
        'mri_root': str(args.mri_root) if args.mri_root else None,
        'segmentation_root': str(args.segmentation_root),
        'edge_percentile': args.edge_percentile,
        'fma_patient_edge_percentile': args.fma_edge_percentile,
        'fma_outputs': fma_outputs,
    }
    (args.output_dir / 'figure_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    print(args.output_dir / f'modality_importance_{args.scope}.png')
    print(args.output_dir / 'patient_comparison_CN_MCI_AD.png')
    for path in fma_outputs.values():
        print(path)
    print(args.output_dir / 'figure_manifest.json')


if __name__ == '__main__':
    main()
