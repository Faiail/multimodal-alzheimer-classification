"""Batch Integrated Gradients over held-out folds without reloading per record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from explain import (attention_capture, collect_attention, explain_heterogeneous,
                     explain_heterogeneous_edges, explain_homogeneous,
                     explain_homogeneous_edges, summarize_brain_attributions)
from explain.checkpoint import (apply_normalizer_state, build_dataset, build_model,
                                held_out_frame, load_checkpoint, prepare_sample)
from train_multiclass import prepare_multiclass_data


def _cpu(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


def _select_indices(frame, per_class: int, seed: int) -> list[int]:
    if per_class == 0:
        return list(range(len(frame)))
    selected = []
    for label in range(3):
        indices = np.flatnonzero(frame['label'].to_numpy() == label)
        rng = np.random.default_rng(seed + label)
        count = min(per_class, len(indices))
        selected.extend(rng.choice(indices, count, replace=False).tolist())
    return sorted(selected)


def _save_record(
    output_dir: Path, checkpoint_path: Path, checkpoint: dict, dataset,
    frame, sample_index: int, result, brain_attribution, clinical_attribution,
    attention, edge_attribution=None, edge_indices=None,
):
    summary = summarize_brain_attributions(brain_attribution)
    row = frame.iloc[sample_index]
    names = checkpoint.get('class_names', ['CN', 'MCI', 'AD'])
    probabilities = torch.softmax(result.output, dim=1)[0]
    fold = int(checkpoint['parameters']['identity']['fold'])
    stem = f'{checkpoint_path.stem}__sample{sample_index}'
    metadata = {
        'checkpoint': str(checkpoint_path.resolve()),
        'fold': fold,
        'sample_index_in_test_fold': sample_index,
        'record_id': str(row.get('PTID', sample_index)),
        'subject_id': str(row.get('subject_id', '')),
        'mri_subject_id': str(row.get('mri_subject_id', '')),
        'true_class': names[int(row['label'])],
        'explained_class': names[int(result.target[0])],
        'predicted_class': names[int(probabilities.argmax())],
        'probabilities': {
            name: float(probabilities[index])
            for index, name in enumerate(names)
        },
        'integration_steps': int(result.integration_steps)
        if hasattr(result, 'integration_steps') else None,
        'convergence_delta': float(result.convergence_delta[0]),
        'graph_is_fixed_during_integration': True,
        'attention_module_names': list(attention),
    }
    (output_dir / f'{stem}.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    arrays = {
        'brain_attribution': _cpu(brain_attribution),
        'region_importance': _cpu(summary['region_importance']),
        'modality_importance': _cpu(summary['modality_importance']),
        'spatial_importance': _cpu(summary['spatial_importance']),
        'signed_region_attribution': _cpu(summary['signed_region_attribution']),
        'roi_label_ids': np.asarray(dataset.valid_ids),
    }
    if getattr(dataset, 'fma_graph', None) is not None:
        arrays['fma_entity_ids'] = np.asarray(dataset.fma_graph.entity_ids)
        arrays['fma_entity_labels'] = np.asarray(
            dataset.fma_graph.entity_labels
        )
    if clinical_attribution is not None:
        arrays['clinical_attribution'] = _cpu(clinical_attribution)
        arrays['clinical_feature_names'] = np.asarray(
            checkpoint['parameters']['model']['clinical_columns']
        )
    if edge_attribution:
        for relation, values in edge_attribution.items():
            if relation == 'dense':
                arrays['edge_attribution__dense'] = _cpu(values)
                continue
            relation_name = '__'.join(relation)
            arrays[f'edge_attribution__{relation_name}'] = _cpu(values)
            arrays[f'edge_attribution__{relation_name}__edge_index'] = _cpu(
                edge_indices[relation]
            )
    for module_name, heads in attention.items():
        safe_name = module_name.replace('.', '__')
        for head_index, head in enumerate(heads):
            prefix = f'attention__{safe_name}__head{head_index}'
            if isinstance(head, dict):
                arrays[f'{prefix}__edge_index'] = _cpu(head['edge_index'])
                arrays[f'{prefix}__coefficients'] = _cpu(head['coefficients'])
            else:
                arrays[prefix] = _cpu(head)
    np.savez_compressed(output_dir / f'{stem}.npz', **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'checkpoint_dir', type=Path,
        help='Directory containing one checkpoint for each outer fold',
    )
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument(
        '--samples-per-class-per-fold', type=int, default=6,
        help='Use 0 for every held-out record',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    checkpoint_paths = sorted(args.checkpoint_dir.glob('*.pth'))
    if not checkpoint_paths:
        raise FileNotFoundError(f'No checkpoints found in {args.checkpoint_dir}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    full_frame = prepare_multiclass_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for checkpoint_path in checkpoint_paths:
        checkpoint = load_checkpoint(checkpoint_path, device)
        parameters = checkpoint['parameters']
        fold = int(parameters['identity']['fold'])
        raw_test_frame = held_out_frame(full_frame, parameters)
        test_frame = apply_normalizer_state(
            raw_test_frame, checkpoint.get('normalizer_state')
        )
        dataset = build_dataset(test_frame, checkpoint, PROJECT_ROOT)
        model = build_model(checkpoint, dataset, device)
        indices = _select_indices(
            test_frame, args.samples_per_class_per_fold, args.seed + 100 * fold
        )
        print(f'Fold {fold}: explaining {len(indices)} records')

        for position, sample_index in enumerate(indices, start=1):
            stem = f'{checkpoint_path.stem}__sample{sample_index}'
            if (
                not args.overwrite
                and (args.output_dir / f'{stem}.json').exists()
                and (args.output_dir / f'{stem}.npz').exists()
            ):
                print(f'  [{position}/{len(indices)}] sample {sample_index}: cached')
                continue
            graph, adjacency, clinical = prepare_sample(
                dataset, sample_index, device
            )
            with torch.no_grad():
                plain_logits = model(graph, adjacency, clinical)
            with attention_capture(model):
                captured_logits = model(graph, adjacency, clinical)
            if not torch.allclose(
                plain_logits, captured_logits, rtol=1e-6, atol=1e-6
            ):
                difference = (plain_logits - captured_logits).abs().max().item()
                raise RuntimeError(
                    f'attention capture changed model output by {difference}'
                )
            attention = collect_attention(model)
            if dataset.is_heterogeneous:
                result, attributed = explain_heterogeneous(
                    model, graph, clinical, target=None,
                    node_types=['brain_region'], steps=args.steps,
                )
                brain = attributed['brain_region'].unsqueeze(0)
                clinical_attr = attributed.get('clinical')
            else:
                result = explain_homogeneous(
                    model, graph, adjacency, clinical,
                    target=None, steps=args.steps,
                )
                brain = result.attributions[0]
                clinical_attr = (
                    result.attributions[1] if clinical is not None else None
                )
            edge_attribution = None
            edge_indices = None
            if parameters['identity']['model'] == 'GCN':
                explained_target = result.target
                if dataset.is_heterogeneous:
                    _, edge_attribution = explain_heterogeneous_edges(
                        model, graph, clinical, target=explained_target,
                        steps=args.steps,
                    )
                    edge_indices = {
                        edge_type: graph[edge_type].edge_index
                        for edge_type in edge_attribution
                    }
                else:
                    edge_result = explain_homogeneous_edges(
                        model, graph, adjacency, clinical,
                        target=explained_target, steps=args.steps,
                    )
                    edge_attribution = {
                        'dense': edge_result.attributions[0]
                    }
            result.integration_steps = args.steps
            _save_record(
                args.output_dir, checkpoint_path, checkpoint, dataset,
                raw_test_frame, sample_index, result, brain, clinical_attr,
                attention, edge_attribution, edge_indices,
            )
            total += 1
            print(
                f'  [{position}/{len(indices)}] sample {sample_index}: '
                f'delta={float(result.convergence_delta[0]):.5g}'
            )
        del model, dataset
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f'Created {total} new explanations in {args.output_dir}')


if __name__ == '__main__':
    main()
