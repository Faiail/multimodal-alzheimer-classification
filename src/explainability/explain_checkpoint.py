"""Run the paper-style explainer on an existing checkpoint and test record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.attention import attention_capture, collect_attention
from src.explainability.integrated_gradients import (
    explain_heterogeneous,
    explain_heterogeneous_edges,
    explain_homogeneous,
    explain_homogeneous_edges,
    summarize_brain_attributions,
)
from src.explainability.checkpoint import (
    apply_normalizer_state,
    build_dataset,
    build_model,
    held_out_frame,
    load_checkpoint,
    prepare_sample,
)
from src.experiment.train_multiclass import prepare_multiclass_data


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def main():
    parser = argparse.ArgumentParser(
        description='Explain one held-out record without retraining the model.'
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--target', type=int, choices=(0, 1, 2))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'Results' / 'explanations')
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    full_frame = prepare_multiclass_data()
    test_frame = held_out_frame(full_frame, checkpoint['parameters'])
    test_frame = apply_normalizer_state(test_frame, checkpoint.get('normalizer_state'))
    dataset = build_dataset(test_frame, checkpoint, PROJECT_ROOT)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f'sample-index must be in [0, {len(dataset) - 1}]')
    model = build_model(checkpoint, dataset, device)
    graph, adjacency, clinical = prepare_sample(dataset, args.sample_index, device)

    with torch.no_grad():
        reference_logits = model(graph, adjacency, clinical)
    with attention_capture(model):
        captured_logits = model(graph, adjacency, clinical)
    if not torch.equal(reference_logits, captured_logits):
        difference = (reference_logits - captured_logits).abs().max().item()
        raise RuntimeError(f'attention capture changed predictions by {difference}')
    attention = collect_attention(model)

    if dataset.is_heterogeneous:
        result, attributed = explain_heterogeneous(
            model, graph, clinical, target=args.target, steps=args.steps
        )
        brain_attribution = attributed['brain_region']
        # A one-record PyG batch stores nodes without a leading batch dimension.
        brain_attribution = brain_attribution.unsqueeze(0)
        fma_attribution = attributed.get('fma_entity')
        clinical_attribution = attributed.get('clinical')
    else:
        result = explain_homogeneous(
            model, graph, adjacency, clinical,
            target=args.target, steps=args.steps
        )
        brain_attribution = result.attributions[0]
        clinical_attribution = (
            result.attributions[1] if clinical is not None else None
        )
        fma_attribution = None

    edge_attribution = None
    edge_indices = None
    if checkpoint['parameters']['identity']['model'] == 'GCN':
        if dataset.is_heterogeneous:
            _, edge_attribution = explain_heterogeneous_edges(
                model, graph, clinical, target=result.target, steps=args.steps
            )
            edge_indices = {
                edge_type: graph[edge_type].edge_index
                for edge_type in edge_attribution
            }
        else:
            edge_result = explain_homogeneous_edges(
                model, graph, adjacency, clinical,
                target=result.target, steps=args.steps
            )
            edge_attribution = {'dense': edge_result.attributions[0]}

    summary = summarize_brain_attributions(brain_attribution)
    row = test_frame.iloc[args.sample_index]
    class_names = checkpoint.get('class_names', ['CN', 'MCI', 'AD'])
    probabilities = torch.softmax(result.output, dim=1)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.checkpoint.stem}__sample{args.sample_index}"
    metadata_path = args.output_dir / f'{stem}.json'
    arrays_path = args.output_dir / f'{stem}.npz'
    metadata = {
        'checkpoint': str(args.checkpoint.resolve()),
        'fold': checkpoint['parameters']['identity']['fold'],
        'sample_index_in_test_fold': args.sample_index,
        'record_id': str(row.get('PTID', args.sample_index)),
        'true_class': class_names[int(row['label'])],
        'explained_class': class_names[int(result.target[0])],
        'predicted_class': class_names[int(probabilities.argmax())],
        'probabilities': {
            name: float(probabilities[index])
            for index, name in enumerate(class_names)
        },
        'integration_steps': args.steps,
        'convergence_delta': float(result.convergence_delta[0]),
        'graph_is_fixed_during_integration': True,
        'attention_module_names': list(attention),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

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
        arrays['fma_entity_labels'] = np.asarray(dataset.fma_graph.entity_labels)
    if clinical_attribution is not None:
        arrays['clinical_attribution'] = _cpu(clinical_attribution)
        arrays['clinical_feature_names'] = np.asarray(
            checkpoint['parameters']['model']['clinical_columns']
        )
    if fma_attribution is not None:
        arrays['fma_attribution'] = _cpu(fma_attribution)
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
            if isinstance(head, dict):
                arrays[f'attention__{safe_name}__head{head_index}__edge_index'] = _cpu(head['edge_index'])
                arrays[f'attention__{safe_name}__head{head_index}__coefficients'] = _cpu(head['coefficients'])
            else:
                arrays[f'attention__{safe_name}__head{head_index}'] = _cpu(head)
    np.savez_compressed(arrays_path, **arrays)
    print(metadata_path)
    print(arrays_path)


if __name__ == '__main__':
    main()
