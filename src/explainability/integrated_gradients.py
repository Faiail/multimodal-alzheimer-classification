"""Integrated Gradients used by the paper-style model explainer.

The implementation has no Captum dependency. It explains a selected class logit
and keeps graph topology fixed along the integration path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


TensorTuple = tuple[torch.Tensor, ...]


@dataclass
class AttributionResult:
    attributions: TensorTuple
    target: torch.Tensor
    output: torch.Tensor
    baseline_output: torch.Tensor
    convergence_delta: torch.Tensor


def _as_tuple(value: torch.Tensor | Sequence[torch.Tensor]) -> TensorTuple:
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def integrated_gradients(
    forward_func: Callable[..., torch.Tensor],
    inputs: torch.Tensor | Sequence[torch.Tensor],
    target: int | torch.Tensor | None = None,
    baselines: torch.Tensor | Sequence[torch.Tensor] | None = None,
    steps: int = 64,
) -> AttributionResult:
    """Attribute one class logit per sample with trapezoidal integration."""
    if steps < 2:
        raise ValueError('steps must be at least 2')
    input_tuple = _as_tuple(inputs)
    if not input_tuple:
        raise ValueError('at least one input tensor is required')
    baseline_tuple = (
        tuple(torch.zeros_like(item) for item in input_tuple)
        if baselines is None else _as_tuple(baselines)
    )
    if len(input_tuple) != len(baseline_tuple):
        raise ValueError('inputs and baselines must contain the same number of tensors')
    for item, baseline in zip(input_tuple, baseline_tuple):
        if item.shape != baseline.shape:
            raise ValueError('each baseline must have the same shape as its input')

    with torch.no_grad():
        output = forward_func(*input_tuple)
        baseline_output = forward_func(*baseline_tuple)
    if output.ndim != 2:
        raise ValueError('forward_func must return [batch, classes] logits')
    batch_size = output.size(0)
    if target is None:
        target_tensor = output.argmax(dim=1)
    elif isinstance(target, int):
        target_tensor = torch.full(
            (batch_size,), target, dtype=torch.long, device=output.device
        )
    else:
        target_tensor = target.to(device=output.device, dtype=torch.long).reshape(-1)
        if target_tensor.numel() != batch_size:
            raise ValueError('target tensor must contain one class index per sample')

    gradient_sums = [torch.zeros_like(item) for item in input_tuple]
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=input_tuple[0].device)
    for step_index, alpha in enumerate(alphas):
        interpolated = tuple(
            (baseline + alpha * (item - baseline)).detach().requires_grad_(True)
            for item, baseline in zip(input_tuple, baseline_tuple)
        )
        logits = forward_func(*interpolated)
        selected = logits.gather(1, target_tensor[:, None]).sum()
        gradients = torch.autograd.grad(selected, interpolated)
        weight = 0.5 if step_index in (0, steps) else 1.0
        for index, gradient in enumerate(gradients):
            gradient_sums[index] += weight * gradient.detach()

    attributions = tuple(
        (item - baseline) * gradient_sum / steps
        for item, baseline, gradient_sum in zip(
            input_tuple, baseline_tuple, gradient_sums
        )
    )
    selected_output = output.gather(1, target_tensor[:, None]).squeeze(1)
    selected_baseline = baseline_output.gather(
        1, target_tensor[:, None]
    ).squeeze(1)
    attributed_difference = sum(
        attribution.reshape(batch_size, -1).sum(dim=1)
        for attribution in attributions
    )
    return AttributionResult(
        attributions=attributions,
        target=target_tensor.detach(),
        output=output.detach(),
        baseline_output=baseline_output.detach(),
        convergence_delta=(
            selected_output - selected_baseline - attributed_difference
        ).detach(),
    )


def explain_homogeneous(
    model,
    x_graph: torch.Tensor,
    adjacency: torch.Tensor,
    x_clinical: torch.Tensor | None = None,
    target: int | torch.Tensor | None = None,
    steps: int = 64,
) -> AttributionResult:
    """Explain graph and optional clinical inputs while holding adjacency fixed."""
    model.eval()
    if x_clinical is None:
        return integrated_gradients(
            lambda graph: model(graph, adjacency, None),
            x_graph,
            target=target,
            steps=steps,
        )
    return integrated_gradients(
        lambda graph, clinical: model(graph, adjacency, clinical),
        (x_graph, x_clinical),
        target=target,
        steps=steps,
    )


def explain_homogeneous_edges(
    model,
    x_graph: torch.Tensor,
    adjacency: torch.Tensor,
    x_clinical: torch.Tensor | None = None,
    target: int | torch.Tensor | None = None,
    steps: int = 64,
) -> AttributionResult:
    """Attribute GCN edge weights while keeping node features fixed."""
    model.eval()
    return integrated_gradients(
        lambda edge_weights: model(x_graph, edge_weights, x_clinical),
        adjacency,
        target=target,
        steps=steps,
    )


def explain_heterogeneous(
    model,
    graph,
    x_clinical: torch.Tensor | None = None,
    target: int | torch.Tensor | None = None,
    node_types: Sequence[str] | None = None,
    steps: int = 64,
) -> tuple[AttributionResult, dict[str, torch.Tensor]]:
    """Explain heterogeneous node inputs while holding every edge fixed."""
    model.eval()
    selected_types = list(node_types or graph.node_types)
    originals = {node_type: graph[node_type].x for node_type in selected_types}
    inputs = tuple(originals[node_type] for node_type in selected_types)
    include_clinical = x_clinical is not None
    if include_clinical:
        inputs += (x_clinical,)

    def forward(*values):
        for node_type, value in zip(selected_types, values):
            graph[node_type].x = value
        clinical = values[-1] if include_clinical else None
        return model(graph, None, clinical)

    try:
        result = integrated_gradients(
            forward, inputs, target=target, steps=steps
        )
    finally:
        for node_type, value in originals.items():
            graph[node_type].x = value

    attributed_nodes = {
        node_type: attribution
        for node_type, attribution in zip(selected_types, result.attributions)
    }
    if include_clinical:
        attributed_nodes['clinical'] = result.attributions[-1]
    return result, attributed_nodes


def explain_heterogeneous_edges(
    model,
    graph,
    x_clinical: torch.Tensor | None = None,
    target: int | torch.Tensor | None = None,
    steps: int = 64,
    edge_types: Sequence[tuple[str, str, str]] | None = None,
) -> tuple[AttributionResult, dict[tuple[str, str, str], torch.Tensor]]:
    """Attribute selected heterogeneous GCN relation weights.

    Edge indices remain fixed.  When ``edge_types`` is omitted, all non-empty
    relations carrying ``edge_weight`` are explained, including FMA mapping
    and ontology relations.
    """
    model.eval()
    selected_edge_types = list(edge_types) if edge_types is not None else [
        edge_type for edge_type in graph.edge_types
        if getattr(graph[edge_type], 'edge_weight', None) is not None
        and graph[edge_type].edge_weight.numel() > 0
    ]
    if not selected_edge_types:
        raise ValueError('the graph contains no weighted relations')
    originals = {
        edge_type: graph[edge_type].edge_weight
        for edge_type in selected_edge_types
    }
    inputs = tuple(originals[edge_type] for edge_type in selected_edge_types)

    def forward(*weights):
        for edge_type, value in zip(selected_edge_types, weights):
            graph[edge_type].edge_weight = value
        return model(graph, None, x_clinical)

    try:
        result = integrated_gradients(
            forward, inputs, target=target, steps=steps
        )
    finally:
        for edge_type, value in originals.items():
            graph[edge_type].edge_weight = value
    return result, dict(zip(selected_edge_types, result.attributions))


def summarize_brain_attributions(
    attribution: torch.Tensor,
    imaging_features: int = 4,
) -> dict[str, torch.Tensor]:
    """Aggregate absolute region and modality contributions as in the paper."""
    if attribution.ndim not in (2, 3):
        raise ValueError(
            'brain attribution must have shape [nodes, features] or '
            '[batch, nodes, features]'
        )
    if attribution.size(-1) < imaging_features:
        raise ValueError('attribution has fewer columns than imaging_features')
    absolute = attribution.abs()
    return {
        'region_importance': absolute.sum(dim=-1),
        'modality_importance': absolute[..., :imaging_features].sum(dim=-2),
        'spatial_importance': absolute[..., imaging_features:].sum(dim=-2),
        'signed_region_attribution': attribution.sum(dim=-1),
    }
