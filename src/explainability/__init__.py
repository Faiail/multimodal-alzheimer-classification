"""Inference-only explanation utilities for trained graph classifiers."""

from .attention import attention_capture, collect_attention
from .checkpoint import (
    apply_normalizer_state,
    build_dataset,
    build_model,
    held_out_frame,
    load_checkpoint,
    prepare_sample,
)
from .explainer_run import BatchExplainerRun, ExplainerRun
from .integrated_gradients import (
    AttributionResult,
    explain_heterogeneous,
    explain_heterogeneous_edges,
    explain_homogeneous,
    explain_homogeneous_edges,
    integrated_gradients,
    summarize_brain_attributions,
)


def exe_explain(parameters: dict, cls: str):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


__all__ = [
    "AttributionResult",
    "BatchExplainerRun",
    "ExplainerRun",
    "apply_normalizer_state",
    "attention_capture",
    "build_dataset",
    "build_model",
    "collect_attention",
    "exe_explain",
    "explain_heterogeneous",
    "explain_heterogeneous_edges",
    "explain_homogeneous",
    "explain_homogeneous_edges",
    "held_out_frame",
    "integrated_gradients",
    "load_checkpoint",
    "prepare_sample",
    "summarize_brain_attributions",
]

