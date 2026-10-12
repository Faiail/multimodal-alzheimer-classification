"""Inference-only explanation utilities for trained graph classifiers."""

from .attention import attention_capture, collect_attention
from .integrated_gradients import (
    AttributionResult,
    explain_heterogeneous,
    explain_heterogeneous_edges,
    explain_homogeneous,
    explain_homogeneous_edges,
    integrated_gradients,
    summarize_brain_attributions,
)

__all__ = [
    'AttributionResult',
    'attention_capture',
    'collect_attention',
    'explain_heterogeneous',
    'explain_heterogeneous_edges',
    'explain_homogeneous',
    'explain_homogeneous_edges',
    'integrated_gradients',
    'summarize_brain_attributions',
]
