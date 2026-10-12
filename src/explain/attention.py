"""Capture pre-dropout coefficients from the dense and sparse GAT layers."""
from __future__ import annotations

from contextlib import contextmanager


def _attention_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, 'capture_attention') and hasattr(module, 'last_attention'):
            yield name, module


@contextmanager
def attention_capture(model):
    """Enable attention recording temporarily and clear stale captures."""
    modules = list(_attention_modules(model))
    previous = [(module, module.capture_attention) for _, module in modules]
    try:
        for _, module in modules:
            module.capture_attention = True
            module.last_attention = None
        yield model
    finally:
        for module, enabled in previous:
            module.capture_attention = enabled


def collect_attention(model) -> dict[str, object]:
    """Return captured coefficients keyed by the full model module path."""
    return {
        name: module.last_attention
        for name, module in _attention_modules(model)
        if module.last_attention is not None
    }
