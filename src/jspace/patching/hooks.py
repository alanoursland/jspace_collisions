"""Residual-stream patching via forward hooks on decoder blocks.

A patch replaces the residual at (layer, position) with a supplied vector
during a forward pass. Positions are absolute token indices; patches on
prompt-prefix positions remain valid when scoring answer continuations,
because the prefix tokenization is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualPatch:
    layer: int
    position: int
    vector: torch.Tensor  # (d_model,)


@contextmanager
def apply_patches(layers: list[torch.nn.Module], patches: list[ResidualPatch]):
    """Context manager installing hooks that overwrite block outputs.

    `layers` is the model's decoder block list (jlens layout: the residual at
    layer l is block l's output). Multiple patches per layer are allowed.
    """
    by_layer: dict[int, list[ResidualPatch]] = {}
    for p in patches:
        by_layer.setdefault(p.layer, []).append(p)

    handles = []

    def make_hook(layer_patches: list[ResidualPatch]):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for p in layer_patches:
                hidden[:, p.position, :] = p.vector.to(hidden.dtype).to(hidden.device)
            return output

        return hook

    try:
        for layer_idx, layer_patches in by_layer.items():
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_patches)))
        yield
    finally:
        for h in handles:
            h.remove()
