"""Activation patching harness (interface + specs).

The harness runs a model on a prompt while intervening on residual-stream
activations at chosen (layer, position) sites. Operations required by the
research program (Experiment Families 1–3):

    replace     h := h_source
    add         h := h + alpha * v
    subtract    h := h - alpha * v
    project_out h := h - (h.v / v.v) v          (remove a direction)
    nullspace   h := h + delta, delta chosen so L(h + delta) ~= L(h)
                (fiber mapping, E3 — delta comes from a sampler, not the
                harness itself)

The concrete implementation (Phase 1) will hook a HuggingFace decoder via
forward hooks and requires the [models] extra. This module stays importable
without torch; only specs and pure-array helpers live here for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PatchOp = Literal["replace", "add", "subtract", "project_out"]


@dataclass(frozen=True)
class PatchSpec:
    """One intervention at one site."""

    op: PatchOp
    layer: int
    position: int  # token index in the target run; -1 = last position
    vector: np.ndarray  # replacement activation or direction, shape (d_model,)
    alpha: float = 1.0  # scale for add/subtract


def apply_patch_array(h: np.ndarray, spec: PatchSpec) -> np.ndarray:
    """Apply a PatchSpec to a single activation vector (reference semantics).

    The model harness must match this function exactly; it doubles as the
    ground truth for harness tests.
    """
    v = np.asarray(spec.vector, dtype=h.dtype)
    if v.shape != h.shape:
        raise ValueError(f"shape mismatch: activation {h.shape}, vector {v.shape}")
    if spec.op == "replace":
        return v.copy()
    if spec.op == "add":
        return h + spec.alpha * v
    if spec.op == "subtract":
        return h - spec.alpha * v
    if spec.op == "project_out":
        denom = float(np.dot(v, v))
        if denom == 0.0:
            raise ValueError("cannot project out zero vector")
        return h - (np.dot(h, v) / denom) * v
    raise ValueError(f"unknown op: {spec.op}")
