"""Lens protocol: anything that maps residual activations to vocab logits.

Aligned with the reference implementation (jlens): a lens is per-layer — the
readout of activation h at source layer l is a vocabulary logit vector.
NumPy at the boundary; implementations may use torch internally.

Implementations:
    JLensAdapter        (adapter.py)  fitted jlens.JacobianLens readout
    LogitLensBaseline   (adapter.py)  identity transport (use_jacobian=False)
    ShuffledLens        (controls.py) E9 control: permuted vocab labels
    RandomTransportLens (controls.py) E9 control: random orthogonal J_l
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Lens(Protocol):
    name: str

    @property
    def source_layers(self) -> list[int]: ...

    def readout_logits(self, activation: np.ndarray, layer: int) -> np.ndarray:
        """Lens logits ``(..., vocab)`` for activations ``(..., d_model)``."""
        ...
