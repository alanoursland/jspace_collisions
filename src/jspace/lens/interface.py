"""Lens interfaces.

CAUTION: these interfaces are written from the technical paper's description
of the J-lens (per-token directions such that L(h) scores how much h promotes
each vocabulary token later in the sequence). They must be reconciled against
the actual API of the Anthropic reference implementation
(github.com/anthropics/jacobian-lens) during Phase 1, before experiment code
depends on details. Keep adapters thin.

Everything here is framework-agnostic: activations and logits are NumPy
arrays at the interface boundary, regardless of what runs underneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Readout:
    """A J-lens readout L(h) for one activation."""

    lens_logits: np.ndarray  # shape (vocab,)
    layer: int
    position: int
    top_k_tokens: tuple[str, ...] = ()  # decoded, sorted by score desc

    def top_k_indices(self, k: int = 20) -> np.ndarray:
        idx = np.argpartition(-self.lens_logits, k)[:k]
        return idx[np.argsort(-self.lens_logits[idx], kind="stable")]


@runtime_checkable
class Lens(Protocol):
    """Anything that maps residual activations to vocabulary-space readouts.

    Implementations planned:
        JLens          adapter over the reference implementation (Phase 1)
        ShuffledLens   label-shuffled control (E9)
        RandomTransportLens  random-orthogonal control (E9)
        Context/phrase/relation lenses (E8, E12)
    """

    def readout(self, activation: np.ndarray, layer: int, position: int) -> Readout:
        """Compute L(h) for a single residual-stream activation."""
        ...

    @property
    def vocab_size(self) -> int: ...
