"""E9 null-control lenses.

Any readout claim must beat these controls (docs/metrics.md). Both wrap a
real lens/model so that everything except the tested property is identical.
"""

from __future__ import annotations

import numpy as np
import torch

from jspace.models.wrapper import ModelWrapper


class ShuffledLens:
    """Applies a fixed random permutation to the base lens's vocab logits.

    Preserves the full logit distribution shape while destroying token
    identity: if readouts look meaningful through this lens, the evaluation
    is reading structure into noise.
    """

    def __init__(self, base, seed: int = 0):
        self.base = base
        self.name = f"shuffled({base.name},seed={seed})"
        self._perm: np.ndarray | None = None  # sized lazily on first readout
        self._rng = np.random.default_rng(seed)

    @property
    def source_layers(self) -> list[int]:
        return self.base.source_layers

    def readout_logits(self, activation: np.ndarray, layer: int) -> np.ndarray:
        logits = self.base.readout_logits(activation, layer)
        if self._perm is None:
            self._perm = self._rng.permutation(logits.shape[-1])
        return logits[self._perm]


class RandomTransportLens:
    """Replaces each J_l with a random orthogonal matrix scaled to match
    ||J_l||_F, then unembeds normally.

    Tests whether results depend on the *fitted* transport or just on
    pushing suitably-scaled residuals through the unembedding.
    """

    def __init__(self, jlens_adapter, wrapper: ModelWrapper, seed: int = 0):
        self.wrapper = wrapper
        self.name = f"random-transport(seed={seed})"
        self._layers = jlens_adapter.source_layers
        rng = np.random.default_rng(seed)
        self._Q: dict[int, torch.Tensor] = {}
        for l in self._layers:
            J = jlens_adapter.lens.jacobians[l]
            d = J.shape[0]
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            scale = float(torch.linalg.matrix_norm(J.float())) / np.sqrt(d)
            self._Q[l] = torch.from_numpy((Q * scale).astype(np.float32))

    @property
    def source_layers(self) -> list[int]:
        return list(self._layers)

    def readout_logits(self, activation: np.ndarray, layer: int) -> np.ndarray:
        h = torch.from_numpy(np.asarray(activation, dtype=np.float32))
        with torch.no_grad():
            transported = h @ self._Q[layer].T
            return self.wrapper.lens_model.unembed(transported).float().cpu().numpy()
