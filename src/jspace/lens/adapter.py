"""Adapters exposing the reference jlens implementation as jspace Lenses.

Requires the [models] extra plus the jlens package
(pip install -e /path/to/jacobian-lens).
"""

from __future__ import annotations

import numpy as np
import torch

import jlens

from jspace.models.wrapper import ModelWrapper


class JLensAdapter:
    """Readout via a fitted jlens.JacobianLens: unembed(J_l @ h)."""

    def __init__(self, lens: "jlens.JacobianLens", wrapper: ModelWrapper, name: str = "jlens"):
        self.lens = lens
        self.wrapper = wrapper
        self.name = name

    @classmethod
    def load(cls, path: str, wrapper: ModelWrapper) -> "JLensAdapter":
        return cls(jlens.JacobianLens.load(path), wrapper)

    @property
    def source_layers(self) -> list[int]:
        return list(self.lens.source_layers)

    def _unembed(self, h: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.wrapper.lens_model.unembed(h).float().cpu().numpy()

    def readout_logits(self, activation: np.ndarray, layer: int) -> np.ndarray:
        h = torch.from_numpy(np.asarray(activation, dtype=np.float32))
        return self._unembed(self.lens.transport(h, layer))


class LogitLensBaseline:
    """Identity transport: unembed(h) directly (the vanilla logit lens)."""

    def __init__(self, wrapper: ModelWrapper, layers: list[int], name: str = "logit-lens"):
        self.wrapper = wrapper
        self._layers = layers
        self.name = name

    @property
    def source_layers(self) -> list[int]:
        return list(self._layers)

    def readout_logits(self, activation: np.ndarray, layer: int) -> np.ndarray:
        h = torch.from_numpy(np.asarray(activation, dtype=np.float32))
        with torch.no_grad():
            return self.wrapper.lens_model.unembed(h).float().cpu().numpy()
