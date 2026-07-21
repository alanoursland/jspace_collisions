import numpy as np
import pytest

from jspace.patching import PatchSpec
from jspace.patching.harness import apply_patch_array


H = np.array([1.0, 2.0, 3.0])
V = np.array([1.0, 0.0, 0.0])


def _spec(op, alpha=1.0, vector=V):
    return PatchSpec(op=op, layer=0, position=-1, vector=vector, alpha=alpha)


def test_replace():
    assert np.allclose(apply_patch_array(H, _spec("replace")), V)


def test_add_subtract():
    assert np.allclose(apply_patch_array(H, _spec("add", alpha=2.0)), [3.0, 2.0, 3.0])
    assert np.allclose(apply_patch_array(H, _spec("subtract")), [0.0, 2.0, 3.0])


def test_project_out_removes_component():
    out = apply_patch_array(H, _spec("project_out"))
    assert np.dot(out, V) == pytest.approx(0.0)
    assert np.allclose(out, [0.0, 2.0, 3.0])


def test_project_out_zero_vector_raises():
    with pytest.raises(ValueError):
        apply_patch_array(H, _spec("project_out", vector=np.zeros(3)))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_patch_array(H, _spec("add", vector=np.ones(4)))
