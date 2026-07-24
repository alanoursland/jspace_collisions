"""Build a usable JacobianLens from a fit() checkpoint's running sum.

Lets us snapshot an in-progress fit (the checkpoint is written atomically
after every prompt) without interrupting it:

    python scripts/lens_from_checkpoint.py data/lens/qwen2.5-0.5b_wikitext100.pt.ckpt \
        data/lens/interim.pt
"""

from __future__ import annotations

import sys

import torch
from jlens import JacobianLens


def main() -> None:
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n = state["n_done"]
    if n == 0:
        raise SystemExit("checkpoint has no completed prompts yet")
    jacobians = {layer: J / n for layer, J in state["jacobian_sum"].items()}
    d_model = next(iter(jacobians.values())).shape[0]
    lens = JacobianLens(jacobians, n_prompts=n, d_model=d_model)
    lens.save(out_path)
    print(f"{lens!r} -> {out_path}")


if __name__ == "__main__":
    main()
