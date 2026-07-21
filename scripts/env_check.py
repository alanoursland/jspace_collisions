"""Report what this environment can run: core layer, model layer, GPU."""

from __future__ import annotations

import importlib.util
import sys


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    core = has("numpy")
    print(f"core layer (numpy): {'ok' if core else 'MISSING - uv pip install -e .'}")
    if core:
        from jspace.prompts import all_pairs

        print(f"jspace import: ok ({len(all_pairs())} benchmark pairs loaded)")
    models = has("torch") and has("transformers")
    print(f"model layer (torch+transformers): {'ok' if models else 'not installed ([models] extra)'}")
    if models:
        import torch

        gpu = torch.cuda.is_available()
        print(f"gpu: {torch.cuda.get_device_name(0) if gpu else 'none (CPU smoke tests only)'}")
    return 0 if core else 1


if __name__ == "__main__":
    raise SystemExit(main())
