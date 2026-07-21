"""Fit the Jacobian lens on an open model using the reference implementation.

Checkpointed and resumable: safe to rerun after interruption. CPU fit of
Qwen2.5-0.5B over ~100 wikitext prompts takes hours; run in the background.

    python scripts/fit_jlens.py [--n-prompts 100] [--max-seq-len 96]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import torch
import transformers

import jlens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--corpus", default="data/fit_corpus_wikitext120.json")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--out", default="data/lens/qwen2.5-0.5b_wikitext100.pt")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    jlens.configure_logging()

    prompts = json.loads(pathlib.Path(args.corpus).read_text())[: args.n_prompts]
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = str(out) + ".ckpt"

    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=ckpt,
        checkpoint_every=1,
        resume=True,
    )
    lens.save(str(out))
    manifest = {
        "model": args.model,
        "corpus": args.corpus,
        "n_prompts": len(prompts),
        "max_seq_len": args.max_seq_len,
        "dim_batch": args.dim_batch,
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "fit_seconds": round(time.time() - t0),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {out} in {manifest['fit_seconds']}s")


if __name__ == "__main__":
    main()
