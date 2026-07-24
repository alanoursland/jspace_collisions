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

import jlens
import torch
import transformers

from jspace.models.wrapper import resolve_device, resolve_dtype


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument(
        "--model-id",
        default=None,
        help="canonical registry ID when --model is a local snapshot",
    )
    ap.add_argument(
        "--model-revision",
        default=None,
        help="immutable model revision or commit recorded in the manifest",
    )
    ap.add_argument("--corpus", default="data/fit_corpus_wikitext120.json")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument(
        "--source-layers",
        type=int,
        nargs="+",
        default=None,
        help="fit only these block-output layers; default = every layer below target",
    )
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    ap.add_argument("--out", default="data/lens/qwen2.5-0.5b_wikitext100.pt")
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    jlens.configure_logging()

    prompts = json.loads(pathlib.Path(args.corpus).read_text())[: args.n_prompts]
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    hf.to(device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = str(out) + ".ckpt"
    resume_from_n_prompts = 0
    if pathlib.Path(ckpt).exists():
        checkpoint_state = torch.load(ckpt, map_location="cpu", weights_only=True)
        resume_from_n_prompts = int(checkpoint_state.get("n_done", 0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=args.source_layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=ckpt,
        checkpoint_every=1,
        resume=True,
    )
    lens.save(str(out))
    fit_seconds_this_run = round(time.time() - t0)
    prior_fit_seconds = 0
    manifest_path = out.with_suffix(".manifest.json")
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        prior_fit_seconds = previous_manifest.get(
            "fit_seconds_total",
            previous_manifest.get("fit_seconds", 0),
        )
    fit_seconds_total = (
        prior_fit_seconds + fit_seconds_this_run
        if prior_fit_seconds or resume_from_n_prompts == 0
        else None
    )
    manifest = {
        "model": args.model_id or args.model,
        "model_path": args.model if args.model_id else None,
        "model_revision": args.model_revision,
        "corpus": args.corpus,
        "n_prompts": len(prompts),
        "max_seq_len": args.max_seq_len,
        "dim_batch": args.dim_batch,
        "source_layers": lens.source_layers,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_allocated_gib": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            if device.type == "cuda"
            else None
        ),
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "resume_from_n_prompts": resume_from_n_prompts,
        "fit_seconds_this_run": fit_seconds_this_run,
        "fit_seconds_total": fit_seconds_total,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved {out} in {fit_seconds_this_run}s")


if __name__ == "__main__":
    main()
