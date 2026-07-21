"""E1: replicate the paper's lens-quality evals on our fitted lens.

Uses the evaluation sets shipped with the reference repo
(data/evaluations/lens-eval-*.json): each item has a prompt whose latent
`intermediates` should surface in the lens readout at the final prompt
position. Metric (per the repo's README): pass@k = mean over items of the
fraction of intermediates whose min-over-layers lens rank <= k.

Content claims calibrate against ShuffledLens (docs/metrics.md): a shuffled
lens should score ~0.

    python experiments/e01_replication/run_lens_evals.py \
        --lens data/lens/qwen2.5-0.5b_wikitext100.pt \
        --evals multihop association \
        --out results/e01_replication/lens_evals_v1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from jspace.lens.adapter import JLensAdapter, LogitLensBaseline
from jspace.lens.controls import RandomTransportLens, ShuffledLens
from jspace.models.wrapper import ModelWrapper

EVAL_DIR = pathlib.Path("/workspace/jacobian-lens/data/evaluations")


def intermediate_token_ids(wrapper: ModelWrapper, word: str) -> list[int]:
    """Candidate first-token ids for an intermediate word.

    Tokenizers split differently with/without a leading space and case; an
    intermediate counts as surfaced if any variant's first token appears.
    """
    ids = []
    for variant in (f" {word}", word, f" {word.lower()}", word.lower()):
        toks = wrapper.tokenizer(variant, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks[0])
    return sorted(set(ids))


def min_rank(lens_logits: np.ndarray, token_ids: list[int]) -> int:
    """Best (lowest) rank of any candidate token id; ranks start at 1."""
    order = np.argsort(-lens_logits)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(min(ranks[t] for t in token_ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--evals", nargs="+", default=["multihop", "association"])
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="workspace band; default = all fitted layers")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/e01_replication/lens_evals_v1")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = ModelWrapper(args.model)
    jl = JLensAdapter.load(args.lens, wrapper)
    layers = args.layers or jl.source_layers
    lenses = [
        jl,
        ShuffledLens(jl, seed=0),
        RandomTransportLens(jl, wrapper, seed=0),
        LogitLensBaseline(wrapper, layers=layers),
    ]

    records = []
    t0 = time.time()
    for eval_name in args.evals:
        items = json.loads((EVAL_DIR / f"lens-eval-{eval_name}.json").read_text())["items"]
        if args.limit:
            items = items[: args.limit]
        for item in items:
            if not isinstance(item["prompt"], str):
                continue  # multi-turn prompts: out of scope for the base model
            cap = wrapper.capture(item["prompt"], layers=layers)
            for lens in lenses:
                per_layer = {
                    layer: lens.readout_logits(cap.activations[layer][-1].numpy(), layer)
                    for layer in layers
                }
                for word in item["intermediates"]:
                    tids = intermediate_token_ids(wrapper, word)
                    best = min(min_rank(per_layer[layer], tids) for layer in layers)
                    records.append(
                        {
                            "eval": eval_name,
                            "item": item["name"],
                            "intermediate": word,
                            "lens": lens.name,
                            "min_rank": best,
                        }
                    )
        print(f"{eval_name}: done ({time.time() - t0:.0f}s)")

    with (out_dir / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    lines = ["# E1 lens-quality evals", "",
             f"model: {args.model} | lens: {args.lens} | layers: {layers}", ""]
    lines.append("| eval | lens | " + " | ".join(f"pass@{k}" for k in args.ks) + " | n |")
    lines.append("|---" * (len(args.ks) + 3) + "|")
    for eval_name in args.evals:
        for lens in lenses:
            rs = [r for r in records if r["eval"] == eval_name and r["lens"] == lens.name]
            if not rs:
                continue
            passes = [
                f"{np.mean([r['min_rank'] <= k for r in rs]):.2f}" for k in args.ks
            ]
            lines.append(
                f"| {eval_name} | {lens.name} | " + " | ".join(passes) + f" | {len(rs)} |"
            )
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
