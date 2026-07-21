"""E2 collision sweep: J-distance vs behavior-distance over minimal pairs.

For every PromptPair in the benchmark and every requested layer:

  1. Run both variants (statement only, no probe); take the residual at the
     final statement token and compute lens readouts under the fitted J-lens
     and each control lens (shuffled labels, random transport, logit lens).
  2. Behavior: P(answer | statement + probe) over the pair's two expected
     answers, for both variants; behavior distance = JS divergence between
     the two variants' answer distributions, plus an answer-flip flag.
  3. Emit one JSON record per (pair, layer, lens) with all J-distance
     metrics, plus per-pair behavior metrics.

H2 predicts: role_reversal / relation_binding / negation pairs show LOW
J-distance (similar bag-of-tokens readout) but HIGH behavior distance —
candidate collisions. Controls calibrate what "low J-distance" means.

Usage (CPU, ~1-2 s/pair/layer for readouts + ~2 s/pair for behavior):

    python experiments/e02_collision_search/run_sweep.py \
        --lens data/lens/qwen2.5-0.5b_wikitext100.pt \
        --layers 6 10 14 18 22 --out results/e02_collision_search/sweep_v1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from jspace.lens.adapter import JLensAdapter, LogitLensBaseline
from jspace.lens.controls import RandomTransportLens, ShuffledLens
from jspace.metrics import (
    cosine_distance,
    js_divergence,
    softmax,
    topk_overlap,
)
from jspace.metrics.behavior import collision_score
from jspace.metrics.jdist import rbo_from_logits
from jspace.models.wrapper import ModelWrapper
from jspace.prompts import all_pairs


def jdistances(la: np.ndarray, lb: np.ndarray) -> dict[str, float]:
    return {
        "cosine": cosine_distance(la, lb),
        "js": js_divergence(softmax(la), softmax(lb)),
        "top20_jaccard": topk_overlap(la, lb, k=20),
        "rbo50": rbo_from_logits(la, lb, k=50),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 10, 14, 18, 22])
    ap.add_argument("--out", default="results/e02_collision_search/sweep_v1")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs for smoke runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = ModelWrapper(args.model)
    jl = JLensAdapter.load(args.lens, wrapper)
    lenses = [
        jl,
        ShuffledLens(jl, seed=args.seed),
        RandomTransportLens(jl, wrapper, seed=args.seed),
        LogitLensBaseline(wrapper, layers=jl.source_layers),
    ]
    layers = [l for l in args.layers if l in jl.source_layers]
    if layers != args.layers:
        print(f"note: restricting layers to fitted set -> {layers}")

    pairs = all_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    # Resume support: pairs already fully present in records.jsonl are
    # skipped, and new records are appended per-pair (container restarts
    # kill long runs; see docs/notebook 2026-07-21 infra entry).
    out_file = out_dir / "records.jsonl"
    done_pairs: set[str] = set()
    if out_file.exists():
        counts: dict[str, int] = {}
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            counts[r["pair_id"]] = counts.get(r["pair_id"], 0) + 1
        expected = len(layers) * len(lenses)
        done_pairs = {pid for pid, n in counts.items() if n >= expected}
        if done_pairs:
            print(f"resuming: {len(done_pairs)}/{len(pairs)} pairs already complete")
            # drop partial records for pairs that were interrupted mid-write
            kept = [
                line
                for line in out_file.read_text().splitlines()
                if json.loads(line)["pair_id"] in done_pairs
            ]
            out_file.write_text("\n".join(kept) + ("\n" if kept else ""))

    t0 = time.time()
    n_new = 0
    with out_file.open("a") as sink:
        for i, pair in enumerate(pairs):
            if pair.pair_id in done_pairs:
                continue
            cap_a = wrapper.capture(pair.prompt_a, layers=layers)
            cap_b = wrapper.capture(pair.prompt_b, layers=layers)
            answers = [pair.expected_a, pair.expected_b]
            beh_a = wrapper.answer_distribution(pair.full_a(), answers)
            beh_b = wrapper.answer_distribution(pair.full_b(), answers)
            behavior = {
                "behavior_js": js_divergence(beh_a, beh_b),
                "answer_a_dist": beh_a.tolist(),
                "answer_b_dist": beh_b.tolist(),
                # does the model actually flip its preferred answer across variants?
                "answer_flip": bool(np.argmax(beh_a) != np.argmax(beh_b)),
                # is each variant answered as expected? (task competence check)
                "correct_a": bool(np.argmax(beh_a) == 0),
                "correct_b": bool(np.argmax(beh_b) == 1),
            }
            pair_records = []
            for layer in layers:
                ha = cap_a.activations[layer][-1].numpy()
                hb = cap_b.activations[layer][-1].numpy()
                for lens in lenses:
                    jd = jdistances(
                        lens.readout_logits(ha, layer), lens.readout_logits(hb, layer)
                    )
                    pair_records.append(
                        {
                            "pair_id": pair.pair_id,
                            "category": pair.category,
                            "layer": layer,
                            "lens": lens.name,
                            **jd,
                            **behavior,
                            "collision_score": collision_score(
                                jd["js"], behavior["behavior_js"]
                            ),
                        }
                    )
            sink.write("".join(json.dumps(r) + "\n" for r in pair_records))
            sink.flush()
            n_new += 1
            if (i + 1) % 10 == 0 or i == len(pairs) - 1:
                rate = (time.time() - t0) / max(n_new, 1)
                print(f"{i + 1}/{len(pairs)} pairs ({rate:.1f}s/pair this run)")

    n_records = sum(1 for _ in out_file.open())
    manifest = {
        "model": args.model,
        "lens": args.lens,
        "layers": layers,
        "n_pairs": len(pairs),
        "n_records": n_records,
        "seconds_this_run": round(time.time() - t0),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{n_records} records in {out_file} ({n_new} pairs this run)")


if __name__ == "__main__":
    main()
