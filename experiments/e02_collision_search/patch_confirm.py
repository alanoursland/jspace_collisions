"""E2 step 5: patch confirmation of collision candidates.

For each top collision pair (near-identical layer-L readouts, opposite
confident behavior), patch variant A's residuals into variant B's probed run
(and vice versa) at layer L and measure whether the answer follows the patch.

Conditions per direction:
  none        baseline
  full_last   swap the residual at the final statement token only
  full_stmt   swap residuals at every statement token
  jvis_stmt   swap only the lens-visible component of the difference
              (projection onto the top right-singular subspace of J_L
              covering 90% of squared singular mass), all statement tokens
  fiber_stmt  swap only the complementary (lens-low-gain / fiber) component

Logic: the sweep already shows L(h_a) ~= L(h_b). If full_stmt transfers
behavior, the site is causally sufficient and the transferred difference is
invisible to the lens — a causal collision. jvis/fiber then localize which
subspace carries it; H1 predicts fiber_stmt ~= full_stmt >> jvis_stmt.

    python experiments/e02_collision_search/patch_confirm.py \
        --lens data/lens/qwen2.5-0.5b_wikitext100.pt --layer 16 --top 8
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jlens
import numpy as np
import torch

from jspace.metrics import js_divergence
from jspace.models.wrapper import ModelWrapper
from jspace.patching.hooks import ResidualPatch
from jspace.prompts import all_pairs


def top_collision_pairs(sweep_dir: str, layer: int, top: int) -> list[str]:
    with (pathlib.Path(sweep_dir) / "records.jsonl").open() as records_file:
        recs = [json.loads(line) for line in records_file]
    cands = [
        r
        for r in recs
        if r["lens"] == "jlens"
        and r["layer"] == layer
        and r["answer_flip"]
        and r["correct_a"]
        and r["correct_b"]
    ]
    cands.sort(key=lambda r: -r["collision_score"])
    seen, out = set(), []
    for r in cands:
        if r["pair_id"] not in seen:
            seen.add(r["pair_id"])
            out.append(r["pair_id"])
        if len(out) == top:
            break
    return out


def lens_visible_projector(J: torch.Tensor, mass: float = 0.90) -> np.ndarray:
    """Projector onto the top right-singular subspace holding `mass` of
    squared singular mass — the directions the lens transport amplifies."""
    _, S, Vh = np.linalg.svd(J.float().numpy(), full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    r = int(np.searchsorted(cum, mass) + 1)
    V = Vh[:r].T  # (d, r)
    return V @ V.T, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="raw")
    ap.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    ap.add_argument("--sweep", default="results/e02_collision_search/sweep_final")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="explicit pair IDs; overrides --top selection from the sweep",
    )
    ap.add_argument("--mass", type=float, default=0.90)
    ap.add_argument("--out", default="results/e02_collision_search/patch_confirm_v1")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = ModelWrapper(
        args.model,
        device=args.device,
        dtype=args.dtype,
        prompt_format=args.prompt_format,
    )
    lens = jlens.JacobianLens.load(args.lens)
    P_vis, rank = lens_visible_projector(lens.jacobians[args.layer], args.mass)
    print(f"lens-visible subspace at L{args.layer}: rank {rank}/{wrapper.d_model} "
          f"({args.mass:.0%} sq-singular mass)")
    P_vis_t = torch.from_numpy(P_vis.astype(np.float32))

    pairs = {p.pair_id: p for p in all_pairs()}
    pair_ids = args.pairs or top_collision_pairs(args.sweep, args.layer, args.top)
    unknown = set(pair_ids) - set(pairs)
    if unknown:
        raise ValueError(f"unknown pair IDs: {sorted(unknown)}")
    L = args.layer

    records = []
    for pid in pair_ids:
        pair = pairs[pid]
        n_a = wrapper.encode(pair.prompt_a).shape[1]
        n_b = wrapper.encode(pair.prompt_b).shape[1]
        if n_a != n_b:
            print(f"skip {pid}: statement token counts differ ({n_a} vs {n_b})")
            continue
        answers = [pair.expected_a, pair.expected_b]
        cap = {
            "a": wrapper.capture(pair.full_a(), layers=[L]),
            "b": wrapper.capture(pair.full_b(), layers=[L]),
        }
        stmt_positions = list(range(n_a))

        def patches_for(
            direction: str,
            condition: str,
            cap=cap,
            stmt_positions=stmt_positions,
        ) -> list[ResidualPatch]:
            src, dst = ("a", "b") if direction == "a->b" else ("b", "a")
            h_src = cap[src].activations[L]
            h_dst = cap[dst].activations[L]
            if condition == "none":
                return []
            if condition == "full_last":
                pos = stmt_positions[-1]
                return [ResidualPatch(L, pos, h_src[pos].clone())]
            vecs = {}
            for pos in stmt_positions:
                delta = h_src[pos] - h_dst[pos]
                if condition == "full_stmt":
                    vecs[pos] = h_src[pos].clone()
                elif condition == "jvis_stmt":
                    vecs[pos] = h_dst[pos] + P_vis_t @ delta
                elif condition == "fiber_stmt":
                    vecs[pos] = h_dst[pos] + (delta - P_vis_t @ delta)
                else:
                    raise ValueError(condition)
            return [ResidualPatch(L, pos, v) for pos, v in vecs.items()]

        for direction in ("a->b", "b->a"):
            dst = "b" if direction == "a->b" else "a"
            dst_prompt = pair.full_b() if dst == "b" else pair.full_a()
            src_dist = wrapper.answer_distribution(
                pair.full_a() if dst == "b" else pair.full_b(), answers
            )
            for condition in (
                "none",
                "full_last",
                "full_stmt",
                "jvis_stmt",
                "fiber_stmt",
            ):
                dist = wrapper.answer_distribution(
                    dst_prompt, answers, patches=patches_for(direction, condition)
                )
                records.append(
                    {
                        "pair_id": pid,
                        "direction": direction,
                        "condition": condition,
                        "dist": dist.tolist(),
                        "js_to_source_behavior": js_divergence(dist, src_dist),
                        "answer": answers[int(np.argmax(dist))],
                    }
                )
        print(f"{pid}: done")

    with (out_dir / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # summary: behavior transfer per condition, averaged over pairs+directions
    lines = ["# Patch confirmation summary", "",
             f"layer L{L}, lens-visible rank {rank}/{wrapper.d_model} at {args.mass:.0%} mass",
             "",
             (
                 "transfer = 1 - JS(patched dist, source-variant dist)"
                 "/JS(baseline dist, source-variant dist)"
             ),
             "(1.0 = behavior fully follows the patch; 0.0 = no effect)", ""]
    lines.append("| condition | mean transfer | flips to source answer |")
    lines.append("|---|---|---|")
    base = {(r["pair_id"], r["direction"]): r["js_to_source_behavior"]
            for r in records if r["condition"] == "none"}
    src_answer = {}
    for pid in {r["pair_id"] for r in records}:
        pair = pairs[pid]
        src_answer[(pid, "a->b")] = pair.expected_a
        src_answer[(pid, "b->a")] = pair.expected_b
    for condition in ("full_last", "full_stmt", "jvis_stmt", "fiber_stmt"):
        rs = [r for r in records if r["condition"] == condition]
        transfers = [
            1
            - r["js_to_source_behavior"]
            / max(base[(r["pair_id"], r["direction"])], 1e-9)
            for r in rs
        ]
        flips = [r["answer"] == src_answer[(r["pair_id"], r["direction"])] for r in rs]
        lines.append(
            f"| {condition} | {np.mean(transfers):.2f} | "
            f"{np.mean(flips):.0%} ({sum(flips)}/{len(flips)}) |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
