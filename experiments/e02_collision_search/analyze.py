"""Summarize an E2 sweep: collision candidates and control calibration.

    python experiments/e02_collision_search/analyze.py results/e02_collision_search/sweep_v1
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np


def load(out_dir: str) -> list[dict]:
    path = pathlib.Path(out_dir) / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    out_dir = sys.argv[1]
    records = load(out_dir)
    lenses = sorted({r["lens"] for r in records})
    layers = sorted({r["layer"] for r in records})
    jlens_name = next(l for l in lenses if l == "jlens")

    lines = ["# E2 sweep summary", ""]

    # 1. J-distance calibration: real lens vs controls, per layer
    lines += ["## Mean same-pair J-distance (JS) by lens and layer", ""]
    lines.append("| lens | " + " | ".join(f"L{l}" for l in layers) + " |")
    lines.append("|---" * (len(layers) + 1) + "|")
    for lens in lenses:
        row = [lens]
        for layer in layers:
            vals = [r["js"] for r in records if r["lens"] == lens and r["layer"] == layer]
            row.append(fmt(float(np.mean(vals))))
        lines.append("| " + " | ".join(row) + " |")

    # 2. Task competence: does the 0.5B model even do the tasks?
    lines += ["", "## Behavior probe competence by category", ""]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for r in records:
        if r["pair_id"] not in seen:
            seen.add(r["pair_id"])
            by_cat[r["category"]].append(r)
    lines.append("| category | pairs | both-correct | answer-flip rate | mean behavior JS |")
    lines.append("|---|---|---|---|---|")
    for cat, rs in sorted(by_cat.items()):
        both = np.mean([r["correct_a"] and r["correct_b"] for r in rs])
        flip = np.mean([r["answer_flip"] for r in rs])
        bjs = np.mean([r["behavior_js"] for r in rs])
        lines.append(f"| {cat} | {len(rs)} | {both:.0%} | {flip:.0%} | {fmt(float(bjs))} |")

    # 3. Top collision candidates under the real lens
    lines += ["", "## Top collision candidates (jlens, ranked by collision_score)", ""]
    jrecs = [r for r in records if r["lens"] == jlens_name]
    top = sorted(jrecs, key=lambda r: -r["collision_score"])[:20]
    lines.append("| pair | layer | J-dist (JS) | top20 jaccard | behavior JS | flip |")
    lines.append("|---|---|---|---|---|---|")
    for r in top:
        lines.append(
            f"| {r['pair_id']} | {r['layer']} | {fmt(r['js'])} | "
            f"{fmt(r['top20_jaccard'])} | {fmt(r['behavior_js'])} | {r['answer_flip']} |"
        )

    # 4. Category x layer collision summary: mean J-distance among
    #    behaviorally-divergent pairs (the H2 signal)
    lines += ["", "## Mean J-dist (JS) for pairs with behavior JS > 0.1 (jlens)", ""]
    lines.append("| category | " + " | ".join(f"L{l}" for l in layers) + " |")
    lines.append("|---" * (len(layers) + 1) + "|")
    cats = sorted({r["category"] for r in jrecs})
    for cat in cats:
        row = [cat]
        for layer in layers:
            vals = [
                r["js"]
                for r in jrecs
                if r["category"] == cat and r["layer"] == layer and r["behavior_js"] > 0.1
            ]
            row.append(fmt(float(np.mean(vals))) if vals else "-")
        lines.append("| " + " | ".join(row) + " |")

    report = "\n".join(lines) + "\n"
    out = pathlib.Path(out_dir) / "summary.md"
    out.write_text(report)
    print(report)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
