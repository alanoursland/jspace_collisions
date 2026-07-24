"""Behavior-only competence gate for raw versus chat-formatted checkpoints.

This isolates whether a model can perform the E2 probes before lens fitting
or collision rates are interpreted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict

import numpy as np

from jspace.metrics import js_divergence
from jspace.models.wrapper import ModelWrapper
from jspace.prompts import all_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--prompt-format", choices=["raw", "chat"], default="raw")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "records.jsonl"

    wrapper = ModelWrapper(
        args.model,
        device=args.device,
        dtype=args.dtype,
        prompt_format=args.prompt_format,
    )
    pairs = all_pairs()
    if args.limit is not None:
        pairs = pairs[: args.limit]

    records = []
    started = time.time()
    for index, pair in enumerate(pairs, start=1):
        answers = [pair.expected_a, pair.expected_b]
        distribution_a = wrapper.answer_distribution(pair.full_a(), answers)
        distribution_b = wrapper.answer_distribution(pair.full_b(), answers)
        records.append(
            {
                "pair_id": pair.pair_id,
                "category": pair.category,
                "behavior_js": js_divergence(distribution_a, distribution_b),
                "answer_a_dist": distribution_a.tolist(),
                "answer_b_dist": distribution_b.tolist(),
                "answer_flip": bool(np.argmax(distribution_a) != np.argmax(distribution_b)),
                "correct_a": bool(np.argmax(distribution_a) == 0),
                "correct_b": bool(np.argmax(distribution_b) == 1),
            }
        )
        if index % 10 == 0 or index == len(pairs):
            print(f"{index}/{len(pairs)} pairs ({time.time() - started:.0f}s)")

    with output_file.open("w") as record_sink:
        for record in records:
            record_sink.write(json.dumps(record) + "\n")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    lines = [
        "# E11 competence control",
        "",
        f"model: {args.model}",
        f"device: {wrapper.device}",
        f"dtype: {wrapper.dtype}",
        f"prompt format: {wrapper.prompt_format}",
        "",
        "| category | pairs | both-correct | answer-flip | mean behavior JS |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, category_records in sorted(by_category.items()):
        both_correct = np.mean(
            [
                record["correct_a"] and record["correct_b"]
                for record in category_records
            ]
        )
        answer_flip = np.mean(
            [record["answer_flip"] for record in category_records]
        )
        mean_behavior_js = np.mean(
            [record["behavior_js"] for record in category_records]
        )
        lines.append(
            f"| {category} | {len(category_records)} | {both_correct:.0%} | "
            f"{answer_flip:.0%} | {mean_behavior_js:.3f} |"
        )

    report = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
