from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import add_runtime_arguments, iter_jsonl, load_runtime, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    samples = 0
    original_tokens = 0
    effective_tokens = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_jsonl(args.data)):
            if args.max_samples is not None and samples >= args.max_samples:
                break
            candidates = [str(value) for value in row["candidates"]]
            label = int(row["label"])
            if not 0 <= label < len(candidates):
                raise ValueError(f"Invalid label in example {index}")

            scores = runtime.score_candidates(candidates)
            losses = [float(score["loss"]) for score in scores]
            prediction = min(range(len(losses)), key=losses.__getitem__)
            is_correct = prediction == label
            correct += int(is_correct)
            samples += 1
            original_tokens += sum(int(score["original_tokens"]) for score in scores)
            effective_tokens += sum(int(score["effective_tokens"]) for score in scores)

            result = {
                "id": row.get("id", index),
                "label": label,
                "prediction": prediction,
                "correct": is_correct,
                "losses": losses,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{samples}] accuracy={correct / samples:.4f}", flush=True)

    removed = original_tokens - effective_tokens
    summary = {
        "samples": samples,
        "accuracy": correct / samples if samples else 0.0,
        "correct": correct,
        "original_tokens_across_candidates": original_tokens,
        "effective_tokens_across_candidates": effective_tokens,
        "token_reduction_percent": (
            100.0 * removed / original_tokens if original_tokens else 0.0
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
