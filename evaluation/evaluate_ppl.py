from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import add_runtime_arguments, iter_jsonl, load_runtime, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size < 2:
        raise ValueError("block-size must be at least 2")

    runtime = load_runtime(args)
    total_nll = 0.0
    total_targets = 0
    total_original = 0
    total_effective = 0
    samples = 0

    for row in iter_jsonl(args.data):
        if args.max_samples is not None and samples >= args.max_samples:
            break
        text = str(row[args.text_field])
        if not text.strip():
            continue
        token_ids = runtime.tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, max(len(token_ids) - 1, 0), args.block_size - 1):
            chunk = token_ids[start : start + args.block_size]
            if len(chunk) < 2:
                continue
            result = runtime.score_perplexity_tokens(chunk)
            total_nll += float(result["total_nll"])
            total_targets += int(result["target_tokens"])
            total_original += int(result["original_tokens"])
            total_effective += int(result["effective_tokens"])
        samples += 1
        print(f"[{samples}] targets={total_targets}", flush=True)

    removed = total_original - total_effective
    summary = {
        "samples": samples,
        "perplexity": math.exp(total_nll / total_targets) if total_targets else None,
        "total_nll": total_nll,
        "target_tokens": total_targets,
        "original_tokens": total_original,
        "effective_tokens": total_effective,
        "token_reduction_percent": (
            100.0 * removed / total_original if total_original else 0.0
        ),
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
