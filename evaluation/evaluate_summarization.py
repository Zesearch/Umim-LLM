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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[str] = []
    references: list[str] = []
    prompt_original = 0
    prompt_effective = 0
    generated_kv = 0
    effective_generated_kv = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_jsonl(args.data)):
            if args.max_samples is not None and index >= args.max_samples:
                break
            result = runtime.generate(
                str(row["prompt"]),
                max_new_tokens=args.max_new_tokens,
            )
            prediction = str(result["prediction"])
            reference = str(row["reference"])
            metrics = result["metrics"]
            predictions.append(prediction)
            references.append(reference)
            prompt_original += int(metrics["original_prompt_tokens"])
            prompt_effective += int(metrics["effective_prompt_tokens"])
            generated_kv += int(metrics["generated_tokens_entering_kv"])
            effective_generated_kv += int(metrics["effective_generated_kv_tokens"])

            output_row = {
                "id": row.get("id", index),
                "prediction": prediction,
                "reference": reference,
                "metrics": metrics,
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{len(predictions)}] generated={metrics['generated_tokens']}", flush=True)

    try:
        import evaluate
    except ImportError as exc:
        raise SystemExit("Install evaluate to compute summarization metrics") from exc

    rouge = evaluate.load("rouge").compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    prompt_removed = prompt_original - prompt_effective
    generated_removed = generated_kv - effective_generated_kv
    summary = {
        "samples": len(predictions),
        "rouge": {key: float(value) for key, value in rouge.items()},
        "prompt_token_reduction_percent": (
            100.0 * prompt_removed / prompt_original if prompt_original else 0.0
        ),
        "generated_kv_reduction_percent": (
            100.0 * generated_removed / generated_kv if generated_kv else 0.0
        ),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
