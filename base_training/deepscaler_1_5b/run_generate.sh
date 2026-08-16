#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated reasoning trajectories}"

NUM_GPUS="${NUM_GPUS:-8}"

torchrun --nproc_per_node="$NUM_GPUS" generate_reasoning_corpus.py \
  --output_dir "$OUTPUT_DIR" \
  --num_return_sequences 3 \
  --model_name agentica-org/DeepScaleR-1.5B-Preview
