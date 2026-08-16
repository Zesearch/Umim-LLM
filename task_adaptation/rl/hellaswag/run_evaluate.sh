#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
RULES_ROOT="${RULES_ROOT:-${SCRIPT_DIR}/../../sft/hellaswag/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints}"
DEVICE="${DEVICE:-cuda:0}"

for THRESHOLD in 2000 200 50 5 1; do
  python "${SCRIPT_DIR}/evaluate.py" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --merge_weights "${CHECKPOINT_ROOT}/threshold_${THRESHOLD}/merge_module_best.pt" \
    --rules_path "${RULES_ROOT}/threshold_${THRESHOLD}/filtered_bigrams_tensor.pt" \
    --device "${DEVICE}"
done
