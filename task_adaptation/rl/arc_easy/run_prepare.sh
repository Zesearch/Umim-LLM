#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
RULES_PATH="${RULES_PATH:-${SCRIPT_DIR}/../../sft/arc_easy/data/threshold_5/filtered_bigrams_tensor.pt}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data/threshold_5}"

python "${SCRIPT_DIR}/prepare_data.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --rules_path "${RULES_PATH}" \
  --output_dir "${DATA_ROOT}" \
  --max_length 128
