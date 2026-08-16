#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
RULES_ROOT="${RULES_ROOT:-${SCRIPT_DIR}/../../sft/hellaswag/data}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"

python "${SCRIPT_DIR}/prepare_data.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --rules_root "${RULES_ROOT}" \
  --output_dir "${DATA_ROOT}" \
  --thresholds 2000 200 50 5 1 \
  --max_length 128
