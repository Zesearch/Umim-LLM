#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${DATA_DIR:?Set DATA_DIR to the prepared DeepScaleR data directory}"

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
NUM_GPUS="${NUM_GPUS:-8}"
EPOCHS="${EPOCHS:-15}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

ARGS=(
  --train_data_path "$DATA_DIR/input_ids_for_train.pt"
  --val_data_path "$DATA_DIR/input_ids_for_val.pt"
  --test_data_path "$DATA_DIR/input_ids_for_test.pt"
  --train_attention_mask_path "$DATA_DIR/attention_mask_for_train.pt"
  --val_attention_mask_path "$DATA_DIR/attention_mask_for_val.pt"
  --test_attention_mask_path "$DATA_DIR/attention_mask_for_test.pt"
  --train_merge_indices_path "$DATA_DIR/merge_indices_for_train.pkl"
  --val_merge_indices_path "$DATA_DIR/merge_indices_for_val.pkl"
  --test_merge_indices_path "$DATA_DIR/merge_indices_for_test.pkl"
  --batch_size 3
  --lr 8e-4
  --epochs "$EPOCHS"
  --max_running_time 28800
  --save_dir "$OUTPUT_DIR"
  --wandb_project UMIM-DeepScaleR
  --wandb_run_name base-threshold-5
  --log_interval 100
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

torchrun --nproc_per_node="$NUM_GPUS" train.py "${ARGS[@]}"
