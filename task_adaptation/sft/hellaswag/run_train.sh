#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../common/train.py"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints}"
GPU_IDS="${GPU_IDS:-localhost:0,1,2,3,4,5,6,7}"
: "${PRETRAINED_MERGE_WEIGHTS:?Set PRETRAINED_MERGE_WEIGHTS to the WikiText base merge-module checkpoint}"

for THRESHOLD in 2000 200 50 5 1; do
  DATA_DIR="${DATA_ROOT}/threshold_${THRESHOLD}"
  CHECKPOINT_NAME="threshold_${THRESHOLD}"

  deepspeed --include "${GPU_IDS}" "${TRAIN_SCRIPT}" \
    --train_data_path "${DATA_DIR}/input_ids_train.pt" \
    --val_data_path "${DATA_DIR}/input_ids_val.pt" \
    --test_data_path "${DATA_DIR}/input_ids_val.pt" \
    --train_attention_mask_path "${DATA_DIR}/attention_mask_train.pt" \
    --val_attention_mask_path "${DATA_DIR}/attention_mask_val.pt" \
    --test_attention_mask_path "${DATA_DIR}/attention_mask_val.pt" \
    --train_merge_indices_path "${DATA_DIR}/merge_indices_train.pkl" \
    --val_merge_indices_path "${DATA_DIR}/merge_indices_val.pkl" \
    --test_merge_indices_path "${DATA_DIR}/merge_indices_val.pkl" \
    --model_path "${MODEL_NAME_OR_PATH}" \
    --save_dir "${CHECKPOINT_ROOT}" \
    --ckpt_subdir "${CHECKPOINT_NAME}" \
    --ckpt_tag best \
    --pretrained_merge_weights "${PRETRAINED_MERGE_WEIGHTS}" \
    --resume_from_checkpoint "${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}" \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 5 \
    --patience 3 \
    --loss_threshold 1e-6 \
    --log_interval 10
done
