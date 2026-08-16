#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../common/train.py"
EXTRACT_SCRIPT="${SCRIPT_DIR}/../common/extract_merge_module.py"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
SFT_CHECKPOINT_ROOT="${SFT_CHECKPOINT_ROOT:-${SCRIPT_DIR}/../../sft/hellaswag/checkpoints}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${SCRIPT_DIR}/weights}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints}"
GPU_IDS="${GPU_IDS:-localhost:0,1,2,3,4,5,6,7}"

mkdir -p "${WEIGHT_ROOT}"

for THRESHOLD in 2000 200 50 5 1; do
  DATA_DIR="${DATA_ROOT}/threshold_${THRESHOLD}"
  SFT_CHECKPOINT="${SFT_CHECKPOINT_ROOT}/threshold_${THRESHOLD}"
  SFT_WEIGHTS="${WEIGHT_ROOT}/sft_threshold_${THRESHOLD}.pt"
  CHECKPOINT_NAME="threshold_${THRESHOLD}"

  python "${EXTRACT_SCRIPT}" \
    --checkpoint_dir "${SFT_CHECKPOINT}" \
    --tag best \
    --output "${SFT_WEIGHTS}"

  deepspeed --include "${GPU_IDS}" "${TRAIN_SCRIPT}" \
    --pref_train_data_path "${DATA_DIR}/preferred_input_ids_train.pt" \
    --pref_train_attention_mask_path "${DATA_DIR}/preferred_attention_mask_train.pt" \
    --pref_train_merge_indices_path "${DATA_DIR}/preferred_merge_indices_train.pkl" \
    --pref_train_ending_start_path "${DATA_DIR}/preferred_ending_start_train.pt" \
    --rej_train_data_path "${DATA_DIR}/rejected_input_ids_train.pt" \
    --rej_train_attention_mask_path "${DATA_DIR}/rejected_attention_mask_train.pt" \
    --rej_train_merge_indices_path "${DATA_DIR}/rejected_merge_indices_train.pkl" \
    --rej_train_ending_start_path "${DATA_DIR}/rejected_ending_start_train.pt" \
    --train_group_ids_path "${DATA_DIR}/group_ids_train.pt" \
    --pref_val_data_path "${DATA_DIR}/preferred_input_ids_val.pt" \
    --pref_val_attention_mask_path "${DATA_DIR}/preferred_attention_mask_val.pt" \
    --pref_val_merge_indices_path "${DATA_DIR}/preferred_merge_indices_val.pkl" \
    --pref_val_ending_start_path "${DATA_DIR}/preferred_ending_start_val.pt" \
    --rej_val_data_path "${DATA_DIR}/rejected_input_ids_val.pt" \
    --rej_val_attention_mask_path "${DATA_DIR}/rejected_attention_mask_val.pt" \
    --rej_val_merge_indices_path "${DATA_DIR}/rejected_merge_indices_val.pkl" \
    --rej_val_ending_start_path "${DATA_DIR}/rejected_ending_start_val.pt" \
    --val_group_ids_path "${DATA_DIR}/group_ids_val.pt" \
    --model_path "${MODEL_NAME_OR_PATH}" \
    --pretrained_merge_weights "${SFT_WEIGHTS}" \
    --save_dir "${CHECKPOINT_ROOT}" \
    --ckpt_subdir "${CHECKPOINT_NAME}" \
    --ckpt_tag best \
    --resume_from_checkpoint "${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}" \
    --batch_size 32 \
    --lr 5e-5 \
    --beta 0.1 \
    --epochs 2 \
    --patience 2 \
    --log_interval 10
done
