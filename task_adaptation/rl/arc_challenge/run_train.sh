#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../common/train.py"
EXTRACT_SCRIPT="${SCRIPT_DIR}/../common/extract_merge_module.py"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data/threshold_5}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${SCRIPT_DIR}/../../sft/arc_challenge/checkpoints/arc_challenge}"
SFT_WEIGHTS="${SFT_WEIGHTS:-${SCRIPT_DIR}/weights/sft_merge_module.pt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints}"
GPU_IDS="${GPU_IDS:-localhost:0,1,2,3,4,5,6,7}"

python "${EXTRACT_SCRIPT}" \
  --checkpoint_dir "${SFT_CHECKPOINT}" \
  --tag best \
  --output "${SFT_WEIGHTS}"

deepspeed --include "${GPU_IDS}" "${TRAIN_SCRIPT}" \
  --pref_train_data_path "${DATA_ROOT}/preferred_input_ids_train.pt" \
  --pref_train_attention_mask_path "${DATA_ROOT}/preferred_attention_mask_train.pt" \
  --pref_train_merge_indices_path "${DATA_ROOT}/preferred_merge_indices_train.pkl" \
  --pref_train_ending_start_path "${DATA_ROOT}/preferred_ending_start_train.pt" \
  --rej_train_data_path "${DATA_ROOT}/rejected_input_ids_train.pt" \
  --rej_train_attention_mask_path "${DATA_ROOT}/rejected_attention_mask_train.pt" \
  --rej_train_merge_indices_path "${DATA_ROOT}/rejected_merge_indices_train.pkl" \
  --rej_train_ending_start_path "${DATA_ROOT}/rejected_ending_start_train.pt" \
  --train_group_ids_path "${DATA_ROOT}/group_ids_train.pt" \
  --pref_val_data_path "${DATA_ROOT}/preferred_input_ids_val.pt" \
  --pref_val_attention_mask_path "${DATA_ROOT}/preferred_attention_mask_val.pt" \
  --pref_val_merge_indices_path "${DATA_ROOT}/preferred_merge_indices_val.pkl" \
  --pref_val_ending_start_path "${DATA_ROOT}/preferred_ending_start_val.pt" \
  --rej_val_data_path "${DATA_ROOT}/rejected_input_ids_val.pt" \
  --rej_val_attention_mask_path "${DATA_ROOT}/rejected_attention_mask_val.pt" \
  --rej_val_merge_indices_path "${DATA_ROOT}/rejected_merge_indices_val.pkl" \
  --rej_val_ending_start_path "${DATA_ROOT}/rejected_ending_start_val.pt" \
  --val_group_ids_path "${DATA_ROOT}/group_ids_val.pt" \
  --model_path "${MODEL_NAME_OR_PATH}" \
  --pretrained_merge_weights "${SFT_WEIGHTS}" \
  --save_dir "${CHECKPOINT_ROOT}" \
  --ckpt_subdir arc_challenge \
  --ckpt_tag best \
  --resume_from_checkpoint "${CHECKPOINT_ROOT}/arc_challenge" \
  --filter_empty_merges \
  --batch_size 2 \
  --lr 5e-4 \
  --beta 0.1 \
  --epochs 10 \
  --patience 3 \
  --log_interval 5
