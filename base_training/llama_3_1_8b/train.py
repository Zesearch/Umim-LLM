import os
import sys
import time
import pickle
import random

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from model import CustomGPT2WithCompression, merge_tokens, MultiHeadPooling
from utils import (
    compute_cross_entropy_loss,
    compute_top1_accuracy,
    compute_top3_overlap,
    compute_top10_overlap,
    compute_top_p_metric,
    compute_mrr,
    save_checkpoint,
    save_checkpoint_deepspeed,
    plot_all_metrics
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, no logs will be sent to Weights & Biases.")

from transformers import AutoModelForCausalLM


class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, merge_indices=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.merge_indices = merge_indices

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "merge_indices": (
                self.merge_indices[idx] if self.merge_indices is not None else None
            )
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    merge_indices = [item["merge_indices"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "merge_indices": merge_indices
    }


@torch.no_grad()
def evaluate_model(teacher_model, student_engine, dataloader, device):

    teacher_model.eval()
    student_engine.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top10 = 0.0
    total_topp = 0.0
    total_mrr = 0.0
    count = 0

    merged_sum = 0
    sample_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        merge_indices_batch = batch["merge_indices"]

        outputs = teacher_model(input_ids, attention_mask=attention_mask)
        original_logits = outputs.logits
        original_probs = F.softmax(original_logits, dim=-1)

        compressed_logits, padding_mask, new_attention_mask, lossfunction_mask = \
            student_engine.module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                merge_indices=merge_indices_batch
            )
        compressed_probs = F.softmax(compressed_logits, dim=-1)


        loss, adjusted_mask = compute_cross_entropy_loss(
            compressed_probs,
            original_probs,
            lossfunction_mask,
            new_attention_mask
        )

        batch_size = input_ids.size(0)
        top1_b = compute_top1_accuracy(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)
        top3_b = compute_top3_overlap(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)
        top10_b = compute_top10_overlap(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)
        topp_b = compute_top_p_metric(original_probs, compressed_probs, lossfunction_mask, adjusted_mask, p=0.9)
        mrr_b = compute_mrr(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)

        total_loss += loss.item() * batch_size
        total_top1 += top1_b * batch_size
        total_top3 += top3_b * batch_size
        total_top10 += top10_b * batch_size
        total_topp += topp_b * batch_size
        total_mrr += mrr_b * batch_size
        count += batch_size


        for i in range(batch_size):
            pair_indices = merge_indices_batch[i]
            if torch.is_tensor(pair_indices):
                pair_indices = pair_indices.cpu().tolist()
            unique_pairs_count = len(set(pair_indices))
            merged_sum += unique_pairs_count
        sample_count += batch_size


    device_loss = torch.tensor([total_loss], dtype=torch.float, device=device)
    device_top1 = torch.tensor([total_top1], dtype=torch.float, device=device)
    device_top3 = torch.tensor([total_top3], dtype=torch.float, device=device)
    device_top10 = torch.tensor([total_top10], dtype=torch.float, device=device)
    device_topp = torch.tensor([total_topp], dtype=torch.float, device=device)
    device_mrr = torch.tensor([total_mrr], dtype=torch.float, device=device)
    device_count = torch.tensor([count], dtype=torch.float, device=device)
    device_merged_sum = torch.tensor([merged_sum], dtype=torch.float, device=device)
    device_sample_count = torch.tensor([sample_count], dtype=torch.float, device=device)


    stats_tensor = torch.stack([
            device_loss, device_top1, device_top3, device_top10,
            device_topp, device_mrr, device_count,
            device_merged_sum, device_sample_count
    ])
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    device_loss, device_top1, device_top3, device_top10, device_topp, device_mrr, device_count, device_merged_sum, device_sample_count = stats_tensor


    rank = dist.get_rank()
    if rank == 0:
        global_count = device_count.item()
        if global_count > 0:
            avg_loss = device_loss.item() / global_count
            avg_top1 = device_top1.item() / global_count
            avg_top3 = device_top3.item() / global_count
            avg_top10 = device_top10.item() / global_count
            avg_topp = device_topp.item() / global_count
            avg_mrr = device_mrr.item() / global_count
        else:
            avg_loss, avg_top1, avg_top3, avg_top10, avg_topp, avg_mrr = [0.0]*6

        if device_sample_count.item() > 0:
            avg_merged_pairs = device_merged_sum.item() / device_sample_count.item()
        else:
            avg_merged_pairs = 0.0

        return (avg_loss, avg_top1, avg_top3, avg_top10, avg_topp, avg_mrr, avg_merged_pairs)
    else:
        return (None, None, None, None, None, None, None)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--train_attention_mask_path", type=str, required=True)
    parser.add_argument("--val_attention_mask_path", type=str, required=True)
    parser.add_argument("--test_attention_mask_path", type=str, required=True)
    parser.add_argument("--train_merge_indices_path", type=str, required=True)
    parser.add_argument("--val_merge_indices_path", type=str, required=True)
    parser.add_argument("--test_merge_indices_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--loss_threshold", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_running_time", type=int, default=28800)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--wandb_project", type=str, default="MyDistProject")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()


    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    start_time = time.time()
    MAX_TIME = args.max_running_time


    train_input_ids = torch.load(args.train_data_path)
    train_attention_mask = torch.load(args.train_attention_mask_path)
    with open(args.train_merge_indices_path, 'rb') as f:
        merge_indices_for_train = pickle.load(f)

    val_input_ids = torch.load(args.val_data_path)
    val_attention_mask = torch.load(args.val_attention_mask_path)
    with open(args.val_merge_indices_path, 'rb') as f:
        merge_indices_for_val = pickle.load(f)

    test_input_ids = torch.load(args.test_data_path)
    test_attention_mask = torch.load(args.test_attention_mask_path)
    with open(args.test_merge_indices_path, 'rb') as f:
        merge_indices_for_test = pickle.load(f)

    train_dataset = TextDataset(train_input_ids, train_attention_mask, merge_indices_for_train)
    val_dataset = TextDataset(val_input_ids, val_attention_mask, merge_indices_for_val)
    test_dataset = TextDataset(test_input_ids, test_attention_mask, merge_indices_for_test)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=custom_collate_fn,drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, collate_fn=custom_collate_fn,drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 sampler=test_sampler, collate_fn=custom_collate_fn,drop_last=True)


    teacher_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to(device,dtype=torch.bfloat16)


    for param in teacher_model.parameters():
        param.requires_grad = False


    merge_module = merge_tokens()
    student_model = CustomGPT2WithCompression(merge_module, teacher_model).to(device,dtype=torch.bfloat16)

    for param in student_model.parameters():
        param.requires_grad = False
    for param in student_model.merge_module.parameters():
        param.requires_grad = True

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,


        "bf16": {
             "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 3e7,
            "allgather_bucket_size": 3e7 ,
            "round_robin_gradients": True,


        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": warmup_steps,
                "last_batch_iteration": -1
            }
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "gradient_predivide_factor": 1.0,
        "communication_data_type": "bf16",
        "fp16_auto_cast": False,
        "memory_efficient_linear": False,
         "steps_per_print": 50000,
         "silent": True,
         "wall_clock_breakdown": False,
         "dump_state": True,

    }

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=student_model,
        model_parameters=(p for p in student_model.parameters() if p.requires_grad),
        config=ds_config
    )


    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    wandb_run_id = None

    def attempt_deepspeed_load(model_engine, load_dir, load_tag):

        success, client_state = model_engine.load_checkpoint(load_dir, tag=load_tag)
        return success, client_state


    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint) and os.path.isdir(os.path.join(args.resume_from_checkpoint, "llama8B")):
        ckpt_dir = args.resume_from_checkpoint
        tag_name = "llama8B"
        success, client_state = attempt_deepspeed_load(model_engine, ckpt_dir, tag_name)
        if success and client_state is not None:
            start_epoch = client_state.get("epoch", 0) + 1
            global_step = client_state.get("global_step", 0)
            best_val_loss = client_state.get("best_val_loss", float('inf'))
            wandb_run_id = client_state.get("wandb_run_id", None)

            if rank == 0:
                print(f"[Resume] epoch={start_epoch}, global_step={global_step}, best_val={best_val_loss}")

        else:
            if rank == 0:
                print("[Resume] No valid checkpoint found or load failed. Start from scratch.")
    else:
        if rank == 0:
            print("No resume checkpoint. Start from scratch.")

    if rank == 0 and WANDB_AVAILABLE:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args)
        }
        if wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        new_id = wandb.run.id
    else:
        new_id = None


    if rank == 0:
        print("DeepSpeed Engine init done. Start training ...")


    epoch_losses = []
    top1_means_per_epoch = []
    top3_means_per_epoch = []
    top10_means_per_epoch = []
    topp_means_per_epoch = []
    mrr_means_per_epoch = []
    merged_count_per_epoch = []

    val_loss_history = []
    val_top1_history = []
    val_top3_history = []
    val_top10_history = []
    val_topp_history = []
    val_mrr_history = []

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)

        model_engine.train()
        teacher_model.eval()

        if rank == 0:
            print(f"\n=== Starting Epoch {epoch + 1}/{args.epochs} ===")

        epoch_loss = 0.0
        top1_sum = 0.0
        top3_sum = 0.0
        top10_sum = 0.0
        topp_sum = 0.0
        mrr_sum = 0.0
        count = 0

        train_merged_sum = 0
        train_sample_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            merge_indices_batch = batch["merge_indices"]


            with torch.no_grad():
                outputs = teacher_model(input_ids, attention_mask=attention_mask)
                original_logits = outputs.logits
                original_probs = F.softmax(original_logits, dim=-1)


            compressed_logits, padding_mask, new_attention_mask, lossfunction_mask = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                merge_indices=merge_indices_batch
            )
            compressed_probs = F.softmax(compressed_logits, dim=-1)


            loss, adjusted_mask = compute_cross_entropy_loss(
                compressed_probs,
                original_probs,
                lossfunction_mask,
                new_attention_mask
            )

            batch_size = input_ids.size(0)
            top1_b = compute_top1_accuracy(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)
            top3_b = compute_top3_overlap(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)
            top10_b = compute_top10_overlap(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)
            topp_b = compute_top_p_metric(original_probs, compressed_probs, lossfunction_mask, adjusted_mask, p=0.9)
            mrr_b = compute_mrr(original_probs, compressed_probs, lossfunction_mask, adjusted_mask)

            epoch_loss += loss.item() * batch_size
            top1_sum += top1_b * batch_size
            top3_sum += top3_b * batch_size
            top10_sum += top10_b * batch_size
            topp_sum += topp_b * batch_size
            mrr_sum += mrr_b * batch_size
            count += batch_size


            for i in range(batch_size):
                pair_indices = merge_indices_batch[i]
                if torch.is_tensor(pair_indices):
                    pair_indices = pair_indices.cpu().tolist()
                unique_pairs_count = len(set(pair_indices))
                train_merged_sum += unique_pairs_count
            train_sample_count += batch_size


            model_engine.backward(loss)
            model_engine.step()

            global_step += 1

            if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}: "
                      f"Loss={loss.item():.4f}, Top1={top1_b:.4f}")
                if WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/top1_batch": top1_b,
                        "train/top3_batch": top3_b,
                        "train/top10_batch": top10_b,
                        "train/top_p_batch": topp_b,
                        "train/mrr_batch": mrr_b,
                        "epoch": epoch,
                        "global_step": global_step
                    })


        epoch_loss_t = torch.tensor([epoch_loss], dtype=torch.float, device=device)
        top1_sum_t   = torch.tensor([top1_sum],   dtype=torch.float, device=device)
        top3_sum_t   = torch.tensor([top3_sum],   dtype=torch.float, device=device)
        top10_sum_t  = torch.tensor([top10_sum],  dtype=torch.float, device=device)
        topp_sum_t   = torch.tensor([topp_sum],   dtype=torch.float, device=device)
        mrr_sum_t    = torch.tensor([mrr_sum],    dtype=torch.float, device=device)
        count_t      = torch.tensor([count],      dtype=torch.float, device=device)
        train_merged_sum_t   = torch.tensor([train_merged_sum], dtype=torch.float, device=device)
        train_sample_count_t = torch.tensor([train_sample_count], dtype=torch.float, device=device)

        stats_tensor = torch.stack([
                         epoch_loss_t,
                         top1_sum_t,
                         top3_sum_t,
                         top10_sum_t,
                         topp_sum_t,
                         mrr_sum_t,
                         count_t,
                         train_merged_sum_t,
                         train_sample_count_t
                        ])
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_t, top1_sum_t, top3_sum_t, top10_sum_t, topp_sum_t,mrr_sum_t, count_t, train_merged_sum_t, train_sample_count_t = stats_tensor


        if rank == 0:
            global_count = count_t.item()
            if global_count > 0:
                epoch_loss_avg = epoch_loss_t.item() / global_count
                avg_top1 = top1_sum_t.item() / global_count
                avg_top3 = top3_sum_t.item() / global_count
                avg_top10 = top10_sum_t.item() / global_count
                avg_topp = topp_sum_t.item() / global_count
                avg_mrr = mrr_sum_t.item() / global_count
            else:
                epoch_loss_avg, avg_top1, avg_top3, avg_top10, avg_topp, avg_mrr = [0.0]*6

            merged_train = 0.0
            if train_sample_count_t.item() > 0:
                merged_train = train_merged_sum_t.item() / train_sample_count_t.item()

            epoch_losses.append(epoch_loss_avg)
            top1_means_per_epoch.append(avg_top1)
            top3_means_per_epoch.append(avg_top3)
            top10_means_per_epoch.append(avg_top10)
            topp_means_per_epoch.append(avg_topp)
            mrr_means_per_epoch.append(avg_mrr)
            merged_count_per_epoch.append(merged_train)

            print(f"[Epoch {epoch+1} Train] "
                  f"Loss: {epoch_loss_avg:.4f} | Top-1: {avg_top1:.4f} | Top-3: {avg_top3:.4f} | "
                  f"Top-10: {avg_top10:.4f} | Top-p: {avg_topp:.4f} | MRR: {avg_mrr:.4f} | "
                  f"MergedPairs: {merged_train:.2f}")

            if WANDB_AVAILABLE:
                wandb.log({
                    "train/loss_epoch": epoch_loss_avg,
                    "train/top1_epoch": avg_top1,
                    "train/top3_epoch": avg_top3,
                    "train/top10_epoch": avg_top10,
                    "train/top_p_epoch": avg_topp,
                    "train/mrr_epoch": avg_mrr,
                    "train/avg_merged_pairs_epoch": merged_train,
                    "epoch": epoch
                })

        val_loss, val_top1, val_top3, val_top10, val_topp, val_mrr, val_merged_pairs = \
            evaluate_model(teacher_model, model_engine, val_dataloader, device)

        if rank == 0:
            val_loss_history.append(val_loss)
            val_top1_history.append(val_top1)
            val_top3_history.append(val_top3)
            val_top10_history.append(val_top10)
            val_topp_history.append(val_topp)
            val_mrr_history.append(val_mrr)

            print(f"[Epoch {epoch+1} Val] "
                  f"Loss: {val_loss:.4f} | Top-1: {val_top1:.4f} | Top-3: {val_top3:.4f} | "
                  f"Top-10: {val_top10:.4f} | Top-p: {val_topp:.4f} | MRR: {val_mrr:.4f} | "
                  f"AvgMerged: {val_merged_pairs:.4f}")

            if WANDB_AVAILABLE:
                wandb.log({
                    "val/loss": val_loss,
                    "val/top1": val_top1,
                    "val/top3": val_top3,
                    "val/top10": val_top10,
                    "val/top_p": val_topp,
                    "val/mrr": val_mrr,
                    "val/avg_merged_pairs": val_merged_pairs,
                    "epoch": epoch
                })

        save_this_epoch = False
        if val_loss is not None and (val_loss < best_val_loss):
            best_val_loss = val_loss
            patience_counter = 0
            save_this_epoch = True

        flag_tensor = torch.tensor([1 if save_this_epoch else 0], dtype=torch.int, device=device)
        dist.broadcast(flag_tensor, src=0)
        save_this_epoch = (flag_tensor.item() == 1)

        if save_this_epoch:
            client_state = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "global_step": global_step,
                    "wandb_run_id": wandb_run_id
            }
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, "deepseed5")
            model_engine.save_checkpoint(ckpt_path, tag="llama8B",client_state=client_state)

            if rank == 0:
                print(f"[Best Model Saved] Val Loss improved to {val_loss:.4f}")

            if rank == 0:

             epoch_file = os.path.join(args.save_dir, "llama_8B_progress_5.txt")
             with open(epoch_file, "w") as f:
               f.write(str(epoch))

        stop_training = False
        if rank == 0:
            if not save_this_epoch:
                patience_counter += 1
                if patience_counter > args.patience:
                    print(f"Early stopping triggered. No improvement for {args.patience} epochs.")
                    stop_training = True

            else:
                patience_counter=0


            if val_loss < args.loss_threshold:
                print(f"Stopping early because val_loss < {args.loss_threshold:.6f}")
                stop_training = True

        stop_training_tensor = torch.tensor([int(stop_training)], device=device)
        dist.broadcast(stop_training_tensor, src=0)
        if stop_training_tensor.item() == 1:
            break


        used_time = time.time() - start_time
        this_epoch_time = time.time() - epoch_start_time
        remaining = MAX_TIME - used_time

        should_stop_tensor = torch.tensor([0], device=device)
        if rank == 0:
            if epoch < args.epochs - 1:
                if remaining < this_epoch_time * 1.1:
                    print("Not enough time left for next epoch, stop now!")
                    should_stop_tensor = torch.tensor([1], device=device)

        dist.broadcast(should_stop_tensor, src=0)
        if should_stop_tensor.item() == 1:
            sys.exit(0)


    ckpt_path = os.path.join(args.save_dir, "deepseed5")
    tag_name = "llama8B"
    if os.path.exists(os.path.join(ckpt_path, tag_name)):
       success ,client_state= model_engine.load_checkpoint(ckpt_path, tag=tag_name)
       if success:
          if rank == 0:
            print(f"Loaded best checkpoint from {ckpt_path}/{tag_name}")
       else:
          if rank == 0:
            print(f"Checkpoint found but load_checkpoint failed for some reason!")
    else:
        if rank == 0:
            print("Warning: best checkpoint not found, using current model for test.")


    test_sampler.set_epoch(0)
    test_loss, test_top1, test_top3, test_top10, test_topp, test_mrr, test_merged_pairs = \
        evaluate_model(teacher_model, model_engine, test_dataloader, device)

    if rank == 0:
        print(f"[Test] Loss: {test_loss:.4f} | Top-1: {test_top1:.4f} | Top-3: {test_top3:.4f} | "
              f"Top-10: {test_top10:.4f} | Top-p: {test_topp:.4f} | MRR: {test_mrr:.4f} | "
              f"AvgMerged: {test_merged_pairs:.4f}")
        if WANDB_AVAILABLE:
            wandb.log({
                "test/loss": test_loss,
                "test/top1": test_top1,
                "test/top3": test_top3,
                "test/top10": test_top10,
                "test/top_p": test_topp,
                "test/mrr": test_mrr,
                "test/avg_merged_pairs": test_merged_pairs
            })
            wandb.finish()

    dist.barrier()
    if rank == 0:
        print("Training finished.")


if __name__ == "__main__":
    main()
