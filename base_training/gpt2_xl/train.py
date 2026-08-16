import os
import pickle
import random
import time
import sys

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from torch.nn.parallel import DistributedDataParallel as DDP


from model import CustomGPT2WithCompression, merge_tokens, MultiHeadPooling
from utils import (
    compute_cross_entropy_loss,
    compute_top1_accuracy,
    compute_top3_overlap,
    compute_top10_overlap,
    compute_top_p_metric,
    compute_mrr,
    save_checkpoint,
    plot_all_metrics
)


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


def ddp_setup(args):

    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def load_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        print("sorry ,not my bad ,your first training ,enjoy it!")
        return None

    checkpoint = torch.load(ckpt_path,map_location="cpu")
    print(f"Loading checkpoint from {ckpt_path}")
    return checkpoint


def evaluate_model(gpt2_model, model_with_compression, dataloader, device):

    if isinstance(model_with_compression, DDP):
        model = model_with_compression.module
    else:
        model = model_with_compression

    if isinstance(gpt2_model, DDP):
        gpt2 = gpt2_model.module
    else:
        gpt2 = gpt2_model

    model.eval()
    gpt2.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top10 = 0.0
    total_topp = 0.0
    total_mrr = 0.0
    count = 0

    merged_sum = 0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            merge_indices_batch = batch["merge_indices"]

            with autocast():
              outputs = gpt2(input_ids, attention_mask=attention_mask)
              original_logits = outputs.logits
              original_probs = F.softmax(original_logits, dim=-1)


              compressed_logits, padding_mask, new_attention_mask, lossfunction_mask = model(
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

            total_loss += loss.item()* batch_size
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


    device_loss         = torch.tensor([total_loss], dtype=torch.float, device=device)
    device_top1         = torch.tensor([total_top1], dtype=torch.float, device=device)
    device_top3         = torch.tensor([total_top3], dtype=torch.float, device=device)
    device_top10        = torch.tensor([total_top10], dtype=torch.float, device=device)
    device_topp         = torch.tensor([total_topp], dtype=torch.float, device=device)
    device_mrr          = torch.tensor([total_mrr],  dtype=torch.float, device=device)
    device_count        = torch.tensor([count],      dtype=torch.float, device=device)
    device_merged_sum   = torch.tensor([merged_sum], dtype=torch.float, device=device)
    device_sample_count = torch.tensor([sample_count], dtype=torch.float, device=device)

    dist.all_reduce(device_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(device_top1,  op=dist.ReduceOp.SUM)
    dist.all_reduce(device_top3,  op=dist.ReduceOp.SUM)
    dist.all_reduce(device_top10, op=dist.ReduceOp.SUM)
    dist.all_reduce(device_topp,  op=dist.ReduceOp.SUM)
    dist.all_reduce(device_mrr,   op=dist.ReduceOp.SUM)
    dist.all_reduce(device_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(device_merged_sum,  op=dist.ReduceOp.SUM)
    dist.all_reduce(device_sample_count, op=dist.ReduceOp.SUM)

    rank = dist.get_rank()


    local_steps = torch.tensor([len(dataloader)], dtype=torch.float, device=device)
    dist.all_reduce(local_steps, op=dist.ReduceOp.SUM)


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
            avg_top1 = avg_top3 = avg_top10 = avg_topp = avg_mrr = 0.0

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


    parser.add_argument("--dist_backend", type=str, default="nccl")


    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss_threshold", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=3)


    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_running_time",type=int,default=28800,
                        help="The default is 8 hours")


    parser.add_argument("--wandb_project", type=str, default="MyDistProject",
                        help="W&B project name to log metrics.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional run name in W&B.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to a checkpoint to resume training from.")


    args = parser.parse_args()
    start_time = time.time()
    MAX_TIME = args.max_running_time


    local_rank = ddp_setup(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")


    checkpoint =None
    start_epoch =0
    global_step =0
    best_val_loss = float('inf')
    wandb_run_id = None

    if args.resume_from_checkpoint is not None:
        checkpoint = load_checkpoint(args.resume_from_checkpoint)
        if checkpoint is not None:
            start_epoch =checkpoint["epoch"]+1
            global_step =checkpoint["global_step"]
            best_val_loss =checkpoint["best_val_loss"]
            wandb_run_id = checkpoint["wandb_run_id"]
            if rank == 0:
                print(f"Resuming from epoch {checkpoint['epoch']} at step {checkpoint['global_step']}")


    else:
        if rank == 0:
         print("Starting training without a checkpoint.")


    new_id = wandb_run_id
    if rank == 0 and WANDB_AVAILABLE:
        wandb_kwargs ={
            "project": args.wandb_project,
            "name" : args.wandb_run_name,
            "config":vars(args),
            "settings": wandb.Settings(init_timeout=300)
        }

        if wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] ="allow"
        wandb.init(**wandb_kwargs)
        new_id = wandb.run.id


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
                                  sampler=train_sampler, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 sampler=test_sampler, collate_fn=custom_collate_fn)


    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device)
    for param in gpt2_model.parameters():
        param.requires_grad = False

    merge_module = merge_tokens()
    model_with_compression = CustomGPT2WithCompression(merge_module, gpt2_model).to(device)


    for param in model_with_compression.parameters():
        param.requires_grad = False
    for param in model_with_compression.merge_module.parameters():
        param.requires_grad = True


    model_with_compression = DDP(
        model_with_compression,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )


    optimizer = torch.optim.AdamW(
        model_with_compression.module.merge_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler()

    if checkpoint is not None :
        model_with_compression.module.merge_module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])


    epoch_losses = []
    top1_means_per_epoch = []
    top3_means_per_epoch = []
    top10_means_per_epoch = []
    topp_means_per_epoch = []
    mrr_means_per_epoch = []
    merged_count_per_epoch =[]

    val_loss_history = []
    val_top1_history = []
    val_top3_history = []
    val_top10_history = []
    val_topp_history = []
    val_mrr_history = []

    patience_counter = 0

    for epoch in range(start_epoch,args.epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n=== Starting Epoch {epoch + 1}/{args.epochs} ===")

        model_with_compression.train()
        gpt2_model.eval()


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

            with autocast():


             with torch.no_grad():
                 outputs = gpt2_model(input_ids, attention_mask=attention_mask)
                 original_logits = outputs.logits
                 original_probs = F.softmax(original_logits, dim=-1)


             compressed_logits, padding_mask, new_attention_mask, lossfunction_mask = model_with_compression(
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

            epoch_loss += loss.item()* batch_size
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


            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model_with_compression.module.merge_module.parameters(),
                max_norm=10
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1


            if rank == 0 and (batch_idx + 1) % args.log_interval == 0:

                avg_pairs_in_batch = 0.0
                if batch_size > 0:
                    total_pairs_batch = 0
                    for i in range(batch_size):
                        pidx = merge_indices_batch[i]
                        if torch.is_tensor(pidx):
                            pidx = pidx.cpu().tolist()
                        total_pairs_batch += len(pidx)
                    avg_pairs_in_batch = total_pairs_batch / batch_size

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
                        "train/avg_merged_pairs_batch": avg_pairs_in_batch,
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
        train_merged_sum_t     = torch.tensor([train_merged_sum], dtype=torch.float, device=device)
        train_sample_count_t   = torch.tensor([train_sample_count], dtype=torch.float, device=device)

        dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(top1_sum_t,   op=dist.ReduceOp.SUM)
        dist.all_reduce(top3_sum_t,   op=dist.ReduceOp.SUM)
        dist.all_reduce(top10_sum_t,  op=dist.ReduceOp.SUM)
        dist.all_reduce(topp_sum_t,   op=dist.ReduceOp.SUM)
        dist.all_reduce(mrr_sum_t,    op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t,      op=dist.ReduceOp.SUM)
        dist.all_reduce(train_merged_sum_t,     op=dist.ReduceOp.SUM)
        dist.all_reduce(train_sample_count_t,   op=dist.ReduceOp.SUM)

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
                epoch_loss_avg = avg_top1 = avg_top3 = avg_top10 = avg_topp = avg_mrr = 0.0

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


        val_sampler.set_epoch(epoch)
        val_loss, val_top1, val_top3, val_top10, val_topp, val_mrr, val_merged_pairs = evaluate_model(
            gpt2_model, model_with_compression, val_dataloader, device
        )

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


            stop_training = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, "best_checkpoint_for_gpt2_xl_5.pt")
                save_checkpoint(model_with_compression, optimizer, scheduler, epoch,
                                val_loss, best_val_loss,global_step,new_id,scaler,ckpt_path)
                print(f"  [Best Model Saved] Val Loss improved to {val_loss:.4f}!")
            else:
                patience_counter += 1
                if patience_counter > args.patience:
                    print(f"Early stopping triggered. No improvement for {args.patience} epochs.")
                    stop_training = True

            if val_loss < args.loss_threshold:
                print(f"Stopping early because val_loss < {args.loss_threshold:.6f}")
                stop_training = True
        else:
            stop_training = False

        stop_training_tensor = torch.tensor([int(stop_training)], device=device)
        dist.broadcast(stop_training_tensor, src=0)
        if stop_training_tensor.item() == 1:
            break
        should_stop_tensor = torch.tensor([0], device=device)
        if rank == 0:
            used_time = time.time()- start_time
            this_epoch_time = time.time() -epoch_start_time
            remaining = MAX_TIME - used_time

            if epoch < args.epochs -1 :
                if remaining < this_epoch_time *1.1:
                    print("Stopping because the remaining allocation is too short for another epoch.")
                    should_stop_tensor = torch.tensor([1], device=device)

        dist.broadcast(should_stop_tensor, src=0)

        if should_stop_tensor.item() == 1:
            sys.exit(0)


    final_model = model_with_compression.module
    checkpoint_path = os.path.join(args.save_dir, "best_checkpoint_for_gpt2_xl_5.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        final_model.merge_module.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: best_checkpoint.pt not found, using current model for test.")

    test_sampler.set_epoch(0)
    test_loss, test_top1, test_top3, test_top10, test_topp, test_mrr, test_merged_pairs = evaluate_model(
        gpt2_model,
        final_model,
        test_dataloader,
        device
    )

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


if __name__ == "__main__":
    main()
