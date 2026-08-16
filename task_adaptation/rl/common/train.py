
import os
import pickle
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from transformers import AutoModelForCausalLM

from model import CustomGPT2WithCompression, merge_tokens


class DPOPairDataset(Dataset):
    def __init__(self, pref_input_ids, pref_attention_mask, pref_merge_indices, pref_ending_start,
                       rej_input_ids, rej_attention_mask, rej_merge_indices, rej_ending_start,
                       group_ids):
        self.pref_input_ids = pref_input_ids
        self.pref_attention_mask = pref_attention_mask
        self.pref_merge_indices = pref_merge_indices
        self.pref_ending_start = pref_ending_start
        self.rej_input_ids = rej_input_ids
        self.rej_attention_mask = rej_attention_mask
        self.rej_merge_indices = rej_merge_indices
        self.rej_ending_start = rej_ending_start
        self.group_ids = group_ids

    def __len__(self):
        return len(self.pref_input_ids)

    def __getitem__(self, idx):
        return {
            "pref_input_ids": self.pref_input_ids[idx],
            "pref_attention_mask": self.pref_attention_mask[idx],
            "pref_merge_indices": self.pref_merge_indices[idx],
            "pref_ending_start": self.pref_ending_start[idx],
            "rej_input_ids": self.rej_input_ids[idx],
            "rej_attention_mask": self.rej_attention_mask[idx],
            "rej_merge_indices": self.rej_merge_indices[idx],
            "rej_ending_start": self.rej_ending_start[idx],
            "group_id": self.group_ids[idx],
        }


def dpo_collate_fn(batch):
    return {
        "pref_input_ids": torch.stack([b["pref_input_ids"] for b in batch]),
        "pref_attention_mask": torch.stack([b["pref_attention_mask"] for b in batch]),
        "pref_merge_indices": [b["pref_merge_indices"] for b in batch],
        "pref_ending_start": torch.stack([b["pref_ending_start"] for b in batch]),
        "rej_input_ids": torch.stack([b["rej_input_ids"] for b in batch]),
        "rej_attention_mask": torch.stack([b["rej_attention_mask"] for b in batch]),
        "rej_merge_indices": [b["rej_merge_indices"] for b in batch],
        "rej_ending_start": torch.stack([b["rej_ending_start"] for b in batch]),
        "group_ids": torch.tensor([b["group_id"] for b in batch], dtype=torch.long),
    }


def compute_ending_log_prob(logits, input_ids, ending_start_pos, padding_mask, lossfunction_mask, new_attention_mask):
    batch_size = input_ids.size(0)


    new_att_counts = new_attention_mask.sum(dim=1)
    loss_func_counts = lossfunction_mask.sum(dim=1)
    diff_per_sample = (new_att_counts - loss_func_counts).long()

    adjusted_mask = new_attention_mask.clone()
    for i in range(batch_size):
        diff_i = diff_per_sample[i].item()
        if diff_i > 0:
            adjusted_mask[i, :diff_i] = 0

    log_probs_list = []

    for i in range(batch_size):
        es = ending_start_pos[i].item()
        diff_i = diff_per_sample[i].item()


        n_ctx_compressed = padding_mask[i][:es].sum().int().item()


        n_skip = max(n_ctx_compressed - diff_i, 0)


        valid_logits = logits[i][adjusted_mask[i].bool()]
        valid_targets = input_ids[i][lossfunction_mask[i].bool()]
        n_valid = min(valid_logits.size(0), valid_targets.size(0))
        valid_logits = valid_logits[:n_valid]
        valid_targets = valid_targets[:n_valid]

        if n_skip >= n_valid:
            log_probs_list.append(torch.tensor(0.0, device=logits.device))
            continue

        ending_logits = valid_logits[n_skip:]
        ending_targets = valid_targets[n_skip:]
        n_ending = ending_logits.size(0)

        if n_ending == 0:
            log_probs_list.append(torch.tensor(0.0, device=logits.device))
            continue

        log_probs = F.log_softmax(ending_logits, dim=-1)
        token_log_probs = log_probs[range(n_ending), ending_targets]
        log_probs_list.append(token_log_probs.mean())

    return torch.stack(log_probs_list)


def compute_dpo_loss(policy_pref_lp, policy_rej_lp, ref_pref_lp, ref_rej_lp, group_ids, beta):
    device = policy_pref_lp.device
    batch_size = policy_pref_lp.size(0)


    log_ratio_pref = policy_pref_lp - ref_pref_lp
    log_ratio_rej = policy_rej_lp - ref_rej_lp
    pair_losses = -F.logsigmoid(beta * (log_ratio_pref - log_ratio_rej))


    unique_groups = group_ids.unique()
    group_losses = []
    for gid in unique_groups:
        mask = (group_ids == gid)
        group_losses.append(pair_losses[mask].mean())

    if len(group_losses) > 0:
        loss = torch.stack(group_losses).mean()
    else:
        loss = torch.tensor(0.0, device=device)

    return loss, batch_size


@torch.no_grad()
def evaluate_accuracy(policy_model, val_dataloader, device):
    policy_model.eval()


    all_pref_scores = []
    all_rej_scores = []
    all_group_ids = []

    for batch in val_dataloader:
        pref_input_ids = batch["pref_input_ids"].to(device)
        pref_attn = batch["pref_attention_mask"].to(device)
        pref_merge = batch["pref_merge_indices"]
        pref_es = batch["pref_ending_start"].to(device)

        rej_input_ids = batch["rej_input_ids"].to(device)
        rej_attn = batch["rej_attention_mask"].to(device)
        rej_merge = batch["rej_merge_indices"]
        rej_es = batch["rej_ending_start"].to(device)

        group_ids = batch["group_ids"].to(device)


        pref_logits, pref_pad_mask, pref_new_attn, pref_loss_mask = policy_model(
            input_ids=pref_input_ids, attention_mask=pref_attn, merge_indices=pref_merge
        )
        pref_lp = compute_ending_log_prob(pref_logits, pref_input_ids, pref_es, pref_pad_mask, pref_loss_mask, pref_new_attn)


        rej_logits, rej_pad_mask, rej_new_attn, rej_loss_mask = policy_model(
            input_ids=rej_input_ids, attention_mask=rej_attn, merge_indices=rej_merge
        )
        rej_lp = compute_ending_log_prob(rej_logits, rej_input_ids, rej_es, rej_pad_mask, rej_loss_mask, rej_new_attn)

        all_pref_scores.append(pref_lp)
        all_rej_scores.append(rej_lp)
        all_group_ids.append(group_ids)

    all_pref_scores = torch.cat(all_pref_scores)
    all_rej_scores = torch.cat(all_rej_scores)
    all_group_ids = torch.cat(all_group_ids)


    ws = dist.get_world_size()
    gathered_pref = [torch.zeros_like(all_pref_scores) for _ in range(ws)]
    gathered_rej = [torch.zeros_like(all_rej_scores) for _ in range(ws)]
    gathered_gid = [torch.zeros_like(all_group_ids) for _ in range(ws)]
    dist.all_gather(gathered_pref, all_pref_scores)
    dist.all_gather(gathered_rej, all_rej_scores)
    dist.all_gather(gathered_gid, all_group_ids)

    if dist.get_rank() == 0:
        all_pref = torch.cat(gathered_pref)
        all_rej = torch.cat(gathered_rej)
        all_gid = torch.cat(gathered_gid)

        correct = 0
        total = 0
        for gid in all_gid.unique():
            mask = (all_gid == gid)
            pref_score = all_pref[mask].mean()
            rej_scores = all_rej[mask]
            all_scores = torch.cat([pref_score.unsqueeze(0), rej_scores])
            if all_scores.argmax() == 0:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pref_train_data_path", type=str, required=True)
    parser.add_argument("--pref_train_attention_mask_path", type=str, required=True)
    parser.add_argument("--pref_train_merge_indices_path", type=str, required=True)
    parser.add_argument("--pref_train_ending_start_path", type=str, required=True)
    parser.add_argument("--rej_train_data_path", type=str, required=True)
    parser.add_argument("--rej_train_attention_mask_path", type=str, required=True)
    parser.add_argument("--rej_train_merge_indices_path", type=str, required=True)
    parser.add_argument("--rej_train_ending_start_path", type=str, required=True)
    parser.add_argument("--train_group_ids_path", type=str, required=True)

    parser.add_argument("--pref_val_data_path", type=str, required=True)
    parser.add_argument("--pref_val_attention_mask_path", type=str, required=True)
    parser.add_argument("--pref_val_merge_indices_path", type=str, required=True)
    parser.add_argument("--pref_val_ending_start_path", type=str, required=True)
    parser.add_argument("--rej_val_data_path", type=str, required=True)
    parser.add_argument("--rej_val_attention_mask_path", type=str, required=True)
    parser.add_argument("--rej_val_merge_indices_path", type=str, required=True)
    parser.add_argument("--rej_val_ending_start_path", type=str, required=True)
    parser.add_argument("--val_group_ids_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--ckpt_subdir", type=str, default="dpo_checkpoint")
    parser.add_argument("--ckpt_tag", type=str, default="best")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pretrained_merge_weights", type=str, required=True,
                        help="Fine-tuned merge module weights (used as policy init AND ref)")
    parser.add_argument("--filter_empty_merges", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()


    deepspeed.init_distributed()
    rank = dist.get_rank()
    local_rank = args.local_rank
    device = torch.device(f"cuda:{local_rank}")


    pref_train_ids = torch.load(args.pref_train_data_path)
    pref_train_attn = torch.load(args.pref_train_attention_mask_path)
    with open(args.pref_train_merge_indices_path, 'rb') as f:
        pref_train_merge = pickle.load(f)
    pref_train_es = torch.load(args.pref_train_ending_start_path)

    rej_train_ids = torch.load(args.rej_train_data_path)
    rej_train_attn = torch.load(args.rej_train_attention_mask_path)
    with open(args.rej_train_merge_indices_path, 'rb') as f:
        rej_train_merge = pickle.load(f)
    rej_train_es = torch.load(args.rej_train_ending_start_path)

    train_group_ids = torch.load(args.train_group_ids_path)

    if args.filter_empty_merges:
        train_valid = [
            index for index in range(len(pref_train_merge))
            if pref_train_merge[index] and rej_train_merge[index]
        ]
        if len(train_valid) < len(pref_train_merge):
            if rank == 0:
                removed = len(pref_train_merge) - len(train_valid)
                print(f"Removed {removed} train pairs with no merge")
            valid_indices = torch.tensor(train_valid, dtype=torch.long)
            pref_train_ids = pref_train_ids[valid_indices]
            pref_train_attn = pref_train_attn[valid_indices]
            pref_train_merge = [pref_train_merge[index] for index in train_valid]
            pref_train_es = pref_train_es[valid_indices]
            rej_train_ids = rej_train_ids[valid_indices]
            rej_train_attn = rej_train_attn[valid_indices]
            rej_train_merge = [rej_train_merge[index] for index in train_valid]
            rej_train_es = rej_train_es[valid_indices]
            train_group_ids = train_group_ids[valid_indices]

    pref_val_ids = torch.load(args.pref_val_data_path)
    pref_val_attn = torch.load(args.pref_val_attention_mask_path)
    with open(args.pref_val_merge_indices_path, 'rb') as f:
        pref_val_merge = pickle.load(f)
    pref_val_es = torch.load(args.pref_val_ending_start_path)

    rej_val_ids = torch.load(args.rej_val_data_path)
    rej_val_attn = torch.load(args.rej_val_attention_mask_path)
    with open(args.rej_val_merge_indices_path, 'rb') as f:
        rej_val_merge = pickle.load(f)
    rej_val_es = torch.load(args.rej_val_ending_start_path)

    val_group_ids = torch.load(args.val_group_ids_path)

    train_dataset = DPOPairDataset(
        pref_train_ids, pref_train_attn, pref_train_merge, pref_train_es,
        rej_train_ids, rej_train_attn, rej_train_merge, rej_train_es,
        train_group_ids
    )
    val_dataset = DPOPairDataset(
        pref_val_ids, pref_val_attn, pref_val_merge, pref_val_es,
        rej_val_ids, rej_val_attn, rej_val_merge, rej_val_es,
        val_group_ids
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=dpo_collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, collate_fn=dpo_collate_fn, drop_last=True)


    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    for param in llm.parameters():
        param.requires_grad = False


    pretrained = torch.load(args.pretrained_merge_weights, map_location="cpu")
    weights = pretrained.get("model_state_dict", pretrained) if isinstance(pretrained, dict) else pretrained
    if not any(key.startswith("attention_pool") for key in weights):
        weights = {
            key.replace("merge_module.", "", 1): value
            for key, value in weights.items()
        }


    policy_merge = merge_tokens()
    policy_merge.load_state_dict(weights)
    policy_model = CustomGPT2WithCompression(policy_merge, llm).to(device, dtype=torch.bfloat16)
    for param in policy_model.parameters():
        param.requires_grad = False
    for param in policy_model.merge_module.parameters():
        param.requires_grad = True

    if rank == 0:
        print(f"[DPO] Loaded merge_module from {args.pretrained_merge_weights}")
        print(f"[DPO] Trainable params: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad):,}")


    ref_merge = merge_tokens()
    ref_merge.load_state_dict(weights)
    ref_model = CustomGPT2WithCompression(ref_merge, llm).to(device, dtype=torch.bfloat16)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()


    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 3e7,
            "allgather_bucket_size": 3e7,
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
        "steps_per_print": 50000,
        "silent": True,
        "wall_clock_breakdown": False,
    }

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy_model,
        model_parameters=(p for p in policy_model.parameters() if p.requires_grad),
        config=ds_config
    )


    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint) \
            and os.path.isdir(os.path.join(args.resume_from_checkpoint, args.ckpt_tag)):
        success, client_state = model_engine.load_checkpoint(
            args.resume_from_checkpoint, tag=args.ckpt_tag)
        if success and client_state is not None:
            start_epoch = client_state.get("epoch", 0) + 1
            global_step = client_state.get("global_step", 0)
            best_val_acc = client_state.get("best_val_acc", 0.0)
            if rank == 0:
                print(f"[Resume] epoch={start_epoch}, step={global_step}, best_val_acc={best_val_acc}")
        else:
            if rank == 0:
                print("[Resume] Load failed. Starting from scratch.")
    else:
        if rank == 0:
            print("No resume checkpoint. Starting from scratch.")


    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs', args.ckpt_subdir))
    else:
        writer = None

    if rank == 0:
        print("DeepSpeed Engine init done. Starting DPO training...")


    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        ref_model.eval()

        if rank == 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        epoch_loss = 0.0
        epoch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
            pref_input_ids = batch["pref_input_ids"].to(device)
            pref_attn = batch["pref_attention_mask"].to(device)
            pref_merge = batch["pref_merge_indices"]
            pref_es = batch["pref_ending_start"].to(device)

            rej_input_ids = batch["rej_input_ids"].to(device)
            rej_attn = batch["rej_attention_mask"].to(device)
            rej_merge = batch["rej_merge_indices"]
            rej_es = batch["rej_ending_start"].to(device)

            group_ids = batch["group_ids"].to(device)


            pref_logits, pref_pad_mask, pref_new_attn, pref_loss_mask = model_engine(
                input_ids=pref_input_ids, attention_mask=pref_attn, merge_indices=pref_merge
            )
            policy_pref_lp = compute_ending_log_prob(
                pref_logits, pref_input_ids, pref_es, pref_pad_mask, pref_loss_mask, pref_new_attn
            )


            rej_logits, rej_pad_mask, rej_new_attn, rej_loss_mask = model_engine(
                input_ids=rej_input_ids, attention_mask=rej_attn, merge_indices=rej_merge
            )
            policy_rej_lp = compute_ending_log_prob(
                rej_logits, rej_input_ids, rej_es, rej_pad_mask, rej_loss_mask, rej_new_attn
            )


            with torch.no_grad():
                ref_pref_logits, ref_pref_pad_mask, ref_pref_new_attn, ref_pref_loss_mask = ref_model(
                    input_ids=pref_input_ids, attention_mask=pref_attn, merge_indices=pref_merge
                )
                ref_pref_lp = compute_ending_log_prob(
                    ref_pref_logits, pref_input_ids, pref_es, ref_pref_pad_mask, ref_pref_loss_mask, ref_pref_new_attn
                )

                ref_rej_logits, ref_rej_pad_mask, ref_rej_new_attn, ref_rej_loss_mask = ref_model(
                    input_ids=rej_input_ids, attention_mask=rej_attn, merge_indices=rej_merge
                )
                ref_rej_lp = compute_ending_log_prob(
                    ref_rej_logits, rej_input_ids, rej_es, ref_rej_pad_mask, ref_rej_loss_mask, ref_rej_new_attn
                )


            loss, n_pairs = compute_dpo_loss(
                policy_pref_lp, policy_rej_lp, ref_pref_lp, ref_rej_lp, group_ids, args.beta
            )

            epoch_loss += loss.item() * n_pairs
            epoch_count += n_pairs


            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

            if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}: "
                      f"DPO_Loss={loss.item():.4f}")
                if writer:
                    writer.add_scalar("train/dpo_loss", loss.item(), global_step)


        epoch_loss_t = torch.tensor([epoch_loss], device=device)
        epoch_count_t = torch.tensor([epoch_count], dtype=torch.float, device=device)
        dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_count_t, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = epoch_loss_t.item() / max(epoch_count_t.item(), 1)
            print(f"[Epoch {epoch+1} Train] Avg DPO Loss: {avg_loss:.4f}")
            if writer:
                writer.add_scalar("train/dpo_loss_epoch", avg_loss, epoch)


        val_acc = evaluate_accuracy(model_engine, val_dataloader, device)

        if rank == 0:
            print(f"[Epoch {epoch+1} Val] Accuracy: {val_acc:.4f} (best: {best_val_acc:.4f})")
            if writer:
                writer.add_scalar("val/accuracy", val_acc, epoch)


        save_this_epoch = False
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_this_epoch = True

        flag_tensor = torch.tensor([1 if save_this_epoch else 0], dtype=torch.int, device=device)
        dist.broadcast(flag_tensor, src=0)
        save_this_epoch = (flag_tensor.item() == 1)

        if save_this_epoch:
            client_state = {
                "epoch": epoch,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "global_step": global_step,
            }
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, args.ckpt_subdir)
            model_engine.save_checkpoint(ckpt_path, tag=args.ckpt_tag, client_state=client_state)
            if rank == 0:
                weight_path = os.path.join(ckpt_path, "merge_module_best.pt")
                torch.save(model_engine.module.merge_module.state_dict(), weight_path)
                print(f"[Best] Saved checkpoint, val_acc={val_acc:.4f}")

            if rank == 0:
                epoch_file = os.path.join(args.save_dir, f"{args.ckpt_subdir}_progress.txt")
                with open(epoch_file, "w") as f:
                    f.write(str(epoch))

        stop_training = False
        if rank == 0:
            if not save_this_epoch:
                patience_counter += 1
                if patience_counter > args.patience:
                    print(f"Early stopping. No improvement for {args.patience} epochs.")
                    stop_training = True
            else:
                patience_counter = 0

        stop_tensor = torch.tensor([int(stop_training)], device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            break


    dist.barrier()
    if rank == 0:
        print(f"\nDPO training finished. Best val accuracy: {best_val_acc:.4f}")
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
