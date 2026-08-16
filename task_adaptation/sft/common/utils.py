import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import math

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType


import torch.distributed as dist
import deepspeed


def compute_cross_entropy_loss(compressed_probs, original_probs,lossfunction_mask, new_attention_mask):


      fixed_length = 128

      new_att_counts = new_attention_mask.sum(dim=1)
      loss_func_counts = lossfunction_mask.sum(dim=1)
      diff_per_sample = new_att_counts - loss_func_counts
      diff_per_sample = diff_per_sample.long()


      adjusted_mask = new_attention_mask.clone()
      for i in range(adjusted_mask .size(0)):
        diff_i = diff_per_sample[i].item()
        if diff_i > 0:

            adjusted_mask[i, :diff_i] = 0


      aligned_original_probs = []
      for i in range(original_probs.size(0)):
          aligned_original_probs.append(original_probs[i][lossfunction_mask[i].bool()])

      aligned_original_probs = [
          torch.cat([prob, torch.zeros((fixed_length - prob.size(0), prob.size(1)), device=prob.device)], dim=0)
          for prob in aligned_original_probs
      ]
      aligned_original_probs = torch.stack(aligned_original_probs)


      aligned_compressed_probs = []

      for i  in range(compressed_probs.size(0)):
         aligned_compressed_probs.append(compressed_probs[i][adjusted_mask[i].bool()])

      aligned_compressed_probs = [
          torch.cat([prob, torch.zeros((fixed_length - prob.size(0), prob.size(1)), device=prob.device)], dim=0)
          for prob in aligned_compressed_probs
      ]
      aligned_compressed_probs = torch.stack(aligned_compressed_probs)


      masked_compressed_probs = (aligned_compressed_probs + 1e-9)
      masked_aligned_original_probs = (aligned_original_probs + 1e-9)

      cross_entropy_loss = -torch.sum(masked_aligned_original_probs * masked_compressed_probs.log()) / lossfunction_mask.sum()
      combined_loss = cross_entropy_loss


      return combined_loss,adjusted_mask


def compute_top1_accuracy(
    original_probs,
    compressed_probs,
    teacher_mask,
    student_mask,
    fixed_length=128
):

    device = original_probs.device
    B, seq_len, V = original_probs.size()

    teacher_list = []
    teacher_mask_list = []

    for i in range(B):

        selected = original_probs[i][teacher_mask[i].bool()]
        k = selected.size(0)


        pad_len = fixed_length - k
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            selected = torch.cat([selected, pad_zeros], dim=0)
        else:

            selected = selected[:fixed_length]
            k = min(k, fixed_length)


        fm = torch.zeros(fixed_length, device=device)
        fm[:k] = 1.0

        teacher_list.append(selected)
        teacher_mask_list.append(fm)

    aligned_teacher = torch.stack(teacher_list, dim=0)
    teacher_final_mask = torch.stack(teacher_mask_list, dim=0)


    student_list = []
    student_mask_list = []

    for i in range(B):
        st_sel = compressed_probs[i][student_mask[i].bool()]
        k_st = st_sel.size(0)

        pad_len = fixed_length - k_st
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            st_sel = torch.cat([st_sel, pad_zeros], dim=0)
        else:
            st_sel = st_sel[:fixed_length]
            k_st = min(k_st, fixed_length)

        fm_st = torch.zeros(fixed_length, device=device)
        fm_st[:k_st] = 1.0

        student_list.append(st_sel)
        student_mask_list.append(fm_st)

    aligned_student = torch.stack(student_list, dim=0)
    student_final_mask = torch.stack(student_mask_list, dim=0)


    top1_orig = torch.argmax(aligned_teacher, dim=-1)
    top1_comp = torch.argmax(aligned_student, dim=-1)


    if not torch.equal(teacher_final_mask, student_final_mask):
        diff_positions = (teacher_final_mask != student_final_mask).sum().item()
        raise ValueError(f"Masks differ in {diff_positions} positions!")


    combined_mask = (teacher_final_mask * student_final_mask)


    match_matrix = (top1_orig == top1_comp).float()

    numer = (match_matrix * combined_mask).sum()
    denom = combined_mask.sum()

    if denom.item() < 1e-9:

        top1_accuracy = 0.0
    else:
        top1_accuracy = (numer / denom).item()

    return top1_accuracy


def compute_top3_overlap(
    original_probs,
    compressed_probs,
    teacher_mask,
    student_mask,
    fixed_length=128
):
    device = original_probs.device
    B, seq_len, V = original_probs.size()

    teacher_list = []
    teacher_mask_list = []

    for i in range(B):
        selected = original_probs[i][teacher_mask[i].bool()]
        k = selected.size(0)


        pad_len = fixed_length - k
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            selected = torch.cat([selected, pad_zeros], dim=0)
        else:
            selected = selected[:fixed_length]
            k = min(k, fixed_length)

        fm = torch.zeros(fixed_length, device=device)
        fm[:k] = 1.0

        teacher_list.append(selected)
        teacher_mask_list.append(fm)

    teacher_aligned = torch.stack(teacher_list, dim=0)
    teacher_final_mask = torch.stack(teacher_mask_list, dim=0)


    student_list = []
    student_mask_list = []

    for i in range(B):
        st_sel = compressed_probs[i][student_mask[i].bool()]
        k_st = st_sel.size(0)

        pad_len = fixed_length - k_st
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            st_sel = torch.cat([st_sel, pad_zeros], dim=0)
        else:
            st_sel = st_sel[:fixed_length]
            k_st = min(k_st, fixed_length)

        fm_st = torch.zeros(fixed_length, device=device)
        fm_st[:k_st] = 1.0

        student_list.append(st_sel)
        student_mask_list.append(fm_st)

    student_aligned = torch.stack(student_list, dim=0)
    student_final_mask = torch.stack(student_mask_list, dim=0)
    combined_mask = (teacher_final_mask * student_final_mask)


    teacher_flat = teacher_aligned.view(B * fixed_length, V)
    student_flat = student_aligned.view(B * fixed_length, V)
    mask_flat = combined_mask.view(B * fixed_length)


    valid_positions = (mask_flat == 1)
    teacher_valid = teacher_flat[valid_positions]
    student_valid = student_flat[valid_positions]


    if teacher_valid.size(0) == 0:
        return 0.0

    _, top3_orig_idx = torch.topk(teacher_valid, k=3, dim=-1)
    _, top3_comp_idx = torch.topk(student_valid, k=3, dim=-1)


    intersection_count = []
    for i in range(3):


        match_i = (top3_orig_idx[:, i].unsqueeze(-1) == top3_comp_idx).any(dim=-1).float()
        intersection_count.append(match_i)


    intersection_count = torch.stack(intersection_count, dim=-1).sum(dim=-1)


    overlap_ratio = (intersection_count / 3.0).mean().item()

    return overlap_ratio


def compute_top10_overlap(
    original_probs,
    compressed_probs,
    teacher_mask,
    student_mask,
    fixed_length=128
):

    device = original_probs.device
    B, seq_len, V = original_probs.size()

    teacher_list = []
    teacher_mask_list = []

    for i in range(B):
        selected = original_probs[i][teacher_mask[i].bool()]
        k = selected.size(0)
        pad_len = fixed_length - k
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            selected = torch.cat([selected, pad_zeros], dim=0)
        else:
            selected = selected[:fixed_length]
            k = min(k, fixed_length)
        fm = torch.zeros(fixed_length, device=device)
        fm[:k] = 1.0

        teacher_list.append(selected)
        teacher_mask_list.append(fm)

    teacher_aligned = torch.stack(teacher_list, dim=0)
    teacher_final_mask = torch.stack(teacher_mask_list, dim=0)

    student_list = []
    student_mask_list = []

    for i in range(B):
        st_sel = compressed_probs[i][student_mask[i].bool()]
        k_st = st_sel.size(0)

        pad_len = fixed_length - k_st
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            st_sel = torch.cat([st_sel, pad_zeros], dim=0)
        else:
            st_sel = st_sel[:fixed_length]
            k_st = min(k_st, fixed_length)

        fm_st = torch.zeros(fixed_length, device=device)
        fm_st[:k_st] = 1.0

        student_list.append(st_sel)
        student_mask_list.append(fm_st)

    student_aligned = torch.stack(student_list, dim=0)
    student_final_mask = torch.stack(student_mask_list, dim=0)

    combined_mask = (teacher_final_mask * student_final_mask)

    teacher_flat = teacher_aligned.view(B * fixed_length, V)
    student_flat = student_aligned.view(B * fixed_length, V)
    mask_flat = combined_mask.view(B * fixed_length)


    valid_positions = (mask_flat == 1)
    teacher_valid = teacher_flat[valid_positions]
    student_valid = student_flat[valid_positions]

    if teacher_valid.size(0) == 0:
        return 0.0


    _, top10_orig_idx = torch.topk(teacher_valid, k=10, dim=-1)
    _, top10_comp_idx = torch.topk(student_valid, k=10, dim=-1)


    intersection_count = []
    for i in range(10):

        match_i = (top10_orig_idx[:, i].unsqueeze(-1) == top10_comp_idx).any(dim=-1).float()
        intersection_count.append(match_i)


    intersection_count = torch.stack(intersection_count, dim=-1).sum(dim=-1)

    overlap_ratio = (intersection_count / 10.0).mean().item()

    return overlap_ratio


def compute_top_p_metric(
    original_probs,
    compressed_probs,
    teacher_mask,
    student_mask,
    p=0.9,
    fixed_length=128
):


    device = original_probs.device
    B, seq_len, V = original_probs.shape

    teacher_list = []
    teacher_mask_list = []

    for i in range(B):
        selected = original_probs[i][teacher_mask[i].bool()]
        k = selected.size(0)

        pad_len = fixed_length - k
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            selected = torch.cat([selected, pad_zeros], dim=0)
        else:
            selected = selected[:fixed_length]
            k = min(k, fixed_length)


        fm = torch.zeros(fixed_length, device=device)
        fm[:k] = 1.0

        teacher_list.append(selected)
        teacher_mask_list.append(fm)

    teacher_aligned = torch.stack(teacher_list, dim=0)
    teacher_final_mask = torch.stack(teacher_mask_list, dim=0)

    student_list = []
    student_mask_list = []

    for i in range(B):
        st_sel = compressed_probs[i][student_mask[i].bool()]
        k_st = st_sel.size(0)

        pad_len = fixed_length - k_st
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            st_sel = torch.cat([st_sel, pad_zeros], dim=0)
        else:
            st_sel = st_sel[:fixed_length]
            k_st = min(k_st, fixed_length)

        fm_st = torch.zeros(fixed_length, device=device)
        fm_st[:k_st] = 1.0

        student_list.append(st_sel)
        student_mask_list.append(fm_st)

    student_aligned = torch.stack(student_list, dim=0)
    student_final_mask = torch.stack(student_mask_list, dim=0)


    combined_mask = teacher_final_mask * student_final_mask


    teacher_flat = teacher_aligned.view(B * fixed_length, V)
    student_flat = student_aligned.view(B * fixed_length, V)
    mask_flat = combined_mask.view(B * fixed_length)


    valid_positions = (mask_flat == 1)
    teacher_valid = teacher_flat[valid_positions]
    student_valid = student_flat[valid_positions]

    M = teacher_valid.size(0)
    if M == 0:
        return 0.0

    sorted_orig_vals, sorted_orig_idx = torch.sort(teacher_valid, dim=-1, descending=True)
    cumsum_orig = torch.cumsum(sorted_orig_vals, dim=-1)

    top_p_mask_orig = (cumsum_orig >= p).float()
    first_true_idx_orig = top_p_mask_orig.argmax(dim=-1)

    sorted_comp_vals, sorted_comp_idx = torch.sort(student_valid, dim=-1, descending=True)
    cumsum_comp = torch.cumsum(sorted_comp_vals, dim=-1)
    top_p_mask_comp = (cumsum_comp >= p).float()
    first_true_idx_comp = top_p_mask_comp.argmax(dim=-1)


    top_p_intersection_ratios = []
    for i in range(M):
        p_set_size_orig = first_true_idx_orig[i].item() + 1
        p_set_size_comp = first_true_idx_comp[i].item() + 1

        p_set_orig = sorted_orig_idx[i, :p_set_size_orig]
        p_set_comp = sorted_comp_idx[i, :p_set_size_comp]


        p_set_orig_expand = p_set_orig.unsqueeze(-1)
        p_set_comp_expand = p_set_comp.unsqueeze(0)
        intersection_count = (p_set_orig_expand == p_set_comp_expand).any(dim=-1).sum().item()

        m = min(p_set_size_orig, p_set_size_comp)
        ratio = intersection_count / m
        top_p_intersection_ratios.append(ratio)


    top_p_metric = sum(top_p_intersection_ratios) / len(top_p_intersection_ratios)
    return top_p_metric


def compute_mrr(
    original_probs,
    compressed_probs,
    teacher_mask,
    student_mask,
    fixed_length=128
):

    device = original_probs.device
    B, seq_len, V = original_probs.size()


    teacher_list = []
    teacher_mask_list = []
    for i in range(B):

        selected = original_probs[i][teacher_mask[i].bool()]
        k = selected.size(0)

        pad_len = fixed_length - k
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            selected = torch.cat([selected, pad_zeros], dim=0)
        else:
            selected = selected[:fixed_length]
            k = min(k, fixed_length)

        fm = torch.zeros(fixed_length, device=device)
        fm[:k] = 1.0

        teacher_list.append(selected)
        teacher_mask_list.append(fm)

    teacher_aligned = torch.stack(teacher_list, dim=0)
    teacher_final_mask = torch.stack(teacher_mask_list, dim=0)

    student_list = []
    student_mask_list = []
    for i in range(B):
        st_sel = compressed_probs[i][student_mask[i].bool()]
        k_st = st_sel.size(0)

        pad_len = fixed_length - k_st
        if pad_len > 0:
            pad_zeros = torch.zeros((pad_len, V), device=device)
            st_sel = torch.cat([st_sel, pad_zeros], dim=0)
        else:
            st_sel = st_sel[:fixed_length]
            k_st = min(k_st, fixed_length)

        fm_st = torch.zeros(fixed_length, device=device)
        fm_st[:k_st] = 1.0

        student_list.append(st_sel)
        student_mask_list.append(fm_st)

    student_aligned = torch.stack(student_list, dim=0)
    student_final_mask = torch.stack(student_mask_list, dim=0)


    combined_mask = teacher_final_mask * student_final_mask


    teacher_flat = teacher_aligned.view(B * fixed_length, V)
    student_flat = student_aligned.view(B * fixed_length, V)
    mask_flat = combined_mask.view(B * fixed_length)


    valid_positions = (mask_flat == 1)
    teacher_valid = teacher_flat[valid_positions]
    student_valid = student_flat[valid_positions]

    M = teacher_valid.size(0)
    if M == 0:
        return 0.0


    correct_tokens = torch.argmax(teacher_valid, dim=-1)


    sorted_vals, sorted_idx = torch.sort(student_valid, dim=-1, descending=True)

    matches = (sorted_idx == correct_tokens.unsqueeze(-1))


    positions = matches.nonzero(as_tuple=False)

    ranks = positions[:, 1].float() + 1.0

    rr = torch.zeros(matches.size(0), device=device)
    rr[positions[:, 0]] = 1.0 / ranks

    mrr = rr.mean().item()
    return mrr


def save_checkpoint(model_ddp, optimizer, scheduler, epoch, val_loss, best_val_loss,global_step,run_id, scaler,path):

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "model_state_dict": model_ddp.module.merge_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "wandb_run_id": run_id
    }
    torch.save(checkpoint, path)

def save_checkpoint_fsdp(model_ddp, optimizer, scheduler, epoch, val_loss, best_val_loss, global_step, run_id, scaler, path):

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType


    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return


    if isinstance(model_ddp, FSDP):

        with FSDP.state_dict_type(
            model_ddp,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        ):
            full_sd = model_ddp.state_dict()

        merge_sd = {k: v for k, v in full_sd.items() if "merge_module" in k}
    else:

        merge_sd = model_ddp.module.merge_module.state_dict()

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "model_state_dict": merge_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "wandb_run_id": run_id
    }

    torch.save(checkpoint, path)
    print(f"[Rank{rank}] checkpoint saved to {path}")

def save_checkpoint_deepspeed(
    model_engine,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    best_val_loss,
    global_step,
    run_id,
    scaler,
    path
):

    merge_module_state = model_engine.module.merge_module.state_dict()

    checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_state_dict": merge_module_state,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "global_step": global_step,
            "wandb_run_id": run_id
    }
    torch.save(checkpoint, path)


def plot_all_metrics(
    epoch_losses,
    top1_means_per_epoch,
    top3_means_per_epoch,
    top10_means_per_epoch,
    topp_means_per_epoch,
    mrr_means_per_epoch,
    val_loss_history,
    val_top1_history,
    val_top3_history,
    val_top10_history,
    val_topp_history,
    val_mrr_history,
    save_path=None
):

    num_epochs = len(epoch_losses)
    epochs = range(1, num_epochs+1)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()


    axes[0].plot(epochs, epoch_losses, label='Train')
    axes[0].plot(epochs, val_loss_history, label='Val')
    axes[0].set_title("Loss")
    axes[0].legend()


    axes[1].plot(epochs, top1_means_per_epoch, label='Train')
    axes[1].plot(epochs, val_top1_history, label='Val')
    axes[1].set_title("Top-1")
    axes[1].legend()


    axes[2].plot(epochs, top3_means_per_epoch, label='Train')
    axes[2].plot(epochs, val_top3_history, label='Val')
    axes[2].set_title("Top-3")
    axes[2].legend()


    axes[3].plot(epochs, top10_means_per_epoch, label='Train')
    axes[3].plot(epochs, val_top10_history, label='Val')
    axes[3].set_title("Top-10")
    axes[3].legend()


    axes[4].plot(epochs, topp_means_per_epoch, label='Train')
    axes[4].plot(epochs, val_topp_history, label='Val')
    axes[4].set_title("Top-p")
    axes[4].legend()


    axes[5].plot(epochs, mrr_means_per_epoch, label='Train')
    axes[5].plot(epochs, val_mrr_history, label='Val')
    axes[5].set_title("MRR")
    axes[5].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
