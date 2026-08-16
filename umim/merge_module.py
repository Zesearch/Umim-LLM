from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.token_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, d_model = tokens.shape
        processed = self.token_processor(tokens)
        processed = processed.view(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        query = self.query[None, :, None, :].expand(
            batch_size,
            self.num_heads,
            1,
            self.head_dim,
        )
        scores = torch.einsum("bnhd,bnkd->bnk", query, processed)
        weights = F.softmax(scores / self.head_dim**0.5, dim=2).unsqueeze(-1)
        pooled = (processed * weights).sum(dim=2)
        return self.output_proj(pooled.reshape(batch_size, d_model))


class MergeModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attention_pool = MultiHeadPooling(d_model, num_heads)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.attention_pool(embeddings)


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_merge_module(
    weights_path: Path,
    *,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int = 4,
) -> MergeModule:
    state_dict = load_torch(weights_path)
    if isinstance(state_dict, dict):
        for key in ("model_state_dict", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict).__name__}")

    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "merge_module."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value

    module = MergeModule(hidden_size, num_heads)
    module.load_state_dict(normalized, strict=True)
    module.to(device=device, dtype=dtype)
    module.eval()
    return module
