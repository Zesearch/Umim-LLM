from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn

from .merge_module import MergeModule
from .rules import MergedElement, NGram, TokenSpan


@torch.inference_mode()
def build_compressed_embeddings(
    input_ids: torch.Tensor,
    embedding_layer: nn.Module,
    merge_module: MergeModule,
    merge_spans: Sequence[TokenSpan],
) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional")

    original = embedding_layer(input_ids)
    if not merge_spans:
        return original.unsqueeze(0)

    pooled_by_start: dict[int, torch.Tensor] = {}
    for size in sorted({end - start for start, end in merge_spans}):
        spans = [span for span in merge_spans if span[1] - span[0] == size]
        indices = torch.tensor(
            [list(range(start, end)) for start, end in spans],
            dtype=torch.long,
            device=original.device,
        )
        pooled = merge_module(original[indices])
        for row, (start, _) in enumerate(spans):
            pooled_by_start[start] = pooled[row]

    keep = torch.ones(len(input_ids), dtype=torch.bool, device=original.device)
    for start, end in merge_spans:
        keep[start + 1 : end] = False
    keep_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
    compressed = original[keep_indices].clone()
    positions = {
        original_index: compressed_index
        for compressed_index, original_index in enumerate(keep_indices.tolist())
    }
    for start, pooled in pooled_by_start.items():
        compressed[positions[start]] = pooled
    return compressed.unsqueeze(0)


class EmbeddingBuilder:
    def __init__(
        self,
        embedding_layer: nn.Module,
        merge_module: MergeModule,
        cache_size: int = 4096,
    ) -> None:
        self.embedding_layer = embedding_layer
        self.merge_module = merge_module
        self.cache_size = cache_size
        self.pool_cache: OrderedDict[NGram, torch.Tensor] = OrderedDict()
        self.device = embedding_layer.weight.device

    @torch.inference_mode()
    def pooled(self, token_ids: NGram) -> torch.Tensor:
        cached = self.pool_cache.get(token_ids)
        if cached is not None:
            self.pool_cache.move_to_end(token_ids)
            return cached

        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        pooled = self.merge_module(self.embedding_layer(ids).unsqueeze(0))
        pooled = pooled.squeeze(0).detach()
        self.pool_cache[token_ids] = pooled
        if len(self.pool_cache) > self.cache_size:
            self.pool_cache.popitem(last=False)
        return pooled

    @torch.inference_mode()
    def build(self, elements: Sequence[MergedElement]) -> torch.Tensor:
        embeddings = [
            self.pooled(element)
            if isinstance(element, tuple)
            else self.embedding_layer.weight[element]
            for element in elements
        ]
        return torch.stack(embeddings, dim=0).unsqueeze(0)
