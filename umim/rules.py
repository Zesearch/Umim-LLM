from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from .merge_module import load_torch


NGram = tuple[int, ...]
TokenSpan = tuple[int, int]
MergedElement = int | NGram

DEFAULT_RULE_FILENAMES = {
    2: "filtered_bigrams_tensor.pt",
    3: "filtered_trigrams_tensor.pt",
    4: "filtered_fourgrams_tensor.pt",
}
PACKERS = {size: struct.Struct(f"<{size}i") for size in DEFAULT_RULE_FILENAMES}


class RuleMatcher:
    def __init__(self, rules: Mapping[int, Iterable[NGram]]) -> None:
        self.packed_rules = {
            int(size): {
                PACKERS[int(size)].pack(*map(int, row)) for row in rows
            }
            for size, rows in rules.items()
        }
        self.sizes_desc = tuple(sorted(self.packed_rules, reverse=True))
        if not self.sizes_desc:
            raise ValueError("At least one rule file is required")

    @classmethod
    def from_files(cls, paths: Mapping[int, Path]) -> "RuleMatcher":
        packed_rules: dict[int, set[bytes]] = {}
        for size, path in paths.items():
            if size not in PACKERS:
                raise ValueError(f"Unsupported n-gram size: {size}")
            tensor = load_torch(path).to(torch.int32)
            if tensor.ndim != 2 or tensor.shape[1] != size:
                raise ValueError(
                    f"Expected [N, {size}] rules in {path}, got {tuple(tensor.shape)}"
                )
            packed_rules[size] = {
                row.tobytes() for row in tensor.contiguous().numpy()
            }
        instance = cls.__new__(cls)
        instance.packed_rules = packed_rules
        instance.sizes_desc = tuple(sorted(packed_rules, reverse=True))
        if not instance.sizes_desc:
            raise ValueError("At least one rule file is required")
        return instance

    @classmethod
    def from_directory(cls, directory: Path) -> "RuleMatcher":
        return cls.from_files(
            {size: directory / name for size, name in DEFAULT_RULE_FILENAMES.items()}
        )

    def contains(self, token_ids: Sequence[int]) -> bool:
        size = len(token_ids)
        if size not in self.packed_rules:
            return False
        return (
            PACKERS[size].pack(*map(int, token_ids))
            in self.packed_rules[size]
        )


def merge_longest_first(
    token_ids: Sequence[int],
    rules: RuleMatcher,
) -> tuple[list[MergedElement], dict[int, int]]:
    merged: list[MergedElement] = []
    counts = {size: 0 for size in rules.packed_rules}
    index = 0
    while index < len(token_ids):
        for size in rules.sizes_desc:
            candidate = tuple(token_ids[index : index + size])
            if len(candidate) == size and rules.contains(candidate):
                merged.append(candidate)
                counts[size] += 1
                index += size
                break
        else:
            merged.append(int(token_ids[index]))
            index += 1
    return merged, counts


def find_merge_spans(
    token_ids: Sequence[int],
    rules: RuleMatcher,
) -> tuple[list[TokenSpan], dict[int, int]]:
    spans: list[TokenSpan] = []
    counts = {size: 0 for size in rules.packed_rules}
    index = 0
    while index < len(token_ids):
        for size in rules.sizes_desc:
            candidate = token_ids[index : index + size]
            if len(candidate) == size and rules.contains(candidate):
                spans.append((index, index + size))
                counts[size] += 1
                index += size
                break
        else:
            index += 1
    return spans, counts


@dataclass(frozen=True)
class MergedGeneratedSpan:
    token_ids: NGram


GeneratedElement = int | MergedGeneratedSpan


def merge_generated_suffix(
    generated_state: list[GeneratedElement],
    rules: RuleMatcher,
) -> NGram | None:
    for size in rules.sizes_desc:
        if len(generated_state) < size:
            continue
        suffix = generated_state[-size:]
        if not all(isinstance(value, int) for value in suffix):
            continue
        token_ids = tuple(int(value) for value in suffix)
        if rules.contains(token_ids):
            del generated_state[-size:]
            generated_state.append(MergedGeneratedSpan(token_ids))
            return token_ids
    return None
