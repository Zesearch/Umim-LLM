from __future__ import annotations

import inspect
from typing import Any

import torch

from .merge_module import MergeModule
from .rules import GeneratedElement, RuleMatcher, merge_generated_suffix


def eos_token_ids(model: Any) -> set[int]:
    eos = model.config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(value) for value in eos}
    return {int(eos)}


def cache_length(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if not cache:
        return 0
    return int(cache[0][0].shape[-2])


def crop_cache(cache: Any, length: int) -> Any:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for index in range(len(cache.key_cache)):
            cache.key_cache[index] = cache.key_cache[index][..., :length, :]
            cache.value_cache[index] = cache.value_cache[index][..., :length, :]
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = length
        return cache
    return tuple(
        tuple(value[..., :length, :] for value in layer)
        for layer in cache
    )


def logits_limit_argument(model: Any) -> str | None:
    parameters = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return name
    return None


def model_forward(
    model: Any,
    logits_argument: str | None,
    **kwargs: Any,
) -> Any:
    if logits_argument is not None:
        kwargs[logits_argument] = 1
    return model(**kwargs)


@torch.inference_mode()
def generate_with_merging(
    *,
    model: Any,
    compressed_prompt_embeddings: torch.Tensor,
    prompt_cache_tokens: int,
    merge_module: MergeModule,
    rules: RuleMatcher,
    eos_ids: set[int],
    max_new_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    device = compressed_prompt_embeddings.device
    logits_argument = logits_limit_argument(model)
    attention_mask = torch.ones(
        (1, prompt_cache_tokens),
        dtype=torch.long,
        device=device,
    )
    outputs = model_forward(
        model,
        logits_argument,
        inputs_embeds=compressed_prompt_embeddings,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    cache = outputs.past_key_values
    if cache_length(cache) != prompt_cache_tokens:
        raise RuntimeError("The prefill cache length does not match the prompt")

    logits = outputs.logits[:, -1, :]
    embedding_layer = model.get_input_embeddings()
    generated_ids: list[int] = []
    generated_state: list[GeneratedElement] = []
    merge_counts = {size: 0 for size in rules.packed_rules}
    removed_tokens = 0
    generated_kv_tokens = 0
    cumulative_uncompressed_kv = 0
    cumulative_umim_kv = 0

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if token_id in eos_ids:
            break

        generated_kv_tokens += 1
        generated_state.append(token_id)
        merged_ids = merge_generated_suffix(generated_state, rules)

        if merged_ids is None:
            step_embedding = embedding_layer(next_token).unsqueeze(1)
        else:
            size = len(merged_ids)
            new_length = cache_length(cache) - (size - 1)
            if new_length < prompt_cache_tokens:
                raise RuntimeError("Generated merging attempted to crop prompt KV states")
            cache = crop_cache(cache, new_length)
            child_ids = torch.tensor([merged_ids], dtype=torch.long, device=device)
            step_embedding = merge_module(embedding_layer(child_ids)).unsqueeze(1)
            merge_counts[size] += 1
            removed_tokens += size - 1

        outputs = model_forward(
            model,
            logits_argument,
            inputs_embeds=step_embedding,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        cumulative_uncompressed_kv += generated_kv_tokens
        cumulative_umim_kv += generated_kv_tokens - removed_tokens

    effective_generated_kv = generated_kv_tokens - removed_tokens
    expected_cache_length = prompt_cache_tokens + effective_generated_kv
    if cache_length(cache) != expected_cache_length:
        raise RuntimeError("The final cache length does not match UMIM accounting")

    return generated_ids, {
        "generated_tokens_entering_kv": generated_kv_tokens,
        "effective_generated_kv_tokens": effective_generated_kv,
        "removed_generated_kv_tokens": removed_tokens,
        "generated_kv_reduction_percent": (
            100.0 * removed_tokens / generated_kv_tokens
            if generated_kv_tokens
            else 0.0
        ),
        "merge_counts": {
            str(size): merge_counts[size] for size in sorted(merge_counts)
        },
        "cumulative_uncompressed_generated_kv": cumulative_uncompressed_kv,
        "cumulative_umim_generated_kv": cumulative_umim_kv,
        "cumulative_generated_kv_reduction_percent": (
            100.0 * (cumulative_uncompressed_kv - cumulative_umim_kv)
            / cumulative_uncompressed_kv
            if cumulative_uncompressed_kv
            else 0.0
        ),
        "final_cache_tokens": cache_length(cache),
    }
