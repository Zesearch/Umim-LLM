from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dynamic_decoding import eos_token_ids, generate_with_merging
from .merge_module import load_merge_module
from .prompt_merging import EmbeddingBuilder, build_compressed_embeddings
from .rules import (
    MergedElement,
    RuleMatcher,
    find_merge_spans,
    merge_longest_first,
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def target_id(element: MergedElement) -> int:
    return element[0] if isinstance(element, tuple) else element


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


class UMIMRuntime:
    @classmethod
    def from_config(
        cls,
        *,
        config_path: str | Path,
        model_path: str,
        device: str = "cuda:0",
        attention_implementation: str | None = None,
        dtype: str = "bfloat16",
    ) -> "UMIMRuntime":
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.resolve().parent
        config["merge_weights"] = resolve_path(
            config["merge_weights"], base_dir
        )
        if "rule_files" in config:
            config["rule_files"] = {
                int(size): resolve_path(value, base_dir)
                for size, value in config["rule_files"].items()
            }
        elif "rules_dir" in config:
            config["rules_dir"] = resolve_path(config["rules_dir"], base_dir)
        else:
            raise ValueError("Config must define rule_files or rules_dir")
        return cls(
            config=config,
            model_path=model_path,
            device=device,
            attention_implementation=attention_implementation,
            dtype=dtype,
        )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        model_path: str,
        device: str,
        attention_implementation: str | None,
        dtype: str,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        torch_dtype = DTYPES[dtype]
        self.device = torch.device(device)

        local_files_only = bool(config.get("local_files_only", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": {"": device},
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if attention_implementation is not None:
            model_kwargs["attn_implementation"] = attention_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.model.eval()

        hidden_size = int(self.model.get_input_embeddings().weight.shape[1])
        self.merge_module = load_merge_module(
            Path(config["merge_weights"]),
            hidden_size=hidden_size,
            device=self.device,
            dtype=torch_dtype,
            num_heads=int(config.get("num_heads", 4)),
        )
        if "rule_files" in config:
            self.rules = RuleMatcher.from_files(config["rule_files"])
        else:
            self.rules = RuleMatcher.from_directory(Path(config["rules_dir"]))
        self.embedding_builder = EmbeddingBuilder(
            self.model.get_input_embeddings(),
            self.merge_module,
            cache_size=int(config.get("surrogate_cache_size", 4096)),
        )

    @torch.inference_mode()
    def _sequence_nll(
        self,
        input_elements: Sequence[MergedElement],
        target_ids: Sequence[int],
    ) -> float:
        if not input_elements or not target_ids:
            return 0.0
        embeddings = self.embedding_builder.build(input_elements)
        attention_mask = torch.ones(
            embeddings.shape[:2], dtype=torch.long, device=embeddings.device
        )
        logits = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
        targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(
            logits[0, -len(targets) :].float(),
            targets,
            reduction="sum",
        )
        return float(loss.item())

    @torch.inference_mode()
    def score_text(self, text: str) -> dict[str, Any]:
        token_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        original_ids = token_ids.tolist()
        elements, merge_counts = merge_longest_first(original_ids, self.rules)
        if len(elements) < 2:
            raise ValueError("Text must contain at least two effective tokens")

        targets = [target_id(element) for element in elements[1:]]
        total_nll = self._sequence_nll(elements[:-1], targets)
        return {
            "loss": total_nll / len(targets),
            "total_nll": total_nll,
            "target_tokens": len(targets),
            "original_tokens": len(original_ids),
            "effective_tokens": len(elements),
            "merge_counts": {
                str(size): merge_counts[size] for size in sorted(merge_counts)
            },
        }

    def score_candidates(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        return [self.score_text(text) for text in texts]

    @torch.inference_mode()
    def score_perplexity_tokens(self, token_ids: Sequence[int]) -> dict[str, Any]:
        original_ids = [int(value) for value in token_ids]
        if len(original_ids) < 2:
            return {
                "total_nll": 0.0,
                "target_tokens": 0,
                "original_tokens": len(original_ids),
                "effective_tokens": len(original_ids),
                "merge_counts": {},
            }

        elements, merge_counts = merge_longest_first(original_ids, self.rules)
        main_targets = [target_id(element) for element in elements[1:]]
        total_nll = self._sequence_nll(elements[:-1], main_targets)
        target_tokens = len(main_targets)

        for index, element in enumerate(elements):
            if not isinstance(element, tuple):
                continue
            prefix = list(elements[:index])
            for offset in range(1, len(element)):
                context: list[MergedElement] = prefix + list(element[:offset])
                total_nll += self._sequence_nll(context, [element[offset]])
                target_tokens += 1

        if target_tokens != len(original_ids) - 1:
            raise RuntimeError("Perplexity target accounting is inconsistent")
        return {
            "total_nll": total_nll,
            "target_tokens": target_tokens,
            "original_tokens": len(original_ids),
            "effective_tokens": len(elements),
            "merge_counts": {
                str(size): merge_counts[size] for size in sorted(merge_counts)
            },
        }

    @torch.inference_mode()
    def generate(self, prompt: str, *, max_new_tokens: int) -> dict[str, Any]:
        token_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
        )["input_ids"]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        spans, prompt_merge_counts = find_merge_spans(token_ids, self.rules)
        original_prompt_tokens = len(token_ids)
        effective_prompt_tokens = original_prompt_tokens - sum(
            end - start - 1 for start, end in spans
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prompt_embeddings = build_compressed_embeddings(
            input_ids,
            self.model.get_input_embeddings(),
            self.merge_module,
            spans,
        )
        generated_ids, generation_metrics = generate_with_merging(
            model=self.model,
            compressed_prompt_embeddings=prompt_embeddings,
            prompt_cache_tokens=effective_prompt_tokens,
            merge_module=self.merge_module,
            rules=self.rules,
            eos_ids=eos_token_ids(self.model),
            max_new_tokens=max_new_tokens,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / 1024**3
            if self.device.type == "cuda"
            else 0.0
        )
        prediction = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        prompt_removed = original_prompt_tokens - effective_prompt_tokens

        return {
            "prediction": prediction,
            "generated_token_ids": generated_ids,
            "metrics": {
                "original_prompt_tokens": original_prompt_tokens,
                "effective_prompt_tokens": effective_prompt_tokens,
                "prompt_reduction_percent": (
                    100.0 * prompt_removed / original_prompt_tokens
                    if original_prompt_tokens
                    else 0.0
                ),
                "prompt_merge_counts": {
                    str(size): prompt_merge_counts[size]
                    for size in sorted(prompt_merge_counts)
                },
                "generated_tokens": len(generated_ids),
                "latency_seconds": latency,
                "peak_cuda_memory_gib": peak_memory,
                "decoding_merge_counts": generation_metrics["merge_counts"],
                **{
                    key: value
                    for key, value in generation_metrics.items()
                    if key != "merge_counts"
                },
            },
        }
