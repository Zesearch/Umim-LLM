import argparse
import os
import pickle
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def example_text(example):
    label = int(example["label"])
    return example["ctx"] + " " + example["endings"][label]


def token_ids(tokenizer, text, max_length):
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )["input_ids"][0].tolist()


def find_merge_indices(tokens, rules):
    merge_indices = []
    index = 0
    while index < len(tokens) - 1:
        pair = (tokens[index], tokens[index + 1])
        if pair in rules:
            merge_indices.append((index, index + 1))
            index += 2
        else:
            index += 1
    return merge_indices


def collect_used_rules(tokenizer, examples, candidates, max_length):
    used = set()
    for example in examples:
        tokens = token_ids(tokenizer, example_text(example), max_length)
        for start, end in find_merge_indices(tokens, candidates):
            used.add((tokens[start], tokens[end]))
    return used


def save_split(tokenizer, examples, rules, split, output_dir, max_length):
    input_ids = []
    attention_masks = []
    merge_indices = []
    total_tokens = 0
    total_removed = 0

    for example in examples:
        encoded = tokenizer(
            example_text(example),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        ids = encoded["input_ids"][0]
        mask = encoded["attention_mask"][0]
        real_tokens = ids[: int(mask.sum().item())].tolist()
        indices = find_merge_indices(real_tokens, rules)

        input_ids.append(ids)
        attention_masks.append(mask)
        merge_indices.append(indices)
        total_tokens += len(real_tokens)
        total_removed += len(indices)

    torch.save(torch.stack(input_ids), os.path.join(output_dir, f"input_ids_{split}.pt"))
    torch.save(
        torch.stack(attention_masks),
        os.path.join(output_dir, f"attention_mask_{split}.pt")
    )
    with open(os.path.join(output_dir, f"merge_indices_{split}.pkl"), "wb") as handle:
        pickle.dump(merge_indices, handle)

    token_reduction = 100.0 * total_removed / max(total_tokens, 1)
    print(f"{split}: samples={len(input_ids)}, TR={token_reduction:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset_name", default="Rowan/hellaswag")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[2000, 200, 50, 5, 1]
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name)
    train_examples = dataset["train"]
    validation_examples = dataset["validation"]

    counts = Counter()
    for example in train_examples:
        tokens = token_ids(tokenizer, example_text(example), args.max_length)
        counts.update(zip(tokens, tokens[1:]))

    for threshold in args.thresholds:
        candidates = {pair for pair, count in counts.items() if count >= threshold}
        rules = collect_used_rules(
            tokenizer,
            train_examples,
            candidates,
            args.max_length
        )
        output_dir = os.path.join(args.output_dir, f"threshold_{threshold}")
        os.makedirs(output_dir, exist_ok=True)

        rule_tensor = torch.tensor(sorted(rules), dtype=torch.int32)
        if not rules:
            rule_tensor = torch.empty((0, 2), dtype=torch.int32)
        torch.save(rule_tensor, os.path.join(output_dir, "filtered_bigrams_tensor.pt"))

        print(f"threshold={threshold}: rules={len(rules)}")
        save_split(
            tokenizer,
            train_examples,
            rules,
            "train",
            output_dir,
            args.max_length
        )
        save_split(
            tokenizer,
            validation_examples,
            rules,
            "val",
            output_dir,
            args.max_length
        )


if __name__ == "__main__":
    main()
