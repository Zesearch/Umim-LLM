import argparse
import os
import pickle
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def example_text(example):
    answer_index = example["choices"]["label"].index(example["answerKey"])
    answer = example["choices"]["text"][answer_index]
    return example["question"] + " " + answer


def question_prefix(tokenizer, example, max_length):
    full_ids = tokenizer(
        example_text(example),
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )["input_ids"][0].tolist()
    question_length = len(
        tokenizer(example["question"], add_special_tokens=False)["input_ids"]
    )
    return full_ids[:question_length]


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
        tokens = question_prefix(tokenizer, example, max_length)
        for start, end in find_merge_indices(tokens, candidates):
            used.add((tokens[start], tokens[end]))
    return used


def save_split(tokenizer, examples, rules, split, output_dir, max_length):
    input_ids = []
    attention_masks = []
    merge_indices = []
    question_tokens = 0
    removed_tokens = 0

    for example in examples:
        encoded = tokenizer(
            example_text(example),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        prefix = question_prefix(tokenizer, example, max_length)
        indices = find_merge_indices(prefix, rules)

        input_ids.append(encoded["input_ids"][0])
        attention_masks.append(encoded["attention_mask"][0])
        merge_indices.append(indices)
        question_tokens += len(prefix)
        removed_tokens += len(indices)

    torch.save(torch.stack(input_ids), os.path.join(output_dir, f"input_ids_{split}.pt"))
    torch.save(
        torch.stack(attention_masks),
        os.path.join(output_dir, f"attention_mask_{split}.pt")
    )
    with open(os.path.join(output_dir, f"merge_indices_{split}.pkl"), "wb") as handle:
        pickle.dump(merge_indices, handle)

    token_reduction = 100.0 * removed_tokens / max(question_tokens, 1)
    print(f"{split}: samples={len(input_ids)}, question TR={token_reduction:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset_name", default="allenai/ai2_arc")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, "ARC-Challenge")
    train_examples = dataset["train"]
    validation_examples = dataset["validation"]

    counts = Counter()
    for example in train_examples:
        tokens = question_prefix(tokenizer, example, args.max_length)
        counts.update(zip(tokens, tokens[1:]))

    candidates = {
        pair for pair, count in counts.items() if count >= args.threshold
    }
    rules = collect_used_rules(
        tokenizer,
        train_examples,
        candidates,
        args.max_length
    )
    os.makedirs(args.output_dir, exist_ok=True)

    rule_tensor = torch.tensor(sorted(rules), dtype=torch.int32)
    if not rules:
        rule_tensor = torch.empty((0, 2), dtype=torch.int32)
    torch.save(
        rule_tensor,
        os.path.join(args.output_dir, "filtered_bigrams_tensor.pt")
    )

    print(f"threshold={args.threshold}: rules={len(rules)}")
    save_split(
        tokenizer,
        train_examples,
        rules,
        "train",
        args.output_dir,
        args.max_length
    )
    save_split(
        tokenizer,
        validation_examples,
        rules,
        "val",
        args.output_dir,
        args.max_length
    )


if __name__ == "__main__":
    main()
