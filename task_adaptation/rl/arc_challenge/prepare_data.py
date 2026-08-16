import argparse
import os
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


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


def save_split(tokenizer, examples, rules, split, output_dir, max_length):
    preferred_ids = []
    preferred_masks = []
    preferred_merges = []
    preferred_starts = []
    rejected_ids = []
    rejected_masks = []
    rejected_merges = []
    rejected_starts = []
    group_ids = []

    for sample_index, example in enumerate(examples):
        question = example["question"]
        choice_labels = example["choices"]["label"]
        choices = example["choices"]["text"]
        answer_index = choice_labels.index(example["answerKey"])
        question_length = len(
            tokenizer(question, add_special_tokens=False)["input_ids"]
        )

        preferred = tokenizer(
            question + " " + choices[answer_index],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        preferred_input_ids = preferred["input_ids"][0]
        preferred_attention_mask = preferred["attention_mask"][0]
        real_ids = preferred_input_ids[
            : int(preferred_attention_mask.sum().item())
        ].tolist()
        merge_indices = find_merge_indices(real_ids[:question_length], rules)

        for choice_index, choice in enumerate(choices):
            if choice_index == answer_index:
                continue

            rejected = tokenizer(
                question + " " + choice,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )

            preferred_ids.append(preferred_input_ids)
            preferred_masks.append(preferred_attention_mask)
            preferred_merges.append(merge_indices)
            preferred_starts.append(question_length)
            rejected_ids.append(rejected["input_ids"][0])
            rejected_masks.append(rejected["attention_mask"][0])
            rejected_merges.append(merge_indices)
            rejected_starts.append(question_length)
            group_ids.append(sample_index)

    torch.save(
        torch.stack(preferred_ids),
        os.path.join(output_dir, f"preferred_input_ids_{split}.pt")
    )
    torch.save(
        torch.stack(preferred_masks),
        os.path.join(output_dir, f"preferred_attention_mask_{split}.pt")
    )
    with open(
        os.path.join(output_dir, f"preferred_merge_indices_{split}.pkl"),
        "wb"
    ) as handle:
        pickle.dump(preferred_merges, handle)
    torch.save(
        torch.tensor(preferred_starts, dtype=torch.long),
        os.path.join(output_dir, f"preferred_ending_start_{split}.pt")
    )

    torch.save(
        torch.stack(rejected_ids),
        os.path.join(output_dir, f"rejected_input_ids_{split}.pt")
    )
    torch.save(
        torch.stack(rejected_masks),
        os.path.join(output_dir, f"rejected_attention_mask_{split}.pt")
    )
    with open(
        os.path.join(output_dir, f"rejected_merge_indices_{split}.pkl"),
        "wb"
    ) as handle:
        pickle.dump(rejected_merges, handle)
    torch.save(
        torch.tensor(rejected_starts, dtype=torch.long),
        os.path.join(output_dir, f"rejected_ending_start_{split}.pt")
    )
    torch.save(
        torch.tensor(group_ids, dtype=torch.long),
        os.path.join(output_dir, f"group_ids_{split}.pt")
    )
    print(f"{split}: samples={len(examples)}, pairs={len(group_ids)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset_name", default="allenai/ai2_arc")
    parser.add_argument("--rules_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rule_tensor = torch.load(args.rules_path, map_location="cpu")
    rules = {tuple(row.tolist()) for row in rule_tensor}
    dataset = load_dataset(args.dataset_name, "ARC-Challenge")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"rules={len(rules)}")
    save_split(
        tokenizer,
        dataset["train"],
        rules,
        "train",
        args.output_dir,
        args.max_length
    )
    save_split(
        tokenizer,
        dataset["validation"],
        rules,
        "val",
        args.output_dir,
        args.max_length
    )


if __name__ == "__main__":
    main()
