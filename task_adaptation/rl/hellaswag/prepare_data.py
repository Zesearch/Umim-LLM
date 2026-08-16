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
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])
        context_length = len(
            tokenizer(context, add_special_tokens=False)["input_ids"]
        )

        preferred = tokenizer(
            context + " " + endings[label],
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
        merge_indices = find_merge_indices(real_ids[:context_length], rules)

        for ending_index, ending in enumerate(endings):
            if ending_index == label:
                continue

            rejected = tokenizer(
                context + " " + ending,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )

            preferred_ids.append(preferred_input_ids)
            preferred_masks.append(preferred_attention_mask)
            preferred_merges.append(merge_indices)
            preferred_starts.append(context_length)
            rejected_ids.append(rejected["input_ids"][0])
            rejected_masks.append(rejected["attention_mask"][0])
            rejected_merges.append(merge_indices)
            rejected_starts.append(context_length)
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
    parser.add_argument("--dataset_name", default="Rowan/hellaswag")
    parser.add_argument("--rules_root", required=True)
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

    for threshold in args.thresholds:
        rule_path = os.path.join(
            args.rules_root,
            f"threshold_{threshold}",
            "filtered_bigrams_tensor.pt"
        )
        rule_tensor = torch.load(rule_path, map_location="cpu")
        rules = {tuple(row.tolist()) for row in rule_tensor}
        output_dir = os.path.join(args.output_dir, f"threshold_{threshold}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"threshold={threshold}: rules={len(rules)}")
        save_split(
            tokenizer,
            dataset["train"],
            rules,
            "train",
            output_dir,
            args.max_length
        )
        save_split(
            tokenizer,
            dataset["validation"],
            rules,
            "val",
            output_dir,
            args.max_length
        )


if __name__ == "__main__":
    main()
