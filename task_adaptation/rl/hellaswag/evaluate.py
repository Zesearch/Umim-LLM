import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import merge_tokens


def merge_sequence(token_ids, rules):
    merged = []
    index = 0
    while index < len(token_ids):
        pair = tuple(token_ids[index:index + 2])
        if len(pair) == 2 and pair in rules:
            merged.append(pair)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


def build_embeddings(sequence, embedding_layer, merge_module, device, dtype):
    embeddings = []
    for item in sequence:
        if isinstance(item, tuple):
            child_ids = torch.tensor(item, device=device).unsqueeze(0)
            child_embeddings = embedding_layer(child_ids)
            embedding = merge_module.attention_pool(child_embeddings)[0]
        else:
            token_id = torch.tensor(item, device=device)
            embedding = embedding_layer(token_id)
        embeddings.append(embedding.to(dtype=dtype))
    return torch.stack(embeddings).unsqueeze(0)


def score_ending(
    context_ids,
    ending_ids,
    model,
    embedding_layer,
    merge_module,
    rules,
    device,
    dtype
):
    if not ending_ids:
        return 0.0

    merged_context = merge_sequence(context_ids, rules)
    context_embeddings = build_embeddings(
        merged_context,
        embedding_layer,
        merge_module,
        device,
        dtype
    )
    context_output = model(
        inputs_embeds=context_embeddings,
        return_dict=True,
        use_cache=True
    )

    ending_tensor = torch.tensor(ending_ids, device=device)
    ending_embeddings = embedding_layer(ending_tensor).unsqueeze(0)
    ending_output = model(
        inputs_embeds=ending_embeddings,
        past_key_values=context_output.past_key_values,
        return_dict=True,
        use_cache=False
    )

    logits = torch.cat(
        [context_output.logits[0, -1:].clone(), ending_output.logits[0, :-1]],
        dim=0
    )
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(ending_ids), device=device),
        ending_tensor
    ]
    return token_log_probs.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset_name", default="Rowan/hellaswag")
    parser.add_argument("--merge_weights", required=True)
    parser.add_argument("--rules_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype
    ).to(device)
    model.eval()

    merge_module = merge_tokens()
    merge_module.load_state_dict(
        torch.load(args.merge_weights, map_location="cpu")
    )
    merge_module = merge_module.to(device=device, dtype=dtype)
    merge_module.eval()

    rule_tensor = torch.load(args.rules_path, map_location="cpu")
    rules = {tuple(row.tolist()) for row in rule_tensor}
    validation = load_dataset(args.dataset_name)["validation"]
    embedding_layer = model.get_input_embeddings()

    correct = 0
    total_tokens = 0
    removed_tokens = 0

    with torch.no_grad():
        for example in tqdm(validation):
            context = example["ctx"]
            context_ids = tokenizer(
                context,
                add_special_tokens=False
            )["input_ids"]
            scores = []
            for ending in example["endings"]:
                full_ids = tokenizer(
                    context + " " + ending,
                    add_special_tokens=False
                )["input_ids"]
                ending_ids = full_ids[len(context_ids):]
                scores.append(
                    score_ending(
                        context_ids,
                        ending_ids,
                        model,
                        embedding_layer,
                        merge_module,
                        rules,
                        device,
                        dtype
                    )
                )

            correct += int(scores.index(max(scores)) == int(example["label"]))
            merged_context = merge_sequence(context_ids, rules)
            total_tokens += len(context_ids)
            removed_tokens += len(context_ids) - len(merged_context)

    accuracy = correct / len(validation)
    token_reduction = 100.0 * removed_tokens / max(total_tokens, 1)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Token reduction: {token_reduction:.2f}%")


if __name__ == "__main__":
    main()
