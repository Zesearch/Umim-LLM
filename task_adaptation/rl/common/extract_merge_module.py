import argparse
import os

import torch


def extract_merge_module(checkpoint_dir, tag):
    state_path = os.path.join(
        checkpoint_dir,
        tag,
        "mp_rank_00_model_states.pt"
    )
    if not os.path.isfile(state_path):
        raise FileNotFoundError(state_path)

    state = torch.load(state_path, map_location="cpu")
    full_state = state.get("module", state)
    merge_state = {
        key.removeprefix("merge_module."): value
        for key, value in full_state.items()
        if key.startswith("merge_module.")
    }
    if not merge_state:
        raise ValueError("The checkpoint does not contain merge-module weights")
    return merge_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tag", default="best")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    weights = extract_merge_module(args.checkpoint_dir, args.tag)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(weights, args.output)
    print(f"Saved {len(weights)} tensors to {args.output}")


if __name__ == "__main__":
    main()
