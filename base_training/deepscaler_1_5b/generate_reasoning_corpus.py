import os
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_backend', type=str, default="nccl")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--max_gen_length', type=int, default=2048)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--model_name', type=str, default="agentica-org/DeepScaleR-1.5B-Preview")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, default="generated_results")
    return parser.parse_args()


def ddp_setup(args):

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = int(os.environ.get('RANK', 0))

    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend)

    torch.cuda.set_device(local_rank)
    return local_rank


def prepare_datasets():
    try:
        rs1 = load_dataset("KbsdJames/Omni-MATH")
        rs2 = load_dataset("RUC-AIBOX/STILL-3-Preview-RL-Data")

        new_dataset_1 = rs1["test"].map(
            lambda x: {"question": x["problem"]},
            remove_columns=rs1["test"].column_names
        )

        new_dataset_2 = rs2["train"].map(
            lambda x: {"question": x["question"]},
            remove_columns=rs2["train"].column_names
        )

        combined_dataset = concatenate_datasets([new_dataset_1, new_dataset_2])
        return combined_dataset

    except Exception as e:
        print(f"Dataset preparation failed: {e}")
        raise


def process_dataset(combined_dataset, tokenizer, args):

    def tokenize_function(examples):
        return tokenizer(
            examples["question"],
            truncation=False,
            padding=False,
        )

    tokenized_dataset = combined_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["question"]
    )

    def filter_long_samples(example):
        return len(example["input_ids"]) <= args.max_length

    filtered_dataset = tokenized_dataset.filter(filter_long_samples)

    def decode_back_to_text(examples):
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in examples["input_ids"]]
        return {"text": texts}

    decoded_dataset = filtered_dataset.map(
        decode_back_to_text,
        batched=True,
        remove_columns=filtered_dataset.column_names
    )

    def tokenize_function_final(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )

    final_dataset = decoded_dataset.map(
        tokenize_function_final,
        batched=True,
        remove_columns=["text"]
    )

    final_dataset = final_dataset.filter(lambda example: len(example["input_ids"]) <= args.max_length)
    if args.max_samples is not None:
        final_dataset = final_dataset.select(range(min(args.max_samples, len(final_dataset))))

    final_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    return final_dataset


def setup_model(args, local_rank):
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=f"cuda:{local_rank}",
            torch_dtype=torch.float16
        )

        model.eval()

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

        return tokenizer, model

    except Exception as e:
        print(f"Model loading failed: {e}")
        raise


def create_dataloader(dataset, args, local_rank):

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    return dataloader


def generate_sequences(model, dataloader, tokenizer, args, local_rank):

    local_results = []

    if local_rank == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Generating")

    for batch_idx, batch in enumerate(dataloader):
        try:

            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}")

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):

                outputs = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_p=0.7,
                    temperature=1.0,
                    max_length=args.max_gen_length,
                    num_return_sequences=args.num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            local_results.extend(decoded)

            if local_rank == 0:
                progress_bar.update(1)

        except Exception as e:
            print(f"Generation failed for batch {batch_idx}: {e}")
            continue

    if local_rank == 0:
        progress_bar.close()

    return local_results


def gather_results(local_results, local_rank, args):

    local_data_pickle = pickle.dumps(local_results)
    local_data_byte = torch.ByteTensor(list(local_data_pickle)).to(f"cuda:{local_rank}")

    local_size = torch.LongTensor([local_data_byte.size(0)]).to(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    size_list = [torch.LongTensor([0]).to(f"cuda:{local_rank}") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    max_size = max(size.item() for size in size_list)
    padded_tensor = torch.zeros((max_size,), dtype=torch.uint8).to(f"cuda:{local_rank}")
    padded_tensor[:local_size.item()] = local_data_byte

    gather_list = [torch.zeros((max_size,), dtype=torch.uint8).to(f"cuda:{local_rank}") for _ in range(world_size)]

    dist.all_gather(gather_list, padded_tensor)

    all_results = []
    if local_rank == 0:
        for i in range(world_size):
            real_size = size_list[i].item()
            byte_i = gather_list[i][:real_size].cpu().numpy().tobytes()
            data_i = pickle.loads(byte_i)
            all_results.extend(data_i)

        print(f"[Rank=0] Collected {len(all_results)} sequences.")

        generated_dataset = Dataset.from_dict({"generated_text": all_results})

        os.makedirs(args.output_dir, exist_ok=True)
        output_prefix = f"{args.output_prefix}"


        pickle_path = os.path.join(args.output_dir, f"{output_prefix}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved generated sequences to {pickle_path}")

        return generated_dataset
    return None


def main():

    args = setup_args()

    torch.manual_seed(args.seed)

    local_rank = ddp_setup(args)

    try:

        if local_rank == 0:
            print("Preparing datasets...")
        combined_dataset = prepare_datasets()

        if local_rank == 0:
            print("Loading model and tokenizer...")
        tokenizer, model = setup_model(args, local_rank)

        if local_rank == 0:
            print("Processing datasets...")
        final_dataset = process_dataset(combined_dataset, tokenizer, args)

        if local_rank == 0:
            print("Creating dataloader...")
        dataloader = create_dataloader(final_dataset, args, local_rank)

        if local_rank == 0:
            print("Generating reasoning trajectories...")
        local_results = generate_sequences(model, dataloader, tokenizer, args, local_rank)

        if local_rank == 0:
            print("Gathering generated trajectories...")
        generated_dataset = gather_results(local_results, local_rank, args)

        if local_rank == 0 and generated_dataset is not None:
            output_path = os.path.join(args.output_dir, f"{args.output_prefix}.jsonl")
            generated_dataset.to_json(output_path)
            print(f"Saved generated dataset to {output_path}")

    except Exception as e:
        print(f"[Rank={local_rank}] Failed: {e}")
    finally:

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
