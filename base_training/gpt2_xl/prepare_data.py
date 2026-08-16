import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pickle
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--threshold", type=int, default=5)
    return parser.parse_args()

def load_data():
    dataset = load_dataset('wikitext','wikitext-103-raw-v1')
    train_data =dataset['train']
    val_data = dataset['validation']
    test_dataset = dataset['test']

    return train_data,val_data,test_dataset


def pre_data_set(data_set,tokenizer,max_length=512):
    raw_text_list = [item['text'] for item in data_set]
    encode_output = tokenizer(raw_text_list, truncation =False , padding =False)
    all_input_ids = [token for seq in encode_output['input_ids'] for token in seq]
    total_length  = (len(all_input_ids)//(max_length))*max_length
    all_input_ids = all_input_ids[:total_length]
    input_ids_original = torch.tensor(all_input_ids).view(-1,max_length).to('cuda')
    attention_mask = torch.ones_like(input_ids_original).to('cuda')
    return input_ids_original,attention_mask


def compute_bigrams(input_ids):

    B, S = input_ids.shape

    bigrams = input_ids.unfold(dimension=1, size=2, step=1)

    bigrams = bigrams.contiguous().view(-1, 2)


    unique_bigrams, counts = torch.unique(bigrams, dim=0, return_counts=True)
    return unique_bigrams, counts

def compute_trigrams(input_ids):

    B, S = input_ids.shape

    trigrams = input_ids.unfold(dimension=1, size=3, step=1)

    trigrams = trigrams.contiguous().view(-1, 3)

    unique_trigrams, counts = torch.unique(trigrams, dim=0, return_counts=True)
    return unique_trigrams, counts


def compute_fourgrams(input_ids):

    B, S = input_ids.shape

    fourgrams = input_ids.unfold(dimension=1, size=4, step=1)

    fourgrams = fourgrams.contiguous().view(-1, 4)

    unique_fourgrams, counts = torch.unique(fourgrams, dim=0, return_counts=True)
    return unique_fourgrams, counts


def filter_high_frequency_bigrams(bigram_list):

    N = len(bigram_list)
    if N <= 1:
        return bigram_list

    token1_dict = defaultdict(list)
    for idx, (t1, t2, freq) in enumerate(bigram_list):
        token1_dict[t1].append(idx)

    filtered = bigram_list[:]
    removed = [False] * N

    for i in range(N):
        if removed[i]:
            continue
        t1_i, t2_i, freq_i = filtered[i]

        candidate_js = token1_dict[t2_i]
        candidate_js = [j for j in candidate_js if j > i and not removed[j]]
        candidate_js.sort()

        for j in candidate_js:
            if removed[i]:
                break
            if removed[j]:
                continue
            t1_j, t2_j, freq_j = filtered[j]

            if freq_i >= freq_j:
                removed[j] = True
            else:
                removed[i] = True
                break

    return [filtered[k] for k in range(N) if not removed[k]]


def filter_high_frequency_trigrams(trigram_list):

    N = len(trigram_list)
    if N <= 1:
        return trigram_list

    dict_12 = defaultdict(list)
    for idx, (t1, t2, t3, freq) in enumerate(trigram_list):
        dict_12[(t1, t2)].append(idx)

    filtered = trigram_list[:]
    removed = [False]*N

    for i in range(N):
        if removed[i]:
            continue
        t1_i, t2_i, t3_i, freq_i = filtered[i]

        candidate_js = dict_12.get((t2_i, t3_i), [])
        candidate_js = [j for j in candidate_js if j > i and not removed[j]]
        candidate_js.sort()

        for j in candidate_js:
            if removed[i]:
                break
            if removed[j]:
                continue
            t1_j, t2_j, t3_j, freq_j = filtered[j]

            if freq_i >= freq_j:
                removed[j] = True
            else:
                removed[i] = True
                break

    return [filtered[k] for k in range(N) if not removed[k]]


def filter_high_frequency_fourgrams(fourgram_list):

    N = len(fourgram_list)
    if N <= 1:
        return fourgram_list

    dict_123 = defaultdict(list)
    for idx, (t1, t2, t3, t4, freq) in enumerate(fourgram_list):
        dict_123[(t1, t2, t3)].append(idx)

    filtered = fourgram_list[:]
    removed = [False]*N

    for i in range(N):
        if removed[i]:
            continue
        t1_i, t2_i, t3_i, t4_i, freq_i = filtered[i]
        candidate_js = dict_123.get((t2_i, t3_i, t4_i), [])
        candidate_js = [j for j in candidate_js if j > i and not removed[j]]
        candidate_js.sort()

        for j in candidate_js:
            if removed[i]:
                break
            if removed[j]:
                continue
            t1_j, t2_j, t3_j, t4_j, freq_j = filtered[j]

            if freq_i >= freq_j:
                removed[j] = True
            else:
                removed[i] = True
                break

    return [filtered[k] for k in range(N) if not removed[k]]


def filter_trigrams_by_fourgrams(trigrams, fourgrams_tokens):

    overlapped_trigrams = set()
    for fg in fourgrams_tokens:
        if len(fg) != 4:
            continue
        overlapped_trigrams.add(fg[0:3])
        overlapped_trigrams.add(fg[1:4])
    filtered_trigrams = [tg for tg in trigrams if tg not in overlapped_trigrams]
    return filtered_trigrams, fourgrams_tokens

def filter_bigrams_by_ngrams(bigrams, trigrams, fourgrams):

    bigrams_from_trigrams = set()
    for tg in trigrams:
        if len(tg) != 3:
            continue
        bigrams_from_trigrams.add(tg[0:2])
        bigrams_from_trigrams.add(tg[1:3])

    bigrams_from_fourgrams = set()
    for fg in fourgrams:
        if len(fg) != 4:
            continue
        bigrams_from_fourgrams.add(fg[0:2])
        bigrams_from_fourgrams.add(fg[1:3])
        bigrams_from_fourgrams.add(fg[2:4])

    bigrams_from_higher = bigrams_from_trigrams.union(bigrams_from_fourgrams)
    filtered_bigrams = [bi for bi in bigrams if bi not in bigrams_from_higher]
    return filtered_bigrams


def candidate_indices_for_n_with_set(seq: torch.Tensor, n: int, ngram_set: set) -> torch.Tensor:

    device = seq.device
    S = seq.size(0)
    if S < n:
        return torch.empty(0, dtype=torch.long, device=device)


    seq_cpu = seq.cpu()
    windows = seq_cpu.unfold(0, n, 1)
    matched_indices = []
    for start_idx in range(windows.size(0)):
        window_tuple = tuple(windows[start_idx].tolist())
        if window_tuple in ngram_set:
            matched_indices.append(start_idx)

    return torch.tensor(matched_indices, dtype=torch.long, device=device)


def vectorized_find_high_freq_phrases_with_set(seq, bigram_set, trigram_set, fourgram_set):

    cand4 = candidate_indices_for_n_with_set(seq, 4, fourgram_set)
    cand3 = candidate_indices_for_n_with_set(seq, 3, trigram_set)
    cand2 = candidate_indices_for_n_with_set(seq, 2, bigram_set)

    cand_list = []
    for idx in cand4.tolist():
        cand_list.append((idx, 4))
    for idx in cand3.tolist():
        cand_list.append((idx, 3))
    for idx in cand2.tolist():
        cand_list.append((idx, 2))

    cand_list.sort(key=lambda x: (x[0], -x[1]))

    final_candidates = []
    last_end = -1
    for (start_idx, n) in cand_list:
        if start_idx >= last_end:
            final_candidates.append((start_idx, n))
            last_end = start_idx + n
    return final_candidates


def process_dataset_with_set(input_ids, bigram_set, trigram_set, fourgram_set):

    B = input_ids.size(0)
    all_merge_indices = []
    for i in range(B):
        seq = input_ids[i]
        merge_indices = vectorized_find_high_freq_phrases_with_set(seq, bigram_set, trigram_set, fourgram_set)
        all_merge_indices.append(merge_indices)
    return all_merge_indices


def convert_candidate_to_range(candidate):

    start, n = candidate
    return tuple(range(start, start + n))

def main():
    args = parse_args()
    drive_dir = args.output_dir
    os.makedirs(drive_dir, exist_ok=True)
    train_data,val_data,test_data = load_data()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    input_ids_for_train,attention_mask_for_train = pre_data_set(train_data,tokenizer,max_length=args.sequence_length)
    input_ids_for_val,attention_mask_for_val = pre_data_set(val_data,tokenizer,max_length=args.sequence_length)
    input_ids_for_test,attention_mask_for_test = pre_data_set(test_data,tokenizer,max_length=args.sequence_length)


    print('compute')
    unique_bigrams, counts_bigrams = compute_bigrams(input_ids_for_train)
    unique_trigrams, counts_trigrams = compute_trigrams(input_ids_for_train)
    unique_fourgrams, counts_fourgrams = compute_fourgrams(input_ids_for_train)


    print('threshold')
    threshold = args.threshold
    mask_bigrams = counts_bigrams >= threshold
    mask_trigrams = counts_trigrams >= threshold
    mask_fourgrams = counts_fourgrams >= threshold

    filtered_bigrams = unique_bigrams[mask_bigrams]
    filtered_count_bigrams = counts_bigrams[mask_bigrams]
    filtered_trigrams = unique_trigrams[mask_trigrams]
    filtered_count_trigrams = counts_trigrams[mask_trigrams]
    filtered_fourgrams = unique_fourgrams[mask_fourgrams]
    filtered_count_fourgrams = counts_fourgrams[mask_fourgrams]


    print('filter by themselves')
    bigram_list = [(b[0], b[1], c) for b, c in zip(filtered_bigrams.tolist(), filtered_count_bigrams.tolist())]
    trigram_list = [(t[0], t[1], t[2], c) for t, c in zip(filtered_trigrams.tolist(), filtered_count_trigrams.tolist())]
    fourgram_list = [(f[0], f[1], f[2], f[3], c) for f, c in zip(filtered_fourgrams.tolist(), filtered_count_fourgrams.tolist())]

    final_bigrams = filter_high_frequency_bigrams(bigram_list)
    final_trigrams = filter_high_frequency_trigrams(trigram_list)
    final_fourgrams = filter_high_frequency_fourgrams(fourgram_list)


    final_bigrams = [(b[0], b[1]) for b in final_bigrams]
    final_trigrams = [(b[0], b[1], b[2]) for b in final_trigrams]
    final_fourgrams = [(b[0], b[1], b[2], b[3]) for b in final_fourgrams]


    print('filter by each other')

    filtered_trigrams, filtered_fourgrams = filter_trigrams_by_fourgrams(final_trigrams,final_fourgrams)
    filtered_bigrams = filter_bigrams_by_ngrams(final_bigrams, final_trigrams, final_fourgrams)


    filtered_trigrams_tensor = torch.tensor(filtered_trigrams, dtype=torch.int32).to('cuda')
    filtered_fourgrams_tensor = torch.tensor(filtered_fourgrams, dtype=torch.int32).to('cuda')
    filtered_bigrams_tensor = torch.tensor(filtered_bigrams, dtype=torch.int32).to('cuda')

    bigram_set   = set(filtered_bigrams)
    trigram_set  = set(filtered_trigrams)
    fourgram_set = set(filtered_fourgrams)

    print('find index based on tensor method')

    merge_indices_batch_for_train = process_dataset_with_set(input_ids_for_train,
                                                    bigram_set,
                                                    trigram_set,
                                                    fourgram_set)

    merge_indices_batch_for_val = process_dataset_with_set(input_ids_for_val,
                                                    bigram_set,
                                                    trigram_set,
                                                    fourgram_set)

    merge_indices_batch_for_test = process_dataset_with_set(input_ids_for_test,
                                                    bigram_set,
                                                    trigram_set,
                                                    fourgram_set)


    merge_indices_for_train = []
    for seq_candidates in merge_indices_batch_for_train:
        seq_ranges = [convert_candidate_to_range(cand) for cand in seq_candidates]
        merge_indices_for_train.append(seq_ranges)

    merge_indices_for_val = []
    for seq_candidates in merge_indices_batch_for_val:
        seq_ranges = [convert_candidate_to_range(cand) for cand in seq_candidates]
        merge_indices_for_val.append(seq_ranges)


    merge_indices_for_test = []
    for seq_candidates in merge_indices_batch_for_test:
        seq_ranges = [convert_candidate_to_range(cand) for cand in seq_candidates]
        merge_indices_for_test.append(seq_ranges)


    sum_removed = 0
    num_seqs = len(merge_indices_for_test)
    print(num_seqs)

    for seq_merges in merge_indices_for_test:
        removed_this_seq = sum(len(tup) - 1 for tup in seq_merges)
        sum_removed += removed_this_seq

    avg_removed_per_seq = sum_removed / num_seqs if num_seqs > 0 else 0
    print("Average tokens removed per sequence:", avg_removed_per_seq)


    torch.save(input_ids_for_train.cpu(), os.path.join(drive_dir, 'input_ids_for_train.pt'))
    torch.save(attention_mask_for_train.cpu(), os.path.join(drive_dir, 'attention_mask_for_train.pt'))
    torch.save(input_ids_for_val.cpu(), os.path.join(drive_dir, 'input_ids_for_val.pt'))
    torch.save(attention_mask_for_val.cpu(), os.path.join(drive_dir, 'attention_mask_for_val.pt'))
    torch.save(input_ids_for_test.cpu(), os.path.join(drive_dir, 'input_ids_for_test.pt'))
    torch.save(attention_mask_for_test.cpu(), os.path.join(drive_dir, 'attention_mask_for_test.pt'))
    torch.save(filtered_trigrams_tensor.cpu(), os.path.join(drive_dir, 'filtered_trigrams_tensor.pt'))
    torch.save(filtered_fourgrams_tensor.cpu(), os.path.join(drive_dir, 'filtered_fourgrams_tensor.pt'))
    torch.save(filtered_bigrams_tensor.cpu(), os.path.join(drive_dir, 'filtered_bigrams_tensor.pt'))

    with open(os.path.join(drive_dir, 'merge_indices_for_train.pkl'), 'wb') as f:
        pickle.dump(merge_indices_for_train, f)
    with open(os.path.join(drive_dir, 'merge_indices_for_val.pkl'), 'wb') as f:
        pickle.dump(merge_indices_for_val, f)
    with open(os.path.join(drive_dir, 'merge_indices_for_test.pkl'), 'wb') as f:
        pickle.dump(merge_indices_for_test, f)


if __name__ == "__main__":
    main()
