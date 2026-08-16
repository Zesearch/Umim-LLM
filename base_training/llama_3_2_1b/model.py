import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadPooling(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.token_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, tokens):
        batch_size, k, d_model = tokens.size()
        x1 = self.token_processor(tokens)

        tokens_proc= x1


        tokens_proc = tokens_proc.view(batch_size, k, self.num_heads, self.head_dim)


        query = self.query.unsqueeze(0)
        query = query.unsqueeze(2)


        tokens_proc = tokens_proc.permute(0, 2, 1, 3)


        query = query.expand(batch_size, self.num_heads, 1, self.head_dim)


        attn_scores = torch.einsum("bnhd,bnhd->bnh", query, tokens_proc) / (self.head_dim ** 0.5)

        attn_weights = F.softmax(attn_scores, dim=2).unsqueeze(-1)

        weighted_sum = (tokens_proc * attn_weights).sum(dim=2)


        merged = weighted_sum.view(batch_size, -1)


        merged = self.output_proj(merged)
        return merged


class merge_tokens(nn.Module):
    def __init__(self):
        super(merge_tokens, self).__init__()
        self.attention_pool = MultiHeadPooling(d_model=2048)

    def forward(self, embeddings, merge_indices_batch, attention_mask):
        batch_size, seq_len, embedding_dim = embeddings.size()
        device = embeddings.device
        new_batch_embeddings = []
        padding_masks = []
        loss_masks_batch = []

        for batch_idx in range(batch_size):
            sample_embeddings = embeddings[batch_idx]
            merge_indices = merge_indices_batch[batch_idx]


            if len(merge_indices) > 0:
                earliest_merge_idx = min(g[0] for g in merge_indices)
            else:
                earliest_merge_idx = None

            new_sample_embeddings = []
            sample_padding_mask = []
            merged_indices = set()

            i = 0
            while i < seq_len:
                if i in merged_indices:
                    i += 1
                    continue

                found_group = False
                for merge_group in merge_indices:
                    if i == merge_group[0] and max(merge_group) < seq_len:
                        indices = list(merge_group)

                        tokens_to_merge = sample_embeddings[indices, :]

                        tokens_to_merge = tokens_to_merge.unsqueeze(0)


                        merged_token = self.attention_pool(tokens_to_merge)
                        merged_token = merged_token.squeeze(0)

                        n = len(merge_group)
                        for _ in range(n - 1):
                            new_sample_embeddings.append(torch.zeros_like(merged_token).unsqueeze(0))
                            sample_padding_mask.append(0)

                        new_sample_embeddings.append(merged_token.unsqueeze(0))
                        sample_padding_mask.append(1)

                        merged_indices.update(indices)


                        i = max(indices) + 1
                        found_group = True
                        break

                if not found_group:
                    new_sample_embeddings.append(sample_embeddings[i, :].unsqueeze(0))
                    sample_padding_mask.append(1)
                    i += 1

            compressed_sample = torch.cat(new_sample_embeddings, dim=0)
            new_batch_embeddings.append(compressed_sample)
            pm = torch.tensor(sample_padding_mask, device=device)
            padding_masks.append(pm)
            loss_masks = pm.clone()
            if earliest_merge_idx is not None:
               loss_masks[:earliest_merge_idx] = 0

            loss_masks_batch.append(loss_masks)


        max_length = 512
        padded_embeddings = []
        padded_masks = []
        lossfunction_mask = []

        for compressed_sample, sample_padding_mask ,loss_mask in zip(new_batch_embeddings, padding_masks,loss_masks_batch):


            padding_length = max_length - compressed_sample.size(0)
            if padding_length > 0:
                padding_token = torch.zeros((padding_length, embedding_dim), device=device)
                compressed_sample = torch.cat([compressed_sample, padding_token], dim=0)
                sample_padding_mask = torch.cat([sample_padding_mask, torch.zeros(padding_length, device=device)])
                loss_mask = torch.cat([loss_mask, torch.zeros(padding_length, device=device)])
            padded_embeddings.append(compressed_sample)
            padded_masks.append(sample_padding_mask)
            lossfunction_mask.append(loss_mask)


        compressed_embeddings = torch.stack(padded_embeddings, dim=0)
        padding_mask = torch.stack(padded_masks, dim=0)
        lossfunction_mask = torch.stack(lossfunction_mask, dim=0)


        compressed_embeddings_no_padding = []
        for i in range(batch_size):
            compressed_embeddings_no_padding.append(
                compressed_embeddings[i][padding_mask[i].bool()]
            )

        processed_sequences = [
            torch.cat([seq, torch.zeros((max_length - seq.size(0), seq.size(1)), device=seq.device)], dim=0)
            for seq in compressed_embeddings_no_padding
        ]
        compressed_embeddings_no_padding = torch.stack(processed_sequences)


        new_attention_mask = torch.zeros(
            compressed_embeddings_no_padding.size(0),
            compressed_embeddings_no_padding.size(1),
            device=compressed_embeddings_no_padding.device
        )
        for i, seq_len in enumerate((padding_mask.sum(dim=1)).int()):
            new_attention_mask[i, :seq_len] = 1.0

        return compressed_embeddings_no_padding,padding_mask,new_attention_mask,lossfunction_mask


class CustomGPT2WithCompression(nn.Module):
    def __init__(self, merge_module,shared_gpt):
        super(CustomGPT2WithCompression, self).__init__()
        self.merge_module = merge_module

        self.model = shared_gpt


    def forward(self, input_ids, attention_mask=None, merge_indices=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        embedding_layer = self.model.get_input_embeddings()

        embedding_output = embedding_layer(input_ids).to(device)


        compressed_embeddings,padding_mask,new_attention_mask ,lossfunction_mask= self.merge_module(embedding_output,merge_indices,attention_mask)
        compressed_embeddings = compressed_embeddings

        outputs = self.model(
            inputs_embeds=compressed_embeddings,
            attention_mask=new_attention_mask
        )

        logits = outputs.logits

        return logits, padding_mask, new_attention_mask,lossfunction_mask
