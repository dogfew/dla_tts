import math

import torch
import torch.nn as nn


class QuantizationEmbedding(nn.Module):
    def __init__(
        self,
        values_range=(..., ...),
        n_bins=256,
        encoder_hidden=256,
        large_zero=False,
        **kwargs
    ):
        super().__init__()
        min_value, max_value = values_range
        self.bins = nn.Parameter(
            torch.cat(
                [torch.tensor([0]), torch.linspace(min_value, max_value, n_bins - 2)]
            ).expm1()
            if large_zero
            else torch.linspace(min_value, max_value, n_bins - 1).expm1(),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(n_bins, encoder_hidden)

    def forward(self, x):
        return self.embedding(torch.bucketize(x, self.bins))


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


def get_non_pad_mask(seq, PAD=0):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, PAD=0):
    """For masking out the padding part of key sequence."""
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_attn_key_pad_mask_heads(seq_k, seq_q, n_heads=2, PAD=0):
    """For masking out the padding part of key sequence."""
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    batch_size = padding_mask.size(0)
    padding_mask = padding_mask.repeat(1, n_heads, 1).view(
        batch_size * n_heads, len_q, -1
    )
    return padding_mask


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask
