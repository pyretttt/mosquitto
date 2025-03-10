from typing import Optional
import math

import torch
import torch.nn as nn

class KVCache():
    def __init__(self, shape):
        self.keys_cache = torch.empty(shape)
        self.values_cache = torch.empty(shape)
        self._cursor = 0

    def update(self, keys, values):
        self.keys_cache[:, self._cursor: self._cursor + keys.size(-2), :] = keys
        self.values_cache[:, self._cursor: self._cursor + values.size(-2), :] = values
        self._cursor += keys.size(-2)

    def __len__(self):
        return self._cursor


class Attention(nn.Module):
    def __init__(
        self, 
        d_k: int,
        dropout: float = 0.2,
        bias: bool = True
    ):
        super().__init__()
        self.d_k = d_k
        self.dropout = dropout
        self.bias = bias
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.linear_proj = nn.Linear(d_k, d_k, bias=bias)
        self.queries = nn.Linear(d_k, d_k, bias=bias)
        self.keys = nn.Linear(d_k, d_k, bias=bias)
        self.values = nn.Linear(d_k, d_k, bias=bias)
        self.std = math.sqrt(d_k)


    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor,
        cache: Optional[KVCache]
    ):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        if cache is not None:
            cache.update(k, v)
            k = cache.keys_cache[:, :len(cache), :]
            v = cache.values_cache[:, :len(cache), :]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.std
        attn_scores = attn_scores.masked_fill(mask[:, -attn_scores.size(-2):, :] == 0, float('-inf'))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        context_sequence = torch.matmul(attn_scores, v) # B x S x F
        out = self.resid_dropout(self.linear_proj(context_sequence))

        return out