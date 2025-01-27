import math
from typing import Callable

import torch, torch.nn as nn

def casual_mask(*shape: int):
    return torch.tril(torch.ones(shape[-2], shape[-1])).view(*shape)


class Attention(nn.Module):
    def __init__(
        self, 
        d_k: int,
        bias: bool = True,
        is_self_attn: bool = False,
        dropout: float = 0.2,
        mask_factory: Callable[..., torch.Tensor] = casual_mask
    ):
        super().__init__()
        self.d_k = d_k
        self.queries_proj = nn.Linear(d_k, d_k, bias=bias) # Weights with shape (d_k x I)
        self.keys_proj = nn.Linear(d_k, d_k, bias=bias) # Weights with shape (d_k x I)
        self.values_proj = nn.Linear(d_k, d_k, bias=bias) # Weights with shape (d_k x I)
        self.linear = nn.Linear(d_k, d_k, bias=bias)
        self.is_self_attn = is_self_attn
        self.std = math.sqrt(self.d_k)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.keys = None
        self.values = None
        self.mask_factory = mask_factory

    def init_keys(self, x: torch.Tensor):
        self.keys = self.keys_proj(x) # (B x S x d_k)
        self.values = self.values_proj(x) # (B x S x d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            x (torch.Tensor): shape (Batch x Sequence x Feature)
            mask (torch.Tensor): shape (Batch x Sequence x Sequence)

        Returns:
            torch.Tensor: shape (B x S x F)
        """
        _, S, _ = x.size()
        queries = self.queries_proj(x) # (B x S x d_k)
        if self.is_self_attn:
            self.init_keys(x)

        attn_scores = torch.matmul(
            queries, 
            self.keys.transpose(-2, -1)
        ) / self.std # (B x S_source x S_target)
        if mask is None:
            mask = self.mask_factory(1, S, self.keys.size(dim=-2))
        
        assert attn_scores.shape[2:] == mask.shape[2:]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_scores = torch.softmax(attn_scores, dim=-1) # over source sequence
        
        attn_scores = self.attn_dropout(attn_scores)

        ctx_sequence = torch.matmul(attn_scores, self.values) # B x S x d_k
        out = self.resid_dropout(self.linear(ctx_sequence)) # B x S x F

        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        d_k: int,
        nheads: int,
        bias: bool = True,
        is_self_attn: bool = False,
        dropout: float = 0.2,
        mask_factory: Callable[..., torch.Tensor] = casual_mask
    ):
        super().__init__()
        assert d_k % nheads == 0, "KVS dimension should be divisible by number of attention heads"
        
        self.d_k = d_k
        self.nheads = nheads
        self.queries_proj = nn.Linear(d_k, d_k, bias=bias) # Weights with shape (d_k x I)
        self.keys_proj = nn.Linear(d_k, d_k, bias=bias) # Weights with shape (d_k x I)
        self.values_proj = nn.Linear(d_k, d_k, bias=bias) # Weights with shape (d_k x I)
        self.linear = nn.Linear(d_k, d_k)
        self.is_self_attn = is_self_attn
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.std = math.sqrt(self.d_k // self.nheads)
        self.keys = None
        self.values = None
        self.mask_factory = mask_factory
        
    
    def init_keys(self, x):
        self.keys = self.multihead_proj(kvs=self.keys_proj, x=x) # (B x H x S x d_k)
        self.values = self.multihead_proj(kvs=self.values_proj, x=x) # (B x H x S x d_k)


    def multihead_proj(
        self, 
        kvs: nn.Linear, 
        x: torch.Tensor
    ):
        # (B x S x d_k) ~> (B x H x S x d_k / H)
        B, S, _ = x.shape
        out = kvs(x)
        out = out.view(B, S, self.nheads, self.d_k // self.nheads).transpose(1, 2)
        return out
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            x (torch.Tensor): shape (Batch x Sequence x Feature)
            mask (torch.Tensor): shape (Batch x Sequence x Sequence)

        Returns:
            torch.Tensor: shape (B x S x F)
        """
        B, S, _ = x.size()
        queries = self.multihead_proj(self.queries_proj, x) # (B x H x S x d_k // n_heads)
        if self.is_self_attn:
            self.init_keys(x)

        attn_scores = torch.matmul(
            queries,
            self.keys.transpose(-2, -1)
        ) / (self.std) # (B x H x S_source x S_target)
        
        if mask is None:
            mask = self.mask_factory(1, 1, S, self.keys.size(dim=-2))

        assert attn_scores.shape[2:] == mask.shape[2:]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_scores = torch.softmax(attn_scores, dim=-1) # over source sequence
        attn_scores = self.attn_dropout(attn_scores)

        ctx_sequence = torch.matmul(attn_scores, self.values) # B x H x S x d_k // n_heads
        ctx_sequence = ctx_sequence.transpose(1, 2).contiguous().view(B, S, self.d_k) # B x S x d_k
        out = self.resid_dropout(self.linear(ctx_sequence)) # B x S x F
        
        return out


if __name__ == "__main__":
    attn = MultiheadAttention(64, 32, 4, is_self_attn=True, dropout=0.0)
    out = attn(torch.randn(4, 2, 64))
    print(out.shape)
    # out = attn.multihead_forward(attn.queries_proj, torch.randn(6, 6, 64))
    # print(out[0].shape)
    # print(len(out))
