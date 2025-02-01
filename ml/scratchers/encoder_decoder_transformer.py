import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratchers.transformer_config import TransformerConfig
from scratchers.transformer.attn import Attention
from scratchers.positional_embeddings import PositionalEmbeddings

def forward_attention(
    queries: torch.Tensor, 
    keys: torch.Tensor, 
    values: torch.Tensor,
    projection: nn.Linear,
    dropout: float,
    mask: torch.Tensor
) -> torch.Tensor:
    d_k = queries.size(-1)
    attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
    attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
    attn_scores = F.dropout(attn_scores, dropout)
    attn_scores = torch.softmax(attn_scores, dim=-1)

    ctx_sequence = torch.matmul(attn_scores, values) # BxSxF
    out = projection(ctx_sequence)
    out = F.dropout(out, dropout)

    return out


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
           TransformerEncoderLayer(config) for _ in range(config.nlayers)
        ])
        self.linear_projector1 = nn.Linear(config.input_size, config.attn_d_k)
        self.pemb = PositionalEmbeddings(config.max_seq_len, config.attn_d_k)
        self.linear_projector2 = nn.Linear(config.attn_d_k, config.input_size)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
   
    @staticmethod
    def self_attn_mask(shape: tuple[int]):
        return torch.tril(
            torch.ones(shape[-2:])
        ).view(*shape)

    def forward(
        self,
        src_seq: torch.Tensor,
        tgt_seq: Optional[torch.Tensor] = None,
    ):
        x = self.linear_projector1(src_seq)
        x = self.pemb(x)
        if tgt_seq is None: # Hack for playground
            tgt_seq = x
        memory = self.encoder(x, mask=torch.ones(x.size(-2), x.size(-2)).unsqueeze(0))
        out = self.decoder(
            torch.concat((memory[:, -1:, :], tgt_seq[:, :-1, :]), dim=-2),
            memory,
            self.self_attn_mask((1, tgt_seq.size(-2), tgt_seq.size(-2))),
            memory_mask=torch.ones(tgt_seq.size(-2), x.size(-2)).unsqueeze(0)
        )
        out = self.linear_projector2(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
           TransformerDecoderLayer(config) for _ in range(config.nlayers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor, 
        src_mask: torch.Tensor,
        memory_mask: torch.Tensor
    ):
        for block in self.blocks:
            x = block(
                x, 
                memory, 
                src_mask,
                memory_mask
            )
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
           TransformerEncoderLayer(config) for _ in range(config.nlayers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ):
        for block in self.blocks:
            x = block(x, mask)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = Attention(
            config.attn_d_k,
            dropout=config.dropout
        )
        self.dropout = config.dropout
        self.layer_norm1 = nn.LayerNorm(config.attn_d_k)
        self.cross_attention_queries = nn.Linear(config.attn_d_k, config.attn_d_k)
        self.cross_attention_keys = nn.Linear(config.attn_d_k, config.attn_d_k)
        self.cross_attention_values = nn.Linear(config.attn_d_k, config.attn_d_k)
        self.cross_attention_projection = nn.Linear(config.attn_d_k, config.attn_d_k)
        self.layer_norm2 = nn.LayerNorm(config.attn_d_k)
        self.ffn = nn.Sequential(
            nn.Linear(config.attn_d_k, config.transformer_proj_dim),
            nn.ReLU(),
            nn.Linear(config.transformer_proj_dim, config.attn_d_k),
            nn.Dropout(config.dropout)
        )
        self.layer_norm3 = nn.LayerNorm(config.attn_d_k)

    def forward(
        self, 
        x: torch.Tensor,
        memory: torch.Tensor, 
        src_mask: torch.Tensor,
        memory_mask: torch.Tensor
    ):
        x = x + self.attention(self.layer_norm1(x), mask=src_mask, cache=None)
        x = x + forward_attention(
            self.cross_attention_queries(self.layer_norm2(x)),
            self.cross_attention_keys(memory), 
            self.cross_attention_values(memory),
            self.cross_attention_projection,
            self.dropout,
            memory_mask
        )
        x = x + self.ffn(self.layer_norm3(x))
        return x
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = Attention(
            config.attn_d_k,
            dropout=config.dropout
        )
        self.layer_norm1 = nn.LayerNorm(config.attn_d_k)
        self.ffn = nn.Sequential(
            nn.Linear(config.attn_d_k, config.transformer_proj_dim),
            nn.ReLU(),
            nn.Linear(config.transformer_proj_dim, config.attn_d_k),
            nn.Dropout(config.dropout)
        )
        self.layer_norm2 = nn.LayerNorm(config.attn_d_k)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x + self.attention(self.layer_norm1(x), mask=mask, cache=None)
        x = x + self.ffn(self.layer_norm2(x))
        return x

if __name__ == "__main__":
    config = TransformerConfig(
        2, 32, 64, 0.2, 2, False, 24, 2, True, True
    )
    transformer = Transformer(
        config
    )
    out = transformer(torch.randn(1, 12, 2))
    print(out.shape)
