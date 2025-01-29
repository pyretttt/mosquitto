from typing import Optional

import torch
import torch.nn as nn

from scratchers.transformer_config import TransformerConfig
from scratchers.transformer.attn import Attention, Cache
from scratchers.positional_embeddings import PositionalEmbeddings

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.attn_d_k)
        self.layer_norm2 = nn.LayerNorm(config.attn_d_k)
        self.attention = Attention(
            config.attn_d_k,
            config.dropout,
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(config.attn_d_k, config.transformer_proj_dim),
            nn.ReLU(),
            nn.Linear(config.transformer_proj_dim, config.attn_d_k),
            nn.Dropout()
        )

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: torch.Tensor,
        cache: Optional[Cache]
    ):
        attn_out = self.attention(self.layer_norm1(x), mask=attn_mask, cache=cache)
        x = x[:, -attn_out.size(-2):, :] + attn_out
        x = x + self.linear_layer(self.layer_norm2(x))
        return x

    
class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear_projector1 = nn.Linear(config.input_size, config.attn_d_k)
        self.pemb = PositionalEmbeddings(config.max_seq_len, config.attn_d_k)
        self.attn_blocks = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.nlayers)
        ])
        self.linear_projector2 = nn.Linear(config.attn_d_k, config.input_size)
        self.use_cache = config.use_cache

    @staticmethod
    def self_attn_mask(shape: tuple[int]):
        return torch.triu(
            torch.ones(shape[-2:])
        ).view(*shape)

    def forward(self, x):
        x = self.linear_projector1(x)
        x = self.pemb(x)
        for block in self.attn_blocks:
            x = block(
                x, 
                TransformerDecoder.self_attn_mask((1, x.size(-2), x.size(-2))), 
                cache=None
            )
        
        x = self.linear_projector2(x)
        return x

    def predict(self, x, max_seq):
        self.eval()
        cache = [Cache() for _ in range(len(self.attn_blocks))]
        
        inputs = x
        for i in range(max_seq):
            out = self.linear_projector1(inputs)
            out = self.pemb(out)
            for idx, block in enumerate(self.attn_blocks):
                out = block(
                    out, 
                    TransformerDecoder.self_attn_mask((1, out.size(-2), out.size(-2))), 
                    cache=cache[idx]
                )
            
            out = self.linear_projector2(out)
            inputs = torch.concat((inputs, out[:, -1:, :]), dim=-2)
        
        return inputs[:, x.size(-2):, :]

if __name__ == "__main__":
    config = TransformerConfig(
        2, 32, 64, 0.2, 1, False, 24, 1, True, True
    )
    decoder = TransformerDecoder(config)
    out = decoder.predict(torch.randn(1, 12, 2), 12)
    print(out.shape)