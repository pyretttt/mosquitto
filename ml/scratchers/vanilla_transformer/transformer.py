from typing import Callable, Optional
from dataclasses import dataclass

import torch, torch.nn as nn

from scratchers.positional_embeddings import PositionalEmbeddings
from scratchers.transformer_config import TransformerConfig

class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        attn_factory: Callable[[TransformerConfig], nn.Module]
    ):
        super().__init__()
        self.attn = attn_factory(config)
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.attn_d_k, config.transformer_proj_dim),
            nn.ReLU(),
            nn.Linear(config.transformer_proj_dim, config.attn_d_k),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.attn_d_k)
        self.layer_norm2 = nn.LayerNorm(config.attn_d_k)

    def init_keys(self, keys):
        self.attn.init_keys(keys)


    def forward(self, x: torch.tensor):
        """forward part of Transformer Decoder

        Args:
            x (torch.tensor): B x S x F shape
        
        Returns:
            x (torch.tensor): B x S x F shape
        """
        if self.config.pre_layer_norm:
            x = x + self.attn(self.layer_norm1(x))
            x = x + self.projection(self.layer_norm2(x))
        else:
            x = self.layer_norm1(x + self.attn(x))
            x = self.layer_norm2(x + self.projection(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig, attn_factory: Callable[[TransformerConfig], nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config=config,
                attn_factory=attn_factory
            ) for _ in range(config.nlayers)
        ])
        self.config = config
        self.pemb = PositionalEmbeddings(
            max_len=config.max_seq_len, d_k=config.attn_d_k
        )
        self.embedding_proj1 = nn.Linear(config.input_size, config.attn_d_k)
        self.embedding_proj2 = nn.Linear(config.attn_d_k, config.input_size)

    def forward(self, x: torch.Tensor):
        """Forwards sequence

        Args:
            x (torch.Tensor): shifted sequence
        """
        x = self.embedding_proj1(x)
        x = self.pemb(x)
        for block in self.blocks:
            block.init_keys(x)
            x = block(x)
        return self.embedding_proj2(x)

    def predict(self, x: torch.Tensor):
        """Predicts next token

        Args:
            x (torch.Tensor): input sequence
        """
        return self(x)[:, -1:, :]
