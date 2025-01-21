from typing import Callable, Optional
from dataclasses import dataclass

import torch, torch.nn as nn

@dataclass
class Config:
    input_size: int
    attn_d_k: int
    transformer_proj_dim: int
    dropout: float
    nlayers: int
    is_self_attn: bool
    max_seq_len: int


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: Config,
        attn_factory: Callable[[Config], nn.Module]
    ):
        super().__init__()
        self.attn = attn_factory(config)
        self.config = config
        self.projection = nn.ModuleList([
            nn.Linear(config.input_size, config.transformer_proj_dim),
            nn.ReLU(),
            nn.Linear(config.transformer_proj_dim, config.input_size),
            nn.Dropout(config.input_size)
        ])
        self.layer_norm1 = nn.LayerNorm(config.input_size)
        self.layer_norm2 = nn.LayerNorm(config.input_size)

    def init_keys(self, keys):
        self.attn.init_keys(keys)


    def forward(self, x: torch.tensor):
        """forward part of Transformer Decoder

        Args:
            x (torch.tensor): B x S x F shape
        
        Returns:
            x (torch.tensor): B x S x F shape
        """
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.projection(self.layer_norm2(x))
        
        return x


# Forward train
# Forward predict

class TransformerDecoder(nn.Module):
    def __init__(self, config: Config, attn_factory: Callable[[Config], nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config=config,
                attn_factory=attn_factory
            ) for _ in range(config.nlayers)
        ])
        self.config = config
    

    def forward(self, x: torch.Tensor, source_seq: torch.Tensor):
        """Forwards sequence

        Args:
            x (torch.Tensor): shifted sequence
            source_seq (torch.Tensor): index of last source sequence element
        """
        if not self.config.is_self_attn:
           for block in self.blocks:
               block.init_keys(x[:, :source_seq, :]) 
        out = self.blocks(x[:, source_seq-1:-1, :])
        return out


    def predict(self, x: torch.Tensor, len: Optional[int]):
        if not self.config.is_self_attn:
           for block in self.blocks:
               block.init_keys(x)
        inputs = x[:, -1, :]
        for i in range(min(len, self.config.max_seq_len)):
            inputs = torch.cat((inputs, self.blocks(inputs)), dim=-2) 
            
        return inputs[:, 1:, :]