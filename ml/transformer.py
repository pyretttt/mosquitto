import torch, torch.nn as nn

from typing import Callable

# Dropouts
# Linear projection
# Layer norm
# Residuals
# Attn
# Activation fn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_k: int, 
        attn_factory: Callable[[int, int, float], nn.Module],
        linear_proj: int,
        dropout: int = 0.2
    ):
        super().__init__()
        self.attn = attn_factory(input_size, d_k, dropout)
        self.input_size = input_size
        self.d_k = d_k
        self.projection = nn.ModuleList([
            nn.Linear(input_size, linear_proj),
            nn.ReLU(),
            nn.Linear(linear_proj, input_size),
            nn.Dropout(input_size)
        ])
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)


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