from typing import Callable, Optional
import math

import torch, torch.nn as nn

class PositionalEmbeddings(nn.Module):
    def __init__(self, max_len: int, d_k: int):
        super().__init__()
        self.max_len = max_len
        self.d_k = d_k
        positions = torch.arange(0, max_len, dtype=float).unsqueeze(1)
        angular_speeds = torch.exp(
			torch.arange(0, d_k, 2, dtype=float) * (-math.log(10_000) /  d_k)
		) # \frac{1}{10000^{(0, 2, 4, 6, ...) * d_k}}

        pe = torch.zeros(max_len, d_k)
        pe[:, ::2] = torch.sin(angular_speeds * positions)
        pe[:, 1::2] = torch.cos(angular_speeds * positions)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x = x * math.sqrt(self.d_k)
        # print(x.shape)
        # print(self.pe[:, : x.size(1)].shape)
        return x + self.pe[:, :x.size(1)]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
