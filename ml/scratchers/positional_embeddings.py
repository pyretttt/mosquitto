import math

import torch, torch.nn as nn

class PositionalEmbeddings(nn.Module):
    def __init__(self, max_len: int, d_k: int):
        super().__init__()
        self.max_len = max_len
        self.d_k = d_k
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        angular_speeds = torch.exp(
			torch.arange(0, d_k, 2, dtype=torch.float) * (-math.log(10_000) /  d_k)
		) # \frac{1}{10000^{(0, 2, 4, 6, ...) * d_k}}

        pe = torch.zeros(max_len, d_k)
        pe[:, ::2] = torch.sin(angular_speeds * positions)
        pe[:, 1::2] = torch.cos(angular_speeds * positions)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x = x * math.sqrt(self.d_k)
        return x + self.pe[:, :x.size(-2), :]
