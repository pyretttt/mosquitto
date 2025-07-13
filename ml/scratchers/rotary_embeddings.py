import torch
import torch.nn as nn

class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int, 
        max_seq_len: int = 4096,
        base: int = 10_000
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[:self.dim // 2] / self.dim)
        ) # (self.dim // 2)
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()
    
    def build_rope_cache(self):
        seq_idx = torch.arange(
            self.max_seq_len, 
            dtype=self.theta.dtype, 
            device=self.theta.device
        ) # (max_seq_len)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float() # (max_seq_len, self.dim // 2)
        cache = torch.stack((idx_theta.cos(), idx_theta.sin()), dim=-1) # (max_seq_len, self.dim // 2, 2)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor):
        """Applies rotary embeddings to input embedding

        Args:
            x (torch.Tensor): input embedding with shape (batch_size, seq_len, dim)
        """
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] # (seq_len, self.dim // 2, 2)

        xshaped = x.view(*x.shape[:-1], -1, 2) # (batch_size, seq_len, dim // 2, 2)
        rope_cache = rope_cache.unsqueeze(0) # (1, seq_len, self.dim // 2, 2)

        x_out = torch.stack([
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], # x cos(m) - y sin(m)
                xshaped[..., 0] * rope_cache[..., 1] + xshaped[..., 1] * rope_cache[..., 0], # x sin(m) + y cos(m)
            ],
            dim=-1
        ) # (batch_size, seq_len, dim // 2, 2)

        x_out = x.flatten(-1) # (batch_size, seq_len, dim)

        return x_out