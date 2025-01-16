import torch, torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        d_k: int,
        bias: bool = True,
        is_self_attn: bool = False
    ):
        super().__init__()
        self.d_k = d_k
        self.input_size = input_size
        self.queries_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.keys_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.values_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.linear = nn.Linear(d_k, input_size, bias=bias)
        self.is_self_attn = is_self_attn
        
    
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
        queries = self.queries_proj(x) # (B x S x d_k)
        if self.is_self_attn:
            self.init_keys(x)
        
        attn_scores = torch.matmul(
            queries, 
            torch.permute(self.keys, (0, 2, 1))
        ) / (self.d_k ** 0.5) # (B x S_source x S_target)
        if mask:
            attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_scores = torch.softmax(attn_scores, dim=-2) # over source sequence
        
        ctx_sequence = torch.matmul(attn_scores, self.values) # B x S x d_k
        out = self.linear(ctx_sequence) # B x S x F
        
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        d_k: int,
        nheads: int,
        bias: bool = True,
        is_self_attn: bool = False
    ):
        super().__init__()
        assert d_k % nheads == 0, "KVS dimension should be divisible by number of attention heads"
        
        self.d_k = d_k
        self.input_size = input_size
        self.nheads = nheads
        self.queries_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.keys_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.values_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.linear = nn.Linear(d_k, input_size)
        self.is_self_attn = is_self_attn
        
    
    def init_keys(self, x):
        self.keys = self.multihead_proj(self.keys_proj, x) # (B x H x S x d_k)
        self.values = self.multihead_proj(self.values_proj, x) # (B x H x S x d_k)


    def multihead_proj(
        self, 
        kvs: nn.Linear, 
        x: torch.Tensor
    ):
        # (B x S x d_k) ~> (B x H x S x d_k / H)
        B, S, _ = x.shape
        out = kvs(x)
        out = torch.reshape(out, (B, S, self.nheads, int(self.d_k // self.nheads)))
        out = out.permute(0, 2, 1, 3)
        return out
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            x (torch.Tensor): shape (Batch x Sequence x Feature)
            mask (torch.Tensor): shape (Batch x Sequence x Sequence)

        Returns:
            torch.Tensor: shape (B x S x F)
        """
        queries = self.multihead_proj(self.queries_proj, x) # (B x H x S x d_k // n_heads)
        if self.is_self_attn:
            self.init_keys(x)
        attn_scores = torch.matmul(
            queries, 
            torch.permute(self.keys, (0, 1, 3, 2))
        ) / (self.d_k ** 0.5) # (B x H x S_source x S_target)
        if mask:
            attn_scores.masked_fill(mask == 0, -1e9)

        attn_scores = torch.softmax(attn_scores, dim=-2) # over source sequence

        ctx_sequence = torch.matmul(attn_scores, self.values) # B x H x S x d_k // n_heads
        ctx_sequence = ctx_sequence.permute(0, 2, 1, 3).flatten(-2) # B x S x d_k
        out = self.linear(ctx_sequence) # B x S x F
        
        return out


if __name__ == "__main__":
    attn = MultiheadAttention(64, 32, 4, is_self_attn=True)
    out = attn(torch.randn(4, 2, 64))
    print(out.shape)
    # out = attn.multihead_forward(attn.queries_proj, torch.randn(6, 6, 64))
    # print(out[0].shape)
    # print(len(out))