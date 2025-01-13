import torch, torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        d_k: int,
        bias: bool = True
    ):
        super().__init__()
        self.d_k = d_k
        self.input_size = input_size
        self.queries_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.keys_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.values_proj = nn.Linear(input_size, d_k, bias=bias) # Weights with shape (d_k x I)
        self.linear = nn.Linear(d_k, input_size, bias=bias)
        
    
    def init_keys(self, x):
        self.keys = self.keys_proj(x) # (B x S x d_k)
        self.values = self.values_proj(x) # (B x S x d_k)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): shape (Batch x Sequence x Feature)

        Returns:
            torch.Tensor: shape (B x S x F)
        """
        queries = self.queries_proj(x) # (B x S x d_k)
        attn_scores = torch.matmul(
            queries, 
            torch.transpose(self.keys, 0, 2, 1)
        ) / (self.d_k ** 0.5) # (B x S x S)
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
        bias: bool = True
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
        
    
    def init_keys(self, x):
        self.keys = self.keys_proj(x) # (B x S x d_k)
        self.values = self.values_proj(x) # (B x S x d_k)


    def multihead_forward(
        self, 
        kvs: nn.Linear, 
        x: torch.Tensor
    ):
        # (B x S x d_k) ~> (B x H x S x d_k / H)
        #               ~> (B x S x H x d_k / H)
        #               ~> (H x B x S x d_k / H)
        out = kvs(x)
        return torch.reshape(out, ())
    
    
    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): shape (Batch x Sequence x Feature)

        Returns:
            torch.Tensor: shape (B x S x F)
        """
        queries = self.queries_proj(x) # (B x S x d_k)
        attn_scores = torch.matmul(
            queries, 
            torch.transpose(self.keys, 0, 2, 1)
        ) / (self.d_k ** 0.5) # (B x S x S)
        attn_scores = torch.softmax(attn_scores, dim=-2) # over source sequence
        
        ctx_sequence = torch.matmul(attn_scores, self.values) # B x S x d_k
        out = self.linear(ctx_sequence) # B x S x F
        
        return out


if __name__ == "__main__":
    attn = MultiheadAttention(64, 32, 4)
    out = attn.multihead_forward(attn.queries_proj, torch.randn(6, 6, 64))
    print(out[0].shape)
    print(len(out))