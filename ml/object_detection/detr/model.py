import torch

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float
    ):
        self.self_attention = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)


    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        src2 = self.norm1(src)
        k = src + pos if pos is not None else pos
        q = src + pos if pos is not None else pos

        src2 = self.self_attention(q, k, values=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(
            self.dropout(
                torch.function.relu(
                    self.linear1(src2)
                )
            )
        )
        src = src + self.dropout2(src2)
        return src