import torch
from torch import nn
import numpy as np

def get_spatial_position_embedding(emb_dim, feat_map):
    H, W = feat_map.shape[2], feat_map.shape[3]
    vertical_indices = torch.arange(H, dtype=torch.float32, device=feat_map.device)
    horizontal_indices = torch.arange(W, dtype=torch.float32, device=feat_map.device)
    grid = torch.meshgrid(vertical_indices, horizontal_indices, indexing="ij") # (H,) (W,) -> (H, W), (H, W)
    grid_h = grid[0].reshape(-1).unsqueeze(1) # H * W, 1
    grid_w = grid[1].reshape(-1).unsqueeze(1) # H * W, 1
    assert grid_h.shape == (H * W, 1) and grid_w.shape == (H * W, 1)

    factor = torch.exp(
        -np.log(10000)
        * torch.arange(0, end=emb_dim // 4, device=feat_map.device).float()
        / (emb_dim // 4)
    ) # (emb_dim // 4)
    assert factor.shape == (emb_dim // 4, )

    grid_h_emb = grid_h * factor # H * W, embed_dim // 4
    grid_w_emb = grid_w * factor # H * W, embed_dim // 4
    assert grid_h_emb.shape == grid_w_emb.shape and grid_h_emb.shape == (H * W, emb_dim // 4)

    print(grid_h)

    pos_h = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1) # H * W, embed_dim // 2

    pos_w = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1) # H * W, embed_dim // 2

    pos = torch.cat([
        pos_h,
        pos_w
    ], dim=-1) # H * W, embed_dim

    return pos


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_inner_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ff_inner_dim, d_model)
            )
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.attn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.ffn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb):
        out = x
        attn_weights = []
        for i in range(self.num_layers):
            in_emb = self.attn_norms[i](out)
            q = in_emb + pos_emb
            k = in_emb + pos_emb
            out_emb, attn_weights = self.attns[i](query=q, key=k, value=in_emb)
            attn_weights.append(attn_weights)
            out_emb = self.attn_dropouts[i](out_emb)
            out = out + out_emb

            ff_in = self.ff_norms[i](out)
            out_ff = self.ffs[i](ff_in)
            out_ff = self.ffn_dropouts[i](out)
            out = out + out_ff

        out = self.output_norm(out)
        return out, torch.stack(attn_weights)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_inner_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ff_inner_dim, d_model)
            )
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.attn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.cross_attn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.ffn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_objects: torch.Tensor,
        encoder_output: torch.Tensor,
        query_embedding: torch.Tensor,
        pos_emb: torch.Tensor
    ):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attns = []

        for i in range(self.num_layers):
            in_emb = self.attn_norms[i](out)
            q = in_emb + pos_emb
            k = in_emb + pos_emb
            self_attn, _ = self.self_attns[i](
                query=q,
                key=k,
                value=in_emb
            )
            self_attn = self.attn_dropouts[i](self_attn)
            out = out + self_attn

            in_emb = self.cross_attn_norms[i](out)
            q = in_emb + query_embedding
            k = encoder_output + pos_emb
            cross_attn, decoder_cross_attn_weight = self.cross_attns[i](
                query=q,
                key=k,
                value=encoder_output
            )
            decoder_cross_attns.append(decoder_cross_attn_weight)
            out_attn = self.cross_attn_dropouts[i](cross_attn)
            out = out + out_attn

            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ffn_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))

        return torch.stack(decoder_outputs), torch.stack(decoder_cross_attn_weight)