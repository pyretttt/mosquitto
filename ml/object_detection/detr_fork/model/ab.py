import math
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass


@dataclass
class AB:
    custom_get_spatial_position_embedding: bool = True
    custom_encoder: bool = True
    custom_decoder: bool = True
    custom_mha: bool = True


def get_spatial_position_embedding(emb_dim, feat_map):
    H, W = feat_map.shape[2], feat_map.shape[3]
    vertical_indices = torch.arange(H, dtype=torch.float32, device=feat_map.device)
    horizontal_indices = torch.arange(W, dtype=torch.float32, device=feat_map.device)
    grid = torch.meshgrid(vertical_indices, horizontal_indices, indexing="ij")  # (H,) (W,) -> (H, W), (H, W)
    # grid[0] - all rows elements have same scalar, from 0 to H vertically
    grid_h = grid[0].reshape(-1).unsqueeze(1)  # H * W, 1
    # grid[1] - all cols elements have same scalar, from 0 to W horizontally
    grid_w = grid[1].reshape(-1).unsqueeze(1)  # H * W, 1
    assert grid_h.shape == (H * W, 1) and grid_w.shape == (H * W, 1)

    factor = torch.exp(
        -np.log(10000) * torch.arange(0, end=emb_dim // 4, device=feat_map.device).float() / (emb_dim // 4)
    )  # (emb_dim // 4)
    assert factor.shape == (emb_dim // 4,)

    grid_h_emb = grid_h * factor  # H * W, embed_dim // 4
    grid_w_emb = grid_w * factor  # H * W, embed_dim // 4
    assert grid_h_emb.shape == grid_w_emb.shape and grid_h_emb.shape == (H * W, emb_dim // 4)

    # for each `y` and `emb_dim` inside feature map contains sin and cosine
    pos_h = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)  # H * W, embed_dim // 2

    # for each `x` and `emb_dim` inside feature map contains sin and cosine
    pos_w = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)  # H * W, embed_dim // 2

    pos = torch.cat([pos_h, pos_w], dim=-1)  # H * W, embed_dim

    return pos


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, d_model: int, ff_inner_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attns = nn.ModuleList(
            [
                MultiheadAttention(d_model=d_model, nheads=num_heads, dropout=dropout_prob, batch_first=True)
                if AB.custom_mha
                else nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim), nn.Dropout(dropout_prob), nn.ReLU(), nn.Linear(ff_inner_dim, d_model)
                )
                for _ in range(num_layers)
            ]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
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
    def __init__(self, num_layers: int, num_heads: int, d_model: int, ff_inner_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.self_attns = nn.ModuleList(
            [
                MultiheadAttention(d_model=d_model, nheads=num_heads, dropout=dropout_prob, batch_first=True)
                if AB.custom_mha
                else nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.cross_attns = nn.ModuleList(
            [
                MultiheadAttention(d_model=d_model, nheads=num_heads, dropout=dropout_prob, batch_first=True)
                if AB.custom_mha
                else nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim), nn.Dropout(dropout_prob), nn.ReLU(), nn.Linear(ff_inner_dim, d_model)
                )
                for _ in range(num_layers)
            ]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.cross_attn_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_objects: torch.Tensor,
        encoder_output: torch.Tensor,
        query_embedding: torch.Tensor,
        pos_emb: torch.Tensor,
    ):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attns = []

        for i in range(self.num_layers):
            in_emb = self.attn_norms[i](out)
            q = in_emb + query_embedding
            k = in_emb + query_embedding
            self_attn, _ = self.self_attns[i](query=q, key=k, value=in_emb)
            self_attn = self.attn_dropouts[i](self_attn)
            out = out + self_attn

            in_emb = self.cross_attn_norms[i](out)
            q = in_emb + query_embedding
            k = encoder_output + pos_emb
            cross_attn, decoder_cross_attn_weight = self.cross_attns[i](query=q, key=k, value=encoder_output)
            decoder_cross_attns.append(decoder_cross_attn_weight)
            out_attn = self.cross_attn_dropouts[i](cross_attn)
            out = out + out_attn

            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ffn_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))

        return torch.stack(decoder_outputs), torch.stack(decoder_cross_attn_weight)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        nheads: int,
        d_model: int,
        batch_first: bool,
        dropout: float,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.nheads = nheads
        self.d_model = d_model
        self.batch_first = batch_first
        self.dropout = dropout
        assert d_model % nheads == 0, "d_model must be divisible by number of heads for multihead attention"
        self.head_dim = d_model // nheads
        self.q_proj = nn.Linear(3 * d_model, d_model, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.attn_dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, tgt_len = query.shape[:2]
        src_len = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(bsz, tgt_len, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, src_len, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, src_len, self.nheads, self.head_dim).transpose(1, 2)
        # N, N_heads, S_src/S_tgt, head_dim

        # N, N_heads, S_tgt, head_dim @ N, N_heads, head_dim, S_src -> N, N_heads, S_tgt, S_src
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            # Key padding mask is N, S_src
            mask = key_padding_mask[:, None, None, :].to(torch.bool)  # N, 1, 1, S_src
            scores = scores.masked_fill(
                mask,
                float("-inf"),
            )  # fills src components, such that tgt sequence do not attend to them

        if is_causal:
            # Actually all elements on main diagona and below it, are not masked
            causal_mask = torch.ones(tgt_len, src_len, dtype=torch.bool, device=scores.device).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(
            attn_weights,
            v,
        )  #  N, N_heads, S_tgt, S_src @ N, N_heads, S_src, head_dim -> N, N_heads, S_tgt, head_dim
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz,
            tgt_len,
            self.d_model,
        )  # N, N_heads, S_tgt, head_dim -> N, S_tgt, N_heads, head_dim ->  N, S_tgt, d_model
        attn_output = self.out_proj(attn_output)  # N, S_tgt, d_model

        if not need_weights:
            attn_weights_to_return = None
        else:
            attn_weights_to_return = attn_weights
            if average_attn_weights:
                attn_weights_to_return = attn_weights.mean(dim=1)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights_to_return
