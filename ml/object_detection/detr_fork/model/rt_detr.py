import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import mobilenet_v3_large
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def get_spatial_position_embedding(pos_emb_dim: int, feat_map: torch.Tensor) -> torch.Tensor:
    """
    2D sine-cosine positional embedding for a feature map.

    Args:
        pos_emb_dim: Embedding dimension (must be divisible by 4).
        feat_map: Tensor of shape (B, C, H, W).

    Returns:
        Tensor of shape (H*W, pos_emb_dim) to be broadcast to batch.
    """
    assert pos_emb_dim % 4 == 0, "Position embedding dimension must be divisible by 4"
    H, W = feat_map.shape[2], feat_map.shape[3]
    device = feat_map.device

    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")  # (H, W), (H, W)
    gh = gh.reshape(-1)
    gw = gw.reshape(-1)

    # factor = 10000^(2i/d_model)
    denom = 10000 ** (
        torch.arange(start=0, end=pos_emb_dim // 4, dtype=torch.float32, device=device) / (pos_emb_dim // 4)
    )

    emb_h = (gh[:, None] / denom).to(torch.float32)
    emb_w = (gw[:, None] / denom).to(torch.float32)

    emb_h = torch.cat([torch.sin(emb_h), torch.cos(emb_h)], dim=-1)  # (H*W, pos_emb_dim//2)
    emb_w = torch.cat([torch.sin(emb_w), torch.cos(emb_w)], dim=-1)  # (H*W, pos_emb_dim//2)
    pos = torch.cat([emb_h, emb_w], dim=-1)  # (H*W, pos_emb_dim)
    return pos


class TransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder (pre-norm) with MultiheadAttention and 2-layer FFN.
    """

    def __init__(self, num_layers: int, num_heads: int, d_model: int, ff_inner_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout1 = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.dropout2 = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x: (B, S, D), pos: (S, D)
        out = x
        attn_weights_all = []
        for i in range(self.num_layers):
            in_emb = self.norm1[i](out)
            q = in_emb + pos  # add 2D position to q/k
            k = in_emb + pos
            attn_out, attn_weights = self.attn[i](q, k, in_emb)
            out = out + self.dropout1[i](attn_out)

            ff_in = self.norm2[i](out)
            ff_out = self.ff[i](ff_in)
            out = out + self.dropout2[i](ff_out)
            attn_weights_all.append(attn_weights)

        return self.out_norm(out), torch.stack(attn_weights_all)


class TransformerDecoder(nn.Module):
    """
    Lightweight Transformer decoder (pre-norm) with self-attention, cross-attention and 2-layer FFN.
    """

    def __init__(self, num_layers: int, num_heads: int, d_model: int, ff_inner_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_self = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_cross = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.drop_self = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.drop_cross = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.drop_ff = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_objects: torch.Tensor,
        encoder_out: torch.Tensor,
        query_embed: torch.Tensor,
        pos: torch.Tensor,
    ):
        # query_objects: (B, Q, D), query_embed: (B, Q, D)
        out = query_objects
        outputs = []
        cross_weights_all = []
        for i in range(self.num_layers):
            # self-attn on queries
            in_emb = self.norm_self[i](out)
            q = in_emb + query_embed
            k = in_emb + query_embed
            v = in_emb
            sa_out, _ = self.self_attn[i](q, k, v)
            out = out + self.drop_self[i](sa_out)

            # cross-attn over encoder memory
            q2 = self.norm_cross[i](out)
            q2 = q2 + query_embed
            k2 = encoder_out + pos
            ca_out, ca_weights = self.cross_attn[i](q2, k2, encoder_out)
            out = out + self.drop_cross[i](ca_out)

            # FFN
            ff_in = self.norm_ff[i](out)
            ff_out = self.ff[i](ff_in)
            out = out + self.drop_ff[i](ff_out)

            outputs.append(self.out_norm(out))
            cross_weights_all.append(ca_weights)

        return torch.stack(outputs), torch.stack(cross_weights_all)


class Lazy1x1Conv(nn.Module):
    """
    Lazy 1x1 conv that initializes on first forward based on input channels.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            in_ch = x.shape[1]
            self.conv = nn.Conv2d(in_ch, self.out_channels, kernel_size=1)
            self.conv.to(x.device, x.dtype)
        return self.conv(x)


class MultiScaleBackbone(nn.Module):
    """
    Wrap a backbone Sequential to produce last 3 pyramid features (higher to lower resolution).
    Uses stride changes to pick scales; falls back gracefully if <3.
    """

    def __init__(self, backbone_seq: nn.Sequential, num_scales: int = 3):
        super().__init__()
        self.body = backbone_seq
        self.num_scales = num_scales

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        h_prev, w_prev = x.shape[2], x.shape[3]
        out = x
        for m in self.body:
            out = m(out)
            h, w = out.shape[2], out.shape[3]
            if h < h_prev or w < w_prev:
                feats.append(out)
                h_prev, w_prev = h, w

        if len(feats) == 0:
            feats = [out for _ in range(self.num_scales)]
        elif len(feats) >= self.num_scales:
            feats = feats[-self.num_scales :]
        else:
            # if fewer scales, repeat last
            while len(feats) < self.num_scales:
                feats.insert(0, feats[0])
        # return from high-res to low-res (P3, P4, P5)
        return feats[-self.num_scales :][::-1]


class HybridEncoder(nn.Module):
    """
    Hybrid encoder with per-level local DWConv branch and cross-scale global attention fusion.
    Operates on list of multi-scale maps (B, D, H_l, W_l).
    """

    def __init__(
        self, num_layers: int, num_heads: int, d_model: int, ff_inner_dim: int, dropout_prob: float, num_levels: int
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = num_levels

        # local conv branch per layer/level
        self.local_dw = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                            nn.Conv2d(d_model, d_model, kernel_size=1),
                            nn.GELU(),
                            nn.Dropout(dropout_prob),
                        )
                        for _ in range(num_levels)
                    ]
                )
                for _ in range(num_layers)
            ]
        )

        # cross-scale attention + FFN per layer
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.drop1 = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.drop2 = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])

        # learned level embeddings to distinguish scales
        self.level_embed = nn.Parameter(torch.randn(num_levels, d_model))

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        # feats: list of (B, D, H, W), length = L
        B = feats[0].shape[0]
        L = len(feats)
        assert L == self.num_levels

        attn_weights_all = []
        for i in range(self.num_layers):
            # local conv branch
            feats_local = []
            for l in range(L):
                x = feats[l]
                x = self.local_dw[i][l](x) + feats[l]
                feats_local.append(x)

            # tokens + pos per level
            tokens = []
            poss = []
            sizes = []
            for l in range(L):
                B, D, H, W = feats_local[l].shape
                sizes.append((H, W))
                t = feats_local[l].permute(0, 2, 3, 1).reshape(B, H * W, D)
                p = get_spatial_position_embedding(D, feats_local[l]) + self.level_embed[l].unsqueeze(0).expand(
                    H * W, -1
                )
                tokens.append(t)
                poss.append(p)

            tokens_cat = torch.cat(tokens, dim=1)
            pos_cat = torch.cat(poss, dim=0)  # (sum_S, D)

            q = self.norm1[i](tokens_cat) + pos_cat
            k = q
            v = self.norm1[i](tokens_cat)
            attn_out, attn_w = self.attn[i](q, k, v)
            tokens_cat = tokens_cat + self.drop1[i](attn_out)

            ff_in = self.norm2[i](tokens_cat)
            tokens_cat = tokens_cat + self.drop2[i](self.ff[i](ff_in))
            attn_weights_all.append(attn_w)

            # split back to levels and reshape to maps
            split_sizes = [s[0] * s[1] for s in sizes]
            tokens_split = list(torch.split(tokens_cat, split_sizes, dim=1))
            new_feats = []
            offset = 0
            for l, (H, W) in enumerate(sizes):
                t = tokens_split[l].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                new_feats.append(t)
                offset += H * W
            feats = new_feats

        return feats, torch.stack(attn_weights_all)


class MSDeformAttn(nn.Module):
    """
    Minimal multi-scale deformable attention that samples around reference points on multi-scale maps.
    """

    def __init__(self, d_model: int, num_heads: int, num_levels: int, num_points: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = d_model // num_heads

        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(
        self, query: torch.Tensor, feats: list[torch.Tensor], reference_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query: (B, Q, D)
        # feats: list of (B, D, H_l, W_l)
        # reference_points: (B, Q, 2) in [0,1]
        B, Q, D = query.shape
        Hs = [f.shape[2] for f in feats]
        Ws = [f.shape[3] for f in feats]

        # predict offsets and attn weights
        offsets = self.sampling_offsets(query)  # (B, Q, H*L*P*2)
        offsets = offsets.view(B, Q, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = torch.tanh(offsets)  # bound offsets in [-1,1] normalized grid space

        attn = self.attention_weights(query)  # (B, Q, H*L*P)
        attn = attn.view(B, Q, self.num_heads, self.num_levels * self.num_points)
        attn = attn.softmax(dim=3)
        attn = attn.view(B, Q, self.num_heads, self.num_levels, self.num_points)

        # prepare per-head features
        sampled_list = []
        weights_list = []
        for lvl, feat in enumerate(feats):
            # (B, D, H, W) -> (B*H, head_dim, H, W) with heads folded into batch for grid_sample
            B_, D_, H_, W_ = feat.shape
            feat = feat.view(B_, self.num_heads, self.head_dim, H_, W_)
            feat = feat.permute(0, 1, 2, 3, 4).reshape(B_ * self.num_heads, self.head_dim, H_, W_)

            # base grid from reference points
            ref = reference_points  # (B, Q, 2) in [0,1]
            ref_grid = ref * 2.0 - 1.0  # to [-1,1]
            off = offsets[:, :, :, lvl]  # (B, Q, heads, P, 2)
            off = off.permute(0, 2, 1, 3, 4)  # (B, heads, Q, P, 2)
            grid = ref_grid[:, None, :, None, :].expand(B, self.num_heads, Q, self.num_points, 2) + off
            grid = grid.reshape(B * self.num_heads, Q * self.num_points, 2)
            grid = grid.view(B * self.num_heads, Q * self.num_points, 1, 2)

            # grid_sample expects (N, Hout, Wout, 2)
            grid = grid.permute(0, 2, 1, 3)  # (B*H, 1, Q*P, 2)
            sampled = F.grid_sample(
                feat,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # (B*H, head_dim, 1, Q*P)
            sampled = sampled.squeeze(2)  # (B*H, head_dim, Q*P)
            sampled = sampled.view(B, self.num_heads, self.head_dim, Q, self.num_points)

            w = attn[:, :, :, lvl]  # (B, Q, heads, P)
            w = w.permute(0, 2, 3, 1)  # (B, heads, P, Q)
            w = w.unsqueeze(2)  # (B, heads, 1, P, Q)
            sampled_list.append(sampled)
            weights_list.append(w)

        # aggregate over levels and points
        agg = 0.0
        for lvl in range(self.num_levels):
            sampled = sampled_list[lvl]  # (B, heads, head_dim, Q, P)
            w = weights_list[lvl]  # (B, heads, 1, P, Q)
            sampled = sampled.permute(0, 1, 2, 4, 3)  # (B, heads, head_dim, P, Q)
            contrib = (sampled * w).sum(dim=3)  # sum over P -> (B, heads, head_dim, Q)
            agg = agg + contrib

        agg = agg.permute(0, 3, 1, 2).reshape(B, Q, self.d_model)  # (B, Q, D)
        out = self.out_proj(agg)
        out = self.dropout(out)
        # return attention weights for debugging: shape (layers? not here). We return flattened attn per level/point
        return out, attn


def get_normalized_reference_points(feat: torch.Tensor) -> torch.Tensor:
    """
    Build normalized [0,1] reference points grid for a feature map.
    Returns (B, H*W, 2) with (x, y) in [0,1].
    """
    B, D, H, W = feat.shape
    device = feat.device
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    ref = torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2).repeat(B, 1, 1)
    return ref


class EfficientHybridEncoder(nn.Module):
    """
    Hybrid encoder with per-level local DWConv branch and multi-scale deformable attention fusion.
    Avoids O(S^2) global attention by using MSDeformAttn per level.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_inner_dim: int,
        dropout_prob: float,
        num_levels: int,
        num_points: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = num_levels

        # local conv branch per layer/level
        self.local_dw = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                            nn.Conv2d(d_model, d_model, kernel_size=1),
                            nn.GELU(),
                            nn.Dropout(dropout_prob),
                        )
                        for _ in range(num_levels)
                    ]
                )
                for _ in range(num_layers)
            ]
        )

        # deformable attention per layer (shared across levels within a layer)
        self.cross_deform = nn.ModuleList(
            [
                MSDeformAttn(d_model, num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout_prob)
                for _ in range(num_layers)
            ]
        )

        # FFN and norms per layer
        self.norm_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ]
        )
        self.drop_attn = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.drop_ff = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])

        # learned level embeddings
        self.level_embed = nn.Parameter(torch.randn(num_levels, d_model))

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        B = feats[0].shape[0]
        L = len(feats)
        assert L == self.num_levels
        attn_weights_layers = []

        for i in range(self.num_layers):
            # local branch
            feats_local = [self.local_dw[i][l](feats[l]) + feats[l] for l in range(L)]

            # deformable cross-scale fusion per level
            updated_feats = []
            attn_this_layer = []
            for l in range(L):
                B_, D_, H_, W_ = feats_local[l].shape
                tokens = feats_local[l].permute(0, 2, 3, 1).reshape(B_, H_ * W_, D_)
                pos = get_spatial_position_embedding(D_, feats_local[l]) + self.level_embed[l].unsqueeze(0).expand(
                    H_ * W_, -1
                )
                q = self.norm_q[i](tokens) + pos  # (B, S_l, D)
                ref = get_normalized_reference_points(feats_local[l])  # (B, S_l, 2)
                out_tok, attn = self.cross_deform[i](q, feats_local, ref)
                tokens = tokens + self.drop_attn[i](out_tok)
                ff_in = self.norm_ff[i](tokens)
                tokens = tokens + self.drop_ff[i](self.ff[i](ff_in))
                new_map = tokens.reshape(B_, H_, W_, D_).permute(0, 3, 1, 2).contiguous()
                updated_feats.append(new_map)
                attn_this_layer.append(attn)

            feats = updated_feats
            # stack attn per level
            attn_weights_layers.append(attn_this_layer)

        return feats, attn_weights_layers


class DeformableDecoder(nn.Module):
    """
    Decoder with self-attention + multi-scale deformable cross-attention.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_inner_dim: int,
        num_levels: int,
        num_points: int = 4,
        dropout_prob: float = 0.1,
        keep_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.keep_ratio = keep_ratio

        self.self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [
                MSDeformAttn(d_model, num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout_prob)
                for _ in range(num_layers)
            ]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_sa = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ca = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.drop_sa = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.drop_ca = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
        self.drop_ff = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])

        # reference points from query embedding
        self.ref_point_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        # lightweight objectness scorer for top-k selection
        self.score_head = nn.Linear(d_model, 1)
        # iterative reference point refinement per layer
        self.refine_mlp = nn.Linear(d_model, 2)

    def forward(
        self,
        query_objects: torch.Tensor,
        feats: list[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # query_objects: (B, Q, D); feats: list of (B, D, H, W); query_embed: (B, Q, D)
        out = query_objects
        q_embed = query_embed
        outputs: list[torch.Tensor] = []
        cross_attn_weights: list[torch.Tensor] = []

        # initial reference points from query embedding
        ref_points = self.ref_point_mlp(q_embed).sigmoid()  # (B, Q, 2)

        for i in range(self.num_layers):
            # self-attention
            sa_in = self.norm_sa[i](out)
            q = sa_in + q_embed
            k = sa_in + q_embed
            sa_out, _ = self.self_attn[i](q, k, sa_in)
            out = out + self.drop_sa[i](sa_out)

            # deformable cross-attention
            ca_in = self.norm_ca[i](out)
            ca_out, attn = self.cross_attn[i](ca_in + q_embed, feats, ref_points)
            out = out + self.drop_ca[i](ca_out)

            # FFN
            ff_in = self.norm_ff[i](out)
            out = out + self.drop_ff[i](self.ff[i](ff_in))

            # iterative reference point refinement (dx, dy scaled and added then clamped via sigmoid)
            delta = torch.tanh(self.refine_mlp(out)) * 0.5  # (B, Q, 2) small step in normalized grid
            ref_points = (ref_points + delta).clamp(0.0, 1.0)

            outputs.append(out)
            cross_attn_weights.append(attn)

            # IoU-aware top-k query selection (except last layer)
            if i < self.num_layers - 1:
                B, Q, D = out.shape
                keep = max(16, int(Q * self.keep_ratio))
                with torch.no_grad():
                    scores = torch.sigmoid(self.score_head(out)).squeeze(-1)  # (B, Q)
                    topk_idx = torch.topk(scores, k=keep, dim=1).indices  # (B, keep)

                # gather for next layer
                batch_idx = torch.arange(B, device=out.device)[:, None]
                out = out[batch_idx, topk_idx]
                q_embed = q_embed[batch_idx, topk_idx]
                ref_points = ref_points[batch_idx, topk_idx]

        return outputs, cross_attn_weights


class RTDETR(nn.Module):
    """
    A compact, self-contained RT-DETR-style model aimed at simplicity.

    Key features for real-time and clarity:
    - Lightweight backbone by default (MobileNetV3-Large features).
    - Multi-scale P3/P4/P5 features with efficient deformable fusion in the encoder.
    - Deformable attention decoder with iterative reference refinement and top-K pruning.
    - Quality-aware classification (QFL/VFL) + IoU head and DETR-style matching for boxes.

    Inputs/targets follow DETR conventions used in model/detr.py:
      - x: images tensor (B, 3, H, W)
      - targets: optional list of dicts with keys:
          'boxes': Tensor[K, 4] in xyxy normalized to [0, 1]
          'labels': Tensor[K] with class indices in [0, num_classes-1]
    """

    def __init__(
        self,
        num_classes: int,
        bg_class_idx: int = 0,
        num_queries: int = 100,
        d_model: int = 192,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        nheads: int = 4,
        ff_inner_dim: int = 384,
        dropout_prob: float = 0.1,
        backbone: str = "mobilenet_v3_large",
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        # matching + loss
        cls_cost_weight: float = 1.0,
        l1_cost_weight: float = 5.0,
        giou_cost_weight: float = 2.0,
        bg_class_weight: float = 0.1,
        nms_threshold: float = 0.7,
        # quality-aware classification
        quality_loss: str = "qfl",  # or "vfl"
        qfl_beta: float = 2.0,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        # top-k pruning ratio per decoder layer (constant)
        keep_ratio: float = 0.5,
    ):
        super().__init__()

        # basic params
        self.num_classes = num_classes
        self.bg_class_idx = bg_class_idx
        assert self.bg_class_idx in (0, num_classes - 1), "Background can only be 0 or num_classes-1"

        self.num_queries = num_queries
        self.d_model = d_model
        self.nms_threshold = nms_threshold

        # matching/loss weights
        self.cls_cost_weight = cls_cost_weight
        self.l1_cost_weight = l1_cost_weight
        self.giou_cost_weight = giou_cost_weight
        self.bg_cls_weight = bg_class_weight
        self.quality_loss = quality_loss
        self.qfl_beta = qfl_beta
        self.vfl_alpha = vfl_alpha
        self.vfl_gamma = vfl_gamma

        # backbone
        if backbone == "mobilenet_v3_large":
            weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            mb = mobilenet_v3_large(weights=weights)
            base = mb.features  # nn.Sequential
        elif backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            rn = torchvision.models.resnet18(weights=weights, norm_layer=torchvision.ops.FrozenBatchNorm2d)
            base = nn.Sequential(*list(rn.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = MultiScaleBackbone(base, num_scales=3)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # per-level projection to d_model
        self.input_proj = nn.ModuleList([Lazy1x1Conv(d_model) for _ in range(3)])

        # hybrid encoder (efficient deformable) and deformable decoder
        self.encoder = EfficientHybridEncoder(
            num_layers=encoder_layers,
            num_heads=nheads,
            d_model=d_model,
            ff_inner_dim=ff_inner_dim,
            dropout_prob=dropout_prob,
            num_levels=3,
            num_points=4,
        )
        self.decoder = DeformableDecoder(
            num_layers=decoder_layers,
            num_heads=nheads,
            d_model=d_model,
            ff_inner_dim=ff_inner_dim,
            num_levels=3,
            num_points=4,
            dropout_prob=dropout_prob,
            keep_ratio=keep_ratio,
        )

        # queries
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, d_model))

        # heads
        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, targets=None, score_thresh: float = 0.0, use_nms: bool = False):
        # backbone multi-scale features
        feats_list = self.backbone(x)  # list of 3 maps (B, C, H, W)
        feats_list = [self.input_proj[i](f) for i, f in enumerate(feats_list)]  # project to (B, D, H, W)

        # hybrid encoder
        enc_feats, enc_attn = self.encoder(feats_list)  # list of (B, D, H, W)

        # queries: content zeros + learned positional query embedding
        B = x.shape[0]
        query_objects = torch.zeros(B, self.num_queries, self.d_model, device=x.device)
        query_embed = self.query_embed.unsqueeze(0).repeat(B, 1, 1)
        dec_out_list, dec_cross_attn_list = self.decoder(query_objects, enc_feats, query_embed)

        # heads per layer
        cls_logits_list = [self.class_head(o) for o in dec_out_list]  # list of (B, Q_l, C)
        box_pred_list = [self.box_head(o).sigmoid() for o in dec_out_list]  # list of (B, Q_l, 4)
        iou_pred_list = [self.iou_head(o).sigmoid() for o in dec_out_list]  # list of (B, Q_l, 1)

        outputs = {}
        losses = defaultdict(list)
        detections = []

        if self.training and targets is not None:
            # training with Hungarian matching per decoder layer
            for layer_idx in range(len(dec_out_list)):
                layer_cls = cls_logits_list[layer_idx]  # (B, Q, C)
                layer_box = box_pred_list[layer_idx]  # (B, Q, 4)
                layer_iou = iou_pred_list[layer_idx]  # (B, Q, 1)

                with torch.no_grad():
                    # flatten predictions over batch for vectorized cost construction
                    cls_prob = layer_cls.reshape(-1, self.num_classes).softmax(dim=-1)  # (B*Q, C)
                    pred_boxes = layer_box.reshape(-1, 4)  # (B*Q, 4)

                    tgt_labels = torch.cat([t["labels"] for t in targets])  # (sum_K,)
                    tgt_boxes = torch.cat([t["boxes"] for t in targets])  # (sum_K, 4) xyxy in [0,1]

                    # classification cost
                    cost_cls = -cls_prob[:, tgt_labels]  # (B*Q, sum_K)

                    # convert predicted boxes to xyxy for l1 + giou
                    pred_xyxy = torchvision.ops.box_convert(pred_boxes, "cxcywh", "xyxy")
                    cost_l1 = torch.cdist(pred_xyxy, tgt_boxes, p=1)
                    cost_giou = -torchvision.ops.generalized_box_iou(pred_xyxy, tgt_boxes)

                    Q = layer_cls.shape[1]
                    total_cost = (
                        self.l1_cost_weight * cost_l1
                        + self.cls_cost_weight * cost_cls
                        + self.giou_cost_weight * cost_giou
                    )
                    total_cost = total_cost.reshape(B, Q, -1).cpu()  # (B, Q, sum_K)

                    num_tgt_per_img = [len(t["labels"]) for t in targets]
                    cost_split = total_cost.split(num_tgt_per_img, dim=-1)

                    match_indices = []
                    for b in range(B):
                        if num_tgt_per_img[b] == 0:
                            match_indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                            continue
                        row_ind, col_ind = linear_sum_assignment(cost_split[b][b])  # (Q, Kb) -> index arrays
                        match_indices.append(
                            (torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long))
                        )

                # build quality-aware targets
                layer_cls_out = layer_cls  # (B, Q, C)
                Bsz, Q, C = layer_cls_out.shape
                target_scores = torch.zeros((Bsz, Q, C), device=layer_cls_out.device, dtype=layer_cls_out.dtype)

                pred_batch_idx = (
                    torch.cat([torch.full_like(mi[0], i) for i, mi in enumerate(match_indices)])
                    if len(match_indices)
                    else torch.empty(0, dtype=torch.long)
                )
                pred_query_idx = (
                    torch.cat([mi[0] for mi in match_indices])
                    if len(match_indices)
                    else torch.empty(0, dtype=torch.long)
                )
                tgt_labels_all = (
                    torch.cat([t["labels"][mi[1]] for t, mi in zip(targets, match_indices)])
                    if len(match_indices)
                    else torch.empty(0, dtype=torch.long, device=layer_cls_out.device)
                )

                # matched pred and target boxes for IoU
                matched_pred_boxes = (
                    layer_box[pred_batch_idx, pred_query_idx]
                    if pred_query_idx.numel() > 0
                    else torch.empty(0, 4, device=layer_box.device)
                )
                matched_tgt_boxes = (
                    torch.cat([t["boxes"][mi[1]] for t, mi in zip(targets, match_indices)], dim=0)
                    if pred_query_idx.numel() > 0
                    else torch.empty(0, 4, device=layer_box.device)
                )
                matched_pred_xyxy = (
                    torchvision.ops.box_convert(matched_pred_boxes, "cxcywh", "xyxy")
                    if matched_pred_boxes.numel() > 0
                    else matched_pred_boxes
                )

                def pairwise_iou_diag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    if a.numel() == 0:
                        return torch.empty(0, device=a.device)
                    x1 = torch.maximum(a[:, 0], b[:, 0])
                    y1 = torch.maximum(a[:, 1], b[:, 1])
                    x2 = torch.minimum(a[:, 2], b[:, 2])
                    y2 = torch.minimum(a[:, 3], b[:, 3])
                    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
                    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
                    union = area_a + area_b - inter + 1e-6
                    return inter / union

                iou_targets = pairwise_iou_diag(matched_pred_xyxy, matched_tgt_boxes)
                if pred_query_idx.numel() > 0:
                    target_scores[(pred_batch_idx, pred_query_idx, tgt_labels_all)] = iou_targets

                # quality-aware classification loss
                logits = layer_cls_out.reshape(-1, self.num_classes)
                targets_scores_flat = target_scores.reshape(-1, self.num_classes)
                if self.quality_loss == "vfl":
                    pred_sigmoid = torch.sigmoid(logits)
                    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets_scores_flat, reduction="none")
                    pos_mask = (targets_scores_flat > 0).float()
                    neg_mask = 1.0 - pos_mask
                    weight = pos_mask * targets_scores_flat + neg_mask * (pred_sigmoid**self.vfl_gamma) * self.vfl_alpha
                    cls_loss = (bce * weight).sum() / max(pos_mask.sum().item(), 1.0)
                else:
                    pred_sigmoid = torch.sigmoid(logits)
                    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets_scores_flat, reduction="none")
                    weight = (pred_sigmoid - targets_scores_flat).abs() ** self.qfl_beta
                    pos = (targets_scores_flat > 0).float().sum().item()
                    norm = max(pos, 1.0)
                    cls_loss = (bce * weight).sum() / norm

                # bbox losses for matched pairs
                l1 = (
                    (
                        nn.functional.l1_loss(matched_pred_xyxy, matched_tgt_boxes, reduction="none").sum()
                        / max(matched_pred_boxes.shape[0], 1)
                    )
                    if matched_pred_boxes.numel() > 0
                    else torch.tensor(0.0, device=layer_box.device)
                )
                giou = (
                    torchvision.ops.generalized_box_iou_loss(matched_pred_xyxy, matched_tgt_boxes).sum()
                    / max(matched_pred_boxes.shape[0], 1)
                    if matched_pred_boxes.numel() > 0
                    else torch.tensor(0.0, device=layer_box.device)
                )

                # IoU regression loss
                iou_pred_flat = (
                    layer_iou[pred_batch_idx, pred_query_idx].squeeze(-1)
                    if pred_query_idx.numel() > 0
                    else torch.empty(0, device=layer_cls_out.device)
                )
                iou_reg = (
                    nn.functional.l1_loss(iou_pred_flat, iou_targets.detach(), reduction="mean")
                    if iou_pred_flat.numel() > 0
                    else torch.tensor(0.0, device=layer_cls_out.device)
                )

                losses["classification"].append(cls_loss * self.cls_cost_weight)
                losses["bbox_regression"].append(l1 * self.l1_cost_weight + giou * self.giou_cost_weight)
                losses["iou_quality"].append(iou_reg)

            outputs["loss"] = losses
        else:
            # inference from final layer
            cls_logits = cls_logits_list[-1]  # (B, Q, C)
            box_pred = box_pred_list[-1]  # (B, Q, 4)
            iou_pred = iou_pred_list[-1].squeeze(-1)  # (B, Q)

            prob = torch.sigmoid(cls_logits)
            if self.bg_class_idx == 0:
                fg_prob = prob[..., 1:]
                scores, labels = (fg_prob * iou_pred.unsqueeze(-1)).max(dim=-1)
                labels = labels + 1
            else:
                fg_prob = prob[..., :-1]
                scores, labels = (fg_prob * iou_pred.unsqueeze(-1)).max(dim=-1)

            boxes_xyxy = torchvision.ops.box_convert(box_pred, "cxcywh", "xyxy")
            for b in range(B):
                s = scores[b]
                l = labels[b]
                bx = boxes_xyxy[b]
                keep = s >= score_thresh
                s = s[keep]
                l = l[keep]
                bx = bx[keep]

                if use_nms and bx.numel() > 0:
                    keep_idx = torchvision.ops.batched_nms(bx, s, l, self.nms_threshold)
                    s = s[keep_idx]
                    l = l[keep_idx]
                    bx = bx[keep_idx]

                detections.append({"boxes": bx, "scores": s, "labels": l})

            outputs["detections"] = detections
            outputs["enc_attn"] = enc_attn
            outputs["dec_attn"] = dec_cross_attn_list

        return outputs


def build_rt_detr(num_classes: int, bg_class_idx: int = 0) -> RTDETR:
    """
    Convenience builder for a very lightweight RTDETR suitable for fast experimentation.
    - MobileNetV3 backbone
    - 2 encoder / 2 decoder layers, 4 heads, d_model=192, Q=100
    """
    return RTDETR(
        num_classes=num_classes,
        bg_class_idx=bg_class_idx,
        num_queries=100,
        d_model=192,
        encoder_layers=2,
        decoder_layers=2,
        nheads=4,
        ff_inner_dim=384,
        dropout_prob=0.1,
        backbone="mobilenet_v3_large",
        pretrained_backbone=False,
        freeze_backbone=False,
        cls_cost_weight=1.0,
        l1_cost_weight=5.0,
        giou_cost_weight=2.0,
        bg_class_weight=0.1,
        nms_threshold=0.7,
    )
