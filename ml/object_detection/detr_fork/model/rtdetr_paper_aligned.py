import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import mobilenet_v3_large
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def get_spatial_position_embedding(pos_emb_dim: int, feat_map: torch.Tensor) -> torch.Tensor:
    """2D sine-cosine positional embedding for a feature map."""
    assert pos_emb_dim % 4 == 0, "Position embedding dimension must be divisible by 4"
    H, W = feat_map.shape[2], feat_map.shape[3]
    device = feat_map.device

    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")
    gh = gh.reshape(-1)
    gw = gw.reshape(-1)

    denom = 10000 ** (
        torch.arange(start=0, end=pos_emb_dim // 4, dtype=torch.float32, device=device) / (pos_emb_dim // 4)
    )

    emb_h = (gh[:, None] / denom).to(torch.float32)
    emb_w = (gw[:, None] / denom).to(torch.float32)

    emb_h = torch.cat([torch.sin(emb_h), torch.cos(emb_h)], dim=-1)
    emb_w = torch.cat([torch.sin(emb_w), torch.cos(emb_w)], dim=-1)
    pos = torch.cat([emb_h, emb_w], dim=-1)
    return pos


class MultiScaleBackbone(nn.Module):
    """Wrap a backbone to produce last 3 pyramid features."""

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
            while len(feats) < self.num_scales:
                feats.insert(0, feats[0])
        # Return features in ascending resolution order [C3, C4, C5]
        return feats[-self.num_scales :]


class RepConvBlock(nn.Module):
    """RepVGG-style convolutional block used in CCFM."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CCFM(nn.Module):
    """
    CNN-based Cross-scale Feature Fusion Module (CCFM/CCFF).
    Implements top-down and bottom-up pathways for efficient multi-scale fusion.
    Paper section: Efficient Hybrid Encoder - CCFM component
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        # Lateral convolutions to project all scales to same dimension
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, 1, bias=False) for in_ch in in_channels_list]
        )

        # Top-down pathway: fusion blocks after upsampling
        self.fpn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    RepConvBlock(out_channels, out_channels, 3, 1), RepConvBlock(out_channels, out_channels, 3, 1)
                )
                for _ in range(len(in_channels_list) - 1)
            ]
        )

        # Bottom-up pathway: downsampling convolutions
        self.downsample_convs = nn.ModuleList(
            [
                nn.Sequential(
                    RepConvBlock(out_channels, out_channels, 3, 2), RepConvBlock(out_channels, out_channels, 3, 1)
                )
                for _ in range(len(in_channels_list) - 1)
            ]
        )

    def forward(self, features):
        """
        Args:
            features: List of [C3, C4, C5] from backbone
        Returns:
            List of fused multi-scale features [P3, P4, P5]
        """
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # Top-down pathway: P5 -> P4 -> P3. P5 upsamped to P4 and P4 is upsampled to P3 fusing info from all scales
        for i in range(len(laterals) - 1, 0, -1):  # [2, 1]
            laterals[i - 1] = (
                laterals[i - 1]
                + F.interpolate(  # P4 + P5, P3 + P4
                    laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
                )
            )
            laterals[i - 1] = self.fpn_blocks[i - 1](laterals[i - 1])

        # Bottom-up pathway: P3 -> P4 -> P5
        outputs = [laterals[0]]  # P3 with info from P4 and P5
        for i in range(len(laterals) - 1):  # 0, 1
            outputs.append(self.downsample_convs[i](outputs[-1]) + laterals[i + 1])  # P3 + P4, P4 + P5

        # It adds info in both ways P3 -> P4 -> P5, and P5 -> P4 -> P3. It also downsamples feature by factor of 2
        return outputs


class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction (AIFI).
    Applies transformer self-attention within a single scale.
    Paper section: Efficient Hybrid Encoder - AIFI component
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, pos_embed=None):
        """
        Args:
            src: Feature tokens (B, HW, C)
            pos_embed: Positional embedding (HW, C)
        """
        # Self-attention with positional encoding
        q = k = src if pos_embed is None else src + pos_embed
        src2, _ = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class HybridEncoder(nn.Module):
    """
    Efficient Hybrid Encoder combining CCFM and AIFI.
    Key innovation: decouples intra-scale interaction (AIFI) and cross-scale fusion (CCFM).
    Paper section: Efficient Hybrid Encoder
    """

    def __init__(
        self,
        in_channels_list,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
        use_encoder_idx: list = [2],  # Apply AIFI only to highest level (S5)
    ):
        super().__init__()
        self.use_encoder_idx = use_encoder_idx
        self.num_levels = len(in_channels_list)

        # CCFM: CNN-based cross-scale feature fusion
        self.ccfm = CCFM(in_channels_list, hidden_dim)

        # AIFI: Attention-based intra-scale feature interaction
        # Applied only to selected scales (typically S5)
        self.encoder_layers = nn.ModuleList([AIFI(hidden_dim, num_heads, dropout) for _ in range(num_encoder_layers)])

        # Level embeddings for multi-scale awareness
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, hidden_dim))
        nn.init.normal_(self.level_embed)

    def forward(self, features):
        """
        Args:
            features: List of multi-scale features [C3, C4, C5] from backbone
        Returns:
            Encoder output features and spatial information
        """
        # Step 1: CCFM for cross-scale fusion
        fused_features = self.ccfm(features)

        # Step 2: Apply AIFI to selected scales (decoupled intra-scale interaction)
        output_features = []
        for idx, feat in enumerate(fused_features):
            if idx in self.use_encoder_idx:
                # Apply AIFI for intra-scale feature interaction
                bs, c, h, w = feat.shape
                feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]

                # Add positional encoding
                pos_embed = get_spatial_position_embedding(c, feat).unsqueeze(0).expand(bs, -1, -1)

                # Apply AIFI layers
                for layer in self.encoder_layers:
                    feat_flat = layer(feat_flat, pos_embed)

                # Reshape back
                feat = feat_flat.permute(0, 2, 1).reshape(bs, c, h, w)

            output_features.append(feat)

        # Flatten for decoder
        src_flatten = []
        spatial_shapes = []
        for feat in output_features:
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            src_flatten.append(feat.flatten(2).transpose(1, 2))  # [B, HW, C]
            assert src_flatten[-1].shape == (bs, h * w, c)

        src_flatten = torch.cat(src_flatten, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # first level starts at zero
                spatial_shapes.prod(1).cumsum(0)[  # Multiply height with width  # Compute cumulative level sum
                    :-1
                ],  # Return all levels but last
            )
        )

        return src_flatten, output_features, spatial_shapes, level_start_index


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention."""

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
        self,
        query: torch.Tensor,
        feats: list[torch.Tensor],
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Q, D = query.shape

        offsets = self.sampling_offsets(query)  # B, Q, num_heads * num_levels * num_points * 2
        offsets = offsets.view(B, Q, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = torch.tanh(offsets)

        attn = self.attention_weights(query)  # B, Q, num_heads * num_levels * num_points
        attn = attn.view(B, Q, self.num_heads, self.num_levels * self.num_points)
        attn = attn.softmax(dim=3)  # B, Q, num_heads, 1
        attn = attn.view(B, Q, self.num_heads, self.num_levels, self.num_points)

        sampled_list = []
        weights_list = []
        for lvl, feat in enumerate(feats):
            B_, D_, H_, W_ = feat.shape
            feat = feat.view(B_, self.num_heads, self.head_dim, H_, W_)
            feat = feat.reshape(B_ * self.num_heads, self.head_dim, H_, W_)

            ref = reference_points
            ref_grid = ref * 2.0 - 1.0  # [-1, 1]
            off = offsets[:, :, :, lvl, :]  # B, Q, num_heads, 1, num_points
            off = off.permute(0, 2, 1, 3, 4)  # B, num_heads, Q, 1, num_points
            grid = ref_grid[:, None, :, None, :].expand(B, self.num_heads, Q, self.num_points, 2) + off
            grid = grid.reshape(B * self.num_heads, Q * self.num_points, 2)
            grid = grid.view(B * self.num_heads, Q * self.num_points, 1, 2)  # B * num_heads, Q * num_ponts, 1, 2

            grid = grid.permute(0, 2, 1, 3)  # B * num_heads, 1, Q * num_ponts, 2
            sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = sampled.squeeze(2)
            sampled = sampled.view(B, self.num_heads, self.head_dim, Q, self.num_points)

            w = attn[:, :, :, lvl, :]
            w = w.permute(0, 2, 3, 1)
            w = w.unsqueeze(2)
            sampled_list.append(sampled)
            weights_list.append(w)

        agg = 0.0
        for lvl in range(self.num_levels):
            sampled = sampled_list[lvl]
            w = weights_list[lvl]
            sampled = sampled.permute(0, 1, 2, 4, 3)
            contrib = (sampled * w).sum(dim=3)
            agg = agg + contrib

        agg = agg.permute(0, 3, 1, 2).reshape(B, Q, self.d_model)
        out = self.out_proj(agg)
        out = self.dropout(out)
        return out, attn


def get_normalized_reference_points(feat: torch.Tensor) -> torch.Tensor:
    """Build normalized [0,1] reference points grid for a feature map."""
    B, D, H, W = feat.shape
    device = feat.device
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ref = torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2).repeat(B, 1, 1)
    return ref


class UncertaintyQuerySelection(nn.Module):
    """
    Uncertainty-minimal Query Selection (IoU-aware).
    Selects high-quality initial queries based on minimal uncertainty.
    Paper section: IoU-aware Query Selection
    """

    def __init__(self, num_queries: int, d_model: int, num_classes: int):
        super().__init__()
        self.num_queries = num_queries

        # Prediction heads for query selection
        self.cls_embed = nn.Linear(d_model, num_classes)
        self.box_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, memory, spatial_shapes, level_start_index):
        """
        Select top-k encoder features based on classification and IoU scores.
        Uncertainty = |cls_score - iou_score|
        """
        bs = memory.shape[0]

        # Predict classes and boxes for all encoder features
        output_class = self.cls_embed(memory)
        output_coord = self.box_embed(memory).sigmoid()

        # Compute classification scores (max across classes)
        cls_scores = output_class.sigmoid().max(-1)[0]  # [B, HW]

        # Compute IoU scores (use predicted box size as proxy)
        iou_scores = output_coord[..., 2] * output_coord[..., 3]  # width * height

        # Uncertainty-minimal selection: minimize |cls_score - iou_score|
        uncertainty = torch.abs(cls_scores - iou_scores)

        # Select top-k features with minimal uncertainty
        topk_indices = torch.topk(1.0 - uncertainty, self.num_queries, dim=1)[1]

        # Gather selected features
        selected_memory = torch.gather(memory, 1, topk_indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))

        return selected_memory, topk_indices


class DeformableDecoder(nn.Module):
    """
    Transformer Decoder with multi-scale deformable attention.
    Paper section: Standard Transformer Decoder
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_inner_dim: int,
        num_levels: int,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = num_levels

        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)]
        )

        self.cross_attn = nn.ModuleList(
            [
                MSDeformAttn(d_model, num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_sa = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ca = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.drop_sa = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.drop_ca = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.drop_ff = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Reference points prediction from query embedding
        self.ref_point_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))

        # Iterative reference point refinement
        self.refine_mlp = nn.Linear(d_model, 2)

    def forward(
        self,
        query_objects: torch.Tensor,
        feats: list[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        out = query_objects
        q_embed = query_embed
        outputs = []
        cross_attn_weights = []

        # Initial reference points from query embedding
        ref_points = self.ref_point_mlp(q_embed).sigmoid()

        for i in range(self.num_layers):
            # Self-attention on queries
            sa_in = self.norm_sa[i](out)
            q = sa_in + q_embed
            k = sa_in + q_embed
            sa_out, _ = self.self_attn[i](q, k, sa_in)
            out = out + self.drop_sa[i](sa_out)

            # Deformable cross-attention
            ca_in = self.norm_ca[i](out)
            ca_out, attn = self.cross_attn[i](ca_in + q_embed, feats, ref_points)
            out = out + self.drop_ca[i](ca_out)

            # FFN
            ff_in = self.norm_ff[i](out)
            out = out + self.drop_ff[i](self.ff[i](ff_in))

            # Iterative reference point refinement
            delta = torch.tanh(self.refine_mlp(out)) * 0.5
            ref_points = (ref_points + delta).clamp(0.0, 1.0)

            outputs.append(out)
            cross_attn_weights.append(attn)

        return outputs, cross_attn_weights


class RTDETR(nn.Module):
    """
    RT-DETR: Real-Time Detection Transformer

    Paper: "DETRs Beat YOLOs on Real-time Object Detection"
    Key components aligned with paper:
    1. Efficient Hybrid Encoder (AIFI + CCFM)
    2. Uncertainty-minimal Query Selection
    3. Standard Transformer Decoder with deformable attention
    4. Quality-aware classification (QFL/VFL)
    """

    def __init__(
        self,
        num_classes: int,
        bg_class_idx: int = 0,
        num_queries: int = 300,
        d_model: int = 256,
        encoder_layers: int = 1,  # Paper uses 1 AIFI layer
        decoder_layers: int = 6,
        nheads: int = 8,
        ff_inner_dim: int = 1024,
        dropout_prob: float = 0.1,
        backbone: str = "mobilenet_v3_large",
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        # Loss weights
        cls_cost_weight: float = 2.0,
        l1_cost_weight: float = 5.0,
        giou_cost_weight: float = 2.0,
        bg_class_weight: float = 0.1,
        nms_threshold: float = 0.7,
        # Quality-aware classification
        quality_loss: str = "qfl",
        qfl_beta: float = 2.0,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        # Apply AIFI only to highest level
        use_encoder_idx: list = [2],
    ):
        super().__init__()

        self.num_classes = num_classes
        self.bg_class_idx = bg_class_idx
        assert self.bg_class_idx in (0, num_classes - 1), "Background can only be 0 or num_classes-1"

        self.num_queries = num_queries
        self.d_model = d_model
        self.nms_threshold = nms_threshold

        # Loss weights
        self.cls_cost_weight = cls_cost_weight
        self.l1_cost_weight = l1_cost_weight
        self.giou_cost_weight = giou_cost_weight
        self.bg_cls_weight = bg_class_weight
        self.quality_loss = quality_loss
        self.qfl_beta = qfl_beta
        self.vfl_alpha = vfl_alpha
        self.vfl_gamma = vfl_gamma

        # Backbone
        if backbone == "mobilenet_v3_large":
            weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            mb = mobilenet_v3_large(weights=weights)
            base = mb.features
        elif backbone == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            rn = torchvision.models.resnet50(weights=weights, norm_layer=torchvision.ops.FrozenBatchNorm2d)
            base = nn.Sequential(*list(rn.children())[:-2])
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

        # Note: Per-level projection is handled inside HybridEncoder.

        # Efficient Hybrid Encoder (AIFI + CCFM)
        # Infer backbone channels dynamically during first forward
        self.encoder = None
        self.encoder_in_channels = None
        self.use_encoder_idx = use_encoder_idx
        self.encoder_layers = encoder_layers
        self.nheads = nheads
        self.ff_inner_dim = ff_inner_dim
        self.dropout_prob = dropout_prob

        # Uncertainty-minimal Query Selection
        self.query_selection = UncertaintyQuerySelection(num_queries, d_model, num_classes)

        # Standard Transformer Decoder
        self.decoder = DeformableDecoder(
            num_layers=decoder_layers,
            num_heads=nheads,
            d_model=d_model,
            ff_inner_dim=ff_inner_dim,
            num_levels=3,
            num_points=4,
            dropout=dropout_prob,
        )

        # Learned query positional embeddings
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, d_model))

        # Prediction heads
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

    def _init_encoder(self, in_channels_list):
        """Initialize encoder based on backbone output channels."""
        self.encoder = HybridEncoder(
            in_channels_list=in_channels_list,
            hidden_dim=self.d_model,
            num_heads=self.nheads,
            num_encoder_layers=self.encoder_layers,
            dropout=self.dropout_prob,
            use_encoder_idx=self.use_encoder_idx,
        )
        self.encoder.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, targets=None, score_thresh: float = 0.0, use_nms: bool = False):
        # Backbone multi-scale features
        feats_list = self.backbone(x)  # S3, S4, S5

        # Initialize encoder on first forward
        if self.encoder is None:
            in_channels = [f.shape[1] for f in feats_list]
            self._init_encoder(in_channels)

        # Pass raw backbone features to the HybridEncoder (it performs projection internally)

        # Efficient Hybrid Encoder (AIFI + CCFM)
        memory, enc_feats, spatial_shapes, level_start_index = self.encoder(feats_list)

        # Uncertainty-minimal Query Selection
        init_queries, query_indices = self.query_selection(memory, spatial_shapes, level_start_index)

        # Prepare query embeddings
        B = x.shape[0]
        query_embed = self.query_embed.unsqueeze(0).repeat(B, 1, 1)

        # Transformer Decoder
        dec_out_list, dec_cross_attn_list = self.decoder(init_queries, enc_feats, query_embed)

        # Prediction heads per layer
        cls_logits_list = [self.class_head(o) for o in dec_out_list]
        box_pred_list = [self.box_head(o).sigmoid() for o in dec_out_list]
        iou_pred_list = [self.iou_head(o).sigmoid() for o in dec_out_list]

        outputs = {}
        losses = defaultdict(list)
        detections = []

        if self.training and targets is not None:
            # Training with Hungarian matching per decoder layer
            for layer_idx in range(len(dec_out_list)):
                layer_cls = cls_logits_list[layer_idx]
                layer_box = box_pred_list[layer_idx]
                layer_iou = iou_pred_list[layer_idx]

                with torch.no_grad():
                    cls_prob = layer_cls.reshape(-1, self.num_classes).softmax(dim=-1)
                    pred_boxes = layer_box.reshape(-1, 4)

                    tgt_labels = torch.cat([t["labels"] for t in targets])
                    tgt_boxes = torch.cat([t["boxes"] for t in targets])

                    cost_cls = -cls_prob[:, tgt_labels]
                    pred_xyxy = torchvision.ops.box_convert(pred_boxes, "cxcywh", "xyxy")
                    cost_l1 = torch.cdist(pred_xyxy, tgt_boxes, p=1)
                    cost_giou = -torchvision.ops.generalized_box_iou(pred_xyxy, tgt_boxes)

                    Q = layer_cls.shape[1]
                    total_cost = (
                        self.l1_cost_weight * cost_l1
                        + self.cls_cost_weight * cost_cls
                        + self.giou_cost_weight * cost_giou
                    )
                    total_cost = total_cost.reshape(B, Q, -1).cpu()

                    num_tgt_per_img = [len(t["labels"]) for t in targets]
                    cost_split = total_cost.split(num_tgt_per_img, dim=-1)

                    match_indices = []
                    for b in range(B):
                        if num_tgt_per_img[b] == 0:
                            match_indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                            continue
                        row_ind, col_ind = linear_sum_assignment(cost_split[b][b])
                        match_indices.append(
                            (torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long))
                        )

                # Build quality-aware targets
                layer_cls_out = layer_cls
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

                # Quality-aware classification loss
                logits = layer_cls_out.reshape(-1, self.num_classes)
                targets_scores_flat = target_scores.reshape(-1, self.num_classes)

                if self.quality_loss == "vfl":
                    pred_sigmoid = torch.sigmoid(logits)
                    bce = F.binary_cross_entropy_with_logits(logits, targets_scores_flat, reduction="none")
                    pos_mask = (targets_scores_flat > 0).float()
                    neg_mask = 1.0 - pos_mask
                    weight = pos_mask * targets_scores_flat + neg_mask * (pred_sigmoid**self.vfl_gamma) * self.vfl_alpha
                    cls_loss = (bce * weight).sum() / max(pos_mask.sum().item(), 1.0)
                else:  # QFL
                    pred_sigmoid = torch.sigmoid(logits)
                    bce = F.binary_cross_entropy_with_logits(logits, targets_scores_flat, reduction="none")
                    weight = (pred_sigmoid - targets_scores_flat).abs() ** self.qfl_beta
                    pos = (targets_scores_flat > 0).float().sum().item()
                    norm = max(pos, 1.0)
                    cls_loss = (bce * weight).sum() / norm

                # Bbox losses
                l1 = (
                    F.l1_loss(matched_pred_xyxy, matched_tgt_boxes, reduction="none").sum()
                    / max(matched_pred_boxes.shape[0], 1)
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
                    F.l1_loss(iou_pred_flat, iou_targets.detach(), reduction="mean")
                    if iou_pred_flat.numel() > 0
                    else torch.tensor(0.0, device=layer_cls_out.device)
                )

                losses["classification"].append(cls_loss * self.cls_cost_weight)
                losses["bbox_regression"].append(l1 * self.l1_cost_weight + giou * self.giou_cost_weight)
                losses["iou_quality"].append(iou_reg)

            outputs["loss"] = losses
        else:
            # Inference from final layer
            cls_logits = cls_logits_list[-1]
            box_pred = box_pred_list[-1]
            iou_pred = iou_pred_list[-1].squeeze(-1)

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

        return outputs


def build_rt_detr(num_classes: int, bg_class_idx: int = 0, backbone: str = "mobilenet_v3_large") -> RTDETR:
    """
    Build RT-DETR model following paper specifications.

    For RT-DETR-R50:
    - num_queries=300
    - d_model=256
    - encoder_layers=1 (AIFI applied only to S5)
    - decoder_layers=6
    - nheads=8

    For lightweight version:
    - Smaller d_model, fewer queries, etc.
    """
    return RTDETR(
        num_classes=num_classes,
        bg_class_idx=bg_class_idx,
        num_queries=300,  # Paper uses 300 queries
        d_model=256,  # Paper uses 256 for R50
        encoder_layers=1,  # Paper uses 1 AIFI layer
        decoder_layers=6,  # Paper uses 6 decoder layers (tunable for speed)
        nheads=8,
        ff_inner_dim=1024,
        dropout_prob=0.1,
        backbone=backbone,
        pretrained_backbone=True,
        freeze_backbone=True,
        cls_cost_weight=2.0,
        l1_cost_weight=5.0,
        giou_cost_weight=2.0,
        bg_class_weight=0.1,
        nms_threshold=0.5,
        quality_loss="qfl",  # Use QFL as in paper
        use_encoder_idx=[2],  # Apply AIFI only to S5
    )
