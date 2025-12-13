import torch
from torch import nn
import numpy as np
import torchvision.models
from torchvision.models import resnet34
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

@dataclass
class AB:
    custom_get_spatial_position_embedding: bool = True
    custom_encoder: bool = True
    custom_decoder: bool = True


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


# class Detr(nn.Module):
#     def __init__(
#         self,
#         config,
#         num_classes: int,
#         bg_class_idx: int
#     ):
#         self.backbone_channels = config['backbone_channels']
#         self.d_model = config['d_model']
#         self.num_queries = config['num_queries']
#         self.num_classes = num_classes
#         self.num_decoder_layers = config['decoder_layers']
#         self.cls_cost_weight = config['cls_cost_weight']
#         self.l1_cost_weight = config['l1_cost_weight']
#         self.giou_cost_weight = config['giou_cost_weight']
#         self.bg_cls_weight = config['bg_class_weight']
#         self.nms_threshold = config['nms_threshold']
#         self.bg_class_idx = bg_class_idx
#         valid_bg_idx = (self.bg_class_idx == 0 or
#                         self.bg_class_idx == (self.num_classes-1))
#         assert valid_bg_idx, "Background can only be 0 or num_classes-1"

#         self.backbone = nn.Sequential(*list(resnet34(
#             weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
#             norm_layer=torchvision.ops.FrozenBatchNorm2d
#         ).children())[:-2])

#         if config["freeze_backbone"]:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         self.backbone_proj = nn.Conv2d(
#             self.backbone_channels,
#             self.d_model,
#             kernel_size=1
#         )
#         self.encoder = TransformerEncoder(
#             num_layers=config['encoder_layers'],
#             num_heads=config['encoder_attn_heads'],
#             d_model=config['d_model'],
#             ff_inner_dim=config['ff_inner_dim'],
#             dropout_prob=config['dropout_prob']
#         )
#         self.decoder = TransformerDecoder(
#             num_layers=config['decoder_layers'],
#             num_heads=config['decoder_attn_heads'],
#             d_model=config['d_model'],
#             ff_inner_dim=config['ff_inner_dim'],
#             dropout_prob=config['dropout_prob']
#         )
#         self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.d_model))
#         self.class_mlp = nn.Linear(self.d_model, self.num_classes)
#         self.bbox_mlp = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.ReLU(),
#             nn.Linear(self.d_model, self.d_model),
#             nn.ReLU(),
#             nn.Linear(self.d_model, 4),
#         )


#     def forward(self, x, targets=None, score_thresh=0, use_nms=False):
#         conv_out = self.backbone(x)
#         conv_out = self.backbone_proj(conv_out) # B, d_model, feat_h, feat_w
#         batch_size, d_model, feat_h, feat_w = conv_out.shape
#         pos_emb = get_spatial_position_embedding(
#             emb_dim=self.d_model, feat_map=conv_out
#         )
#         conv_out = (
#             conv_out.reshape(batch_size, d_model, feat_h * feat_w)
#             .transpose(1, 2)
#         ) # B, feat_h * feat_w, d_model
#         enc_output, enc_attn_weights = self.encoder(conv_out, pos_emb) # (B, feat_h * feat_w, d_model; num_layers, B, feat_h * feat_w, feat_h * feat_w)
#         query_objects = torch.zeros_like(
#             self.query_embed.unsqueeze(0)
#                 .repeat((batch_size, 1, 1))
#         ) # B, num_queries, d_model

#         query_objects, decoder_attn_weights = self.decoder.forward(
#             query_objects=query_objects,
#             encoder_output=enc_output,
#             query_embedding=self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)),
#             pos_emb=pos_emb
#         )
#         cls_output = self.class_mlp(query_objects)
#         bbox_output = self.bbox_mlp(query_objects).sigmoid()

#         losses = defaultdict(list)
#         detections = []
#         detr_output = {}

#         if self.training:
#             for idx in range(self.num_decoder_layers):
#                 cls_idx_out = cls_output[idx]
#                 bbox_idx_out = bbox_output[idx]
#                 with torch.no_grad():
#                     class_prob = (cls_idx_out
#                         .reshape((-1, self.num_classes))
#                         .softmax(dim=-1)
#                     ) # B, num_queries, num_classes -> B * num_queries, num_classes

#                     pred_boxes = bbox_idx_out.reshape((-1, 4)) # B * num_queries, num_classes
#                     target_labels = torch.cat([target["labels"] for target in targets]) # num_targets_for_entire_batch,
#                     target_boxes = torch.cat([target["boxes"] for target in targets]) # num_targets_for_entire_batch, 4

#                     # Takes probabilities of classes present in batch, along all batch and all queries
#                     cost_classification = -class_prob[:, target_labels] # (B*num_queries)

#                     # Turns cxcywhwy to x1y1x2y2
#                     pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
#                         pred_boxes,
#                         "cxcywh",
#                         "xyxy"
#                     )
#                     # Computes distance of each predicted box with each gt box
#                     cost_localization_l1 = torch.cdist(
#                         pred_boxes_x1y1x2y2,
#                         target_boxes,
#                         p=1
#                     ) # (B * num_queries, num_classes) @ (num_targets_for_entire_batch, 4) -> (B*num_queries, num_targets_for_entire_batch)
#                     cost_localization_giou = -torchvision.ops.generalized_box_iou(
#                         pred_boxes_x1y1x2y2,
#                         target_boxes
#                     )
#                     total_cost = (
#                         self.l1_cost_weight * cost_localization_l1
#                         + self.cls_cost_weight * cost_classification
#                         + self.giou_cost_weight * cost_localization_giou
#                     ).reshape(batch_size, self.num_queries, -1)

#                     num_targets_per_image = [len(target["labels"]) for target in targets]
#                     total_cost_per_batch_image = total_cost.split(
#                         num_targets_per_image,
#                         dim=-1
#                     ).cpu()
#                     # total_cost_per_batch_image[0]=(B,num_queries,num_targets_0th_image)
#                     # total_cost_per_batch_image[i]=(B,num_queries,num_targets_ith_image)

#                     match_indices = []
#                     for batch_idx in range(batch_size):
#                         batch_idx_assignments = linear_sum_assignment(
#                             total_cost_per_batch_image[batch_idx][batch_idx]
#                         )
#                         batch_idx_pred, batch_idx_target = batch_idx_assignments
#                         match_indices.append((
#                             torch.as_tensor(batch_idx_pred, dtype=torch.int64),
#                             torch.as_tensor(batch_idx_target, dtype=torch.int64)
#                         ))

#                 pred_batch_idxs = torch.cat([
#                     torch.ones_line(pred_idx) * i
#                     for i, (pred_idx, _) in enumerate(match_indices)
#                 ])
#                 pred_query_idx = torch.cat([pred_idx for pred_idx, _ in match_indices])

#         else: