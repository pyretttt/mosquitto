from typing import List
from ab import (
    config, 
    generate_anchors, 
    get_iou,
    assign_target_to_regions,
    boxes_to_transformations,
    apply_transformations_to_anchors,
    custom_sample_positive_negative,
    clamp_boxes_to_shape,
    scale_boxes_by_aspect_ratio,
    modify_proposals,
    filter_roi_predictions,
    Backbone
)

import torch
import torch.nn as nn

class RPN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        scales: List[int],
        aspect_ratios: List[float],
        model_config: dict
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.model_config = model_config
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        self.rpn_conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.reg_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1
        )
        self.classification_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors,
            kernel_size=1
        )
        
        
    def forward(
        self,
        image: torch.Tensor,
        feat_map: torch.Tensor,
        target: dict
    ):
        """
        1. Forward classification and regression head
        2. Generate anchors
        3. Apply regression to anchors, to get proposals
        4. Assign gt box targets to each anchor
        5. Turn matched gt box to regression targets
        6. Sample positive and negatives
        7. Apply classification for all labels >= 0, and localization loss for positive labels only
        """
        conv_out = self.rpn_conv(feat_map)
        regression_pred = (
            self.reg_conv(conv_out)
                .permute(0, 2, 3 ,1)
                .view(-1, 4)
        ) # N * H * W * num_achors x 4
        classification_scores = torch.sigmoid(
            self.classification_head(conv_out)
        ).permute(
            0, 2, 3, 1 # ~> N x H x W x num_achors
        ).view(
            -1, self.num_anchors # ~> N * H * W x num_achors
        )
        
        anchors = generate_anchors(
            image=image,
            feat=feat_map,
            scales=self.scales,
            aspect_ratios=self.aspect_ratios
        ) # N * H * W * Num_anchors x 4
        proposals = apply_transformations_to_anchors(
            anchors=anchors, 
            transformations=regression_pred.detach().reshape(-1, 1, 4) # add class dimension for compatibility
        ).reshape(-1, 4) # N * H * W x Num_anchors
        proposals = clamp_boxes_to_shape(
            proposals, 
            image.shape[-2:]
        )
        proposals = modify_proposals(
            proposals=proposals,
            cls_scores=classification_scores.detach(),
            image_shape=image.shape[-2:],
            rpn_prenms_topk=(
                self.model_config['rpn_train_prenms_topk'] if self.training else self.model_config['rpn_test_prenms_topk']
            ),
            box_min_size=16,
            nms_iou_threshold=(
                self.model_config['rpn_train_topk'] if self.training else self.model_config['rpn_test_topk']
            )
        )

        if not self.training:
            return proposals

        labels, matched_gt_boxes = assign_target_to_regions(
            gt_boxes=target["bboxes"][0],
            regions=anchors
        ) # N
        target_transformations = boxes_to_transformations(
            boxes=matched_gt_boxes, 
            anchors=anchors
        ) # N x 4
        negative_indices, positive_indices = custom_sample_positive_negative(
            labels=labels,
            pos_count=int(self.model_config['rpn_pos_fraction'] * self.model_config['rpn_batch_size']),
            total_count=self.model_config['rpn_batch_size']
        )
        
        sampled_indices = torch.where(negative_indices | positive_indices)[0]
        
        classification_loss = torch.nn.functional.cross_entropy(
            input=classification_scores[sampled_indices],
            target=labels[sampled_indices].flatten()
        )
        localization_loss = torch.nn.functional.smooth_l1_loss(
            input=regression_pred[positive_indices],
            target=target_transformations[positive_indices],
            reduction="sum",
            beta=1.0/9.0
        )
        return dict(
            classification_loss=classification_loss,
            localization_loss=localization_loss
        )