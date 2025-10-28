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
import torchvision
import math

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
            fea=feat_map,
            scales=self.scales,
            aspect_ratios=self.aspect_ratios
        ) # N x H x W x Num_anchors
        proposals = apply_transformations_to_anchors(
            anchors=anchors, 
            transformations=regression_pred.detach().reshape(-1, 1, 4) # add class dimension for compatibility
        ).reshape(-1, 4) # N * H * W x Num_anchors
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
        
        if not self.train:
            return proposals
        