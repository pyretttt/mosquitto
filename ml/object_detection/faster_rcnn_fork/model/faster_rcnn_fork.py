from typing import List
from ab import (
    generate_anchors, 
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

import torchvision
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
        self.classification_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors,
            kernel_size=1
        )
        for layer in [self.rpn_conv, self.reg_conv, self.classification_conv]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
        
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
        conv_out = nn.ReLU()(self.rpn_conv(feat_map))
        regression_pred = (
            self.reg_conv(conv_out)
                .permute(0, 2, 3 ,1)
                .reshape(-1, 4)
        ) # N * H * W * num_achors x 4
        classification_scores = (
            self.classification_conv(conv_out)
        ).permute(
            0, 2, 3, 1 # ~> N x H x W x num_achors
        ).reshape(
            -1, 1 # ~> N * H * W x num_achors
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
        ).reshape(-1, 4) # N * H * W * Num_anchors x 4
        proposals = clamp_boxes_to_shape(
            proposals, 
            image.shape[-2:]
        )
        proposals, scores = modify_proposals(
            proposals=proposals,
            cls_scores=classification_scores.detach(),
            image_shape=image.shape[-2:],
            rpn_prenms_topk=(
                self.model_config['rpn_train_prenms_topk'] if self.training else self.model_config['rpn_test_prenms_topk']
            ),
            box_min_size=16,
            nms_iou_threshold=self.model_config['rpn_nms_threshold'],
            post_nms_topk=self.model_config['rpn_train_topk'] if self.training else self.model_config['rpn_test_topk']
        )

        rpn_output = dict(
            proposals=proposals,
            scores=scores
        )
        if not self.training:
            return rpn_output

        labels, matched_gt_boxes = assign_target_to_regions(
            gt_boxes=target["bboxes"][0],
            regions=anchors
        ) # N, N x 4
        assert (
            len(labels.shape) == 1
            and len(matched_gt_boxes.shape) == 2 
            and matched_gt_boxes.size(dim=1) == 4
        )
        labels = labels.to(dtype=torch.float32)
        
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
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=classification_scores[sampled_indices].flatten(),
            target=labels[sampled_indices]
        )
        localization_loss = torch.nn.functional.smooth_l1_loss(
            input=regression_pred[positive_indices],
            target=target_transformations[positive_indices],
            reduction="sum",
            beta=1.0/9.0
        ) / (sampled_indices.numel())
        rpn_output["rpn_classification_loss"] = classification_loss
        rpn_output["rpn_localization_loss"] = localization_loss
        return rpn_output



class ROIHead(nn.Module):
    def __init__(self, model_config, num_classes, in_channels):
        self.model_config = model_config
        self.num_classes = num_classes
        self.fc1 = nn.Linear(
            in_channels=in_channels * self.model_config['roi_pool_size'] ** 2,
            out_features=self.model_config['fc_inner_dim'],
        )
        self.fc2 = nn.Linear(
            in_channels=self.model_config['fc_inner_dim'],
            out_features=self.model_config['fc_inner_dim'],
        )
        # Probabilities for all classes
        self.classifier_head = nn.Linear(
            in_channels=self.model_config['fc_inner_dim'],
            out_features=num_classes,
        )
        # Regression for all classes
        self.regression_head = nn.Linear(
            in_channels=self.model_config['fc_inner_dim'],
            out_features=4 * num_classes
        )
    
        torch.nn.init.normal_(self.classifier_head.weight, std=0.01)
        torch.nn.init.constant_(self.classifier_head.bias, 0)

        torch.nn.init.normal_(self.regression_head.weight, std=0.001)
        torch.nn.init.constant_(self.regression_head.bias, 0)
    
    
    def forward(self, feat, proposals, image_shape, target):
        """
        1. Compute roi features w.r.t. to obtained proposals.
        2. Based roi features compute regression targets and classification scores.
        3. If training compute gt regression targets and gt labels based on proposals.
        """
        
        h_scale, w_scale = (
            image_shape[0] / feat.shape[-2],
            image_shape[1] / feat.shape[-1]
        )
        assert h_scale == w_scale
        proposal_roi_pool_feat = torchvision.ops.roi_pool(
            input=feat,
            boxes=proposals,
            output_size=self.model_config['roi_pool_size'],
            spatial_scale=w_scale
        ) # Num_proposals x C x roi_pool_size x roi_pool_size
        proposal_roi_pool_feat = proposal_roi_pool_feat.flatten(start_dim=1)
        proposal_roi_pool_feat = torch.nn.functional.relu(self.fc1(proposal_roi_pool_feat))
        proposal_roi_pool_feat = torch.nn.functional.relu(self.fc2(proposal_roi_pool_feat))
        
        classifier_scores = self.classifier_head(proposal_roi_pool_feat) # Num_boxes x num_classes
        regression_pred = self.regression_head(proposal_roi_pool_feat) # Num_boxes x num_classes * 4
        assert (
            classifier_scores.size(-1) == self.num_classes 
            and regression_pred.size(-1) == self.num_classes * 4
        )
        regression_pred = regression_pred.reshape(-1, self.num_classes, 4)
        
        frcnn_output = dict()
        if self.training:
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]
            # Add ground truth bounding boxes for better learning, they may or may not be sampled later
            proposals = torch.cat([proposals, gt_boxes], dim=0)
            # N x 4
            labels, matched_gt_boxes_for_proposals = assign_target_to_regions(
                gt_boxes=gt_boxes,
                regions=proposals,
                gt_labels=gt_labels,
                should_match_all_gt_boxes=False,
                below_threshold_label=-1,
                between_threshold_label=0
            )
            negative_indices, positive_indices = custom_sample_positive_negative(
                labels,
                pos_count=int(self.model_config['roi_pos_fraction'] * self.model_config["roi_batch_size"])
                total_count=self.model_config["roi_batch_size"]
            )
            sampled_idxs = torch.where(positive_indices | negative_indices)[0]
            proposals = proposals[sampled_idxs]
            gt_boxes = gt_boxes[sampled_idxs]
            labels = labels[sampled_idxs]
            regression_targets = boxes_to_transformations(
                boxes=matched_gt_boxes_for_proposals[sampled_idxs],
                anchors=proposals
            ) # N x 4
            
            # This is wrong sample first