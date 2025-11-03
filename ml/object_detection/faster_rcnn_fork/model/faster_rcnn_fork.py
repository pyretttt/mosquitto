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
        super().__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']
        
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
        
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)

    
    def forward(self, feat, proposals, image_shape, target):
        """
        1. If training sample some labels and gt boxes
        2. Compute roi features based on proposals
        3. Compute classifier scores and regression targets
        4. If training:
        4.1. Compute target boxes transforms out of gt boxes
        4.2. Compute localization and classification errors
        5. If infering:
        5.1. Compute prediction boxes, based on predicted regression targets. Clamp boxes to image shape.
        5.2. Apply NMS based on scores
        """
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
                low_iou=self.low_bg_iou,
                high_iou=self.iou_threshold,
                below_threshold_label=-1,
                between_threshold_label=0
            )
            negative_indices, positive_indices = custom_sample_positive_negative(
                labels,
                pos_count=self.roi_pos_count,
                total_count=self.roi_batch_size
            )
            sampled_idxs = torch.where(positive_indices | negative_indices)[0]
            proposals = proposals[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            labels = labels[sampled_idxs] # (Num_boxes,)
        

        h_scale, w_scale = (
            2 ** float(torch.tensor(feat.shape[-2] / image_shape[0]).log2().round()),
            2 ** float(torch.tensor(feat.shape[-1] / image_shape[1]).log2().round()),
        )
        assert h_scale == w_scale
        proposal_roi_pool_feat = torchvision.ops.roi_pool(
            input=feat,
            boxes=[proposals],
            output_size=self.pool_size,
            spatial_scale=w_scale
        ) # Num_proposals x C x roi_pool_size x roi_pool_size
        proposal_roi_pool_feat = proposal_roi_pool_feat.flatten(start_dim=1) # Num_proposals x C * roi_pool_size * roi_pool_size
        proposal_roi_pool_feat = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feat))
        proposal_roi_pool_feat = torch.nn.functional.relu(self.fc7(proposal_roi_pool_feat))
        
        classifier_scores = self.cls_layer(proposal_roi_pool_feat) # Num_boxes x num_classes
        regression_pred = self.bbox_reg_layer(proposal_roi_pool_feat) # Num_boxes x num_classes * 4
        assert (
            classifier_scores.size(-1) == self.num_classes 
            and regression_pred.size(-1) == self.num_classes * 4
        )
        regression_pred = regression_pred.reshape(-1, self.num_classes, 4)
        
        if self.training:
            classification_loss = torch.nn.functional.cross_entropy(
                input=classifier_scores,
                target=labels
            )
            # Run localization loss only for foreground objects. Zero class is implicitly background
            fg_proposals_indices = torch.where(labels > 0)[0]
            fg_cls_labels = labels[fg_proposals_indices] # Will use them as indices for regression_pred

            regression_targets = boxes_to_transformations(matched_gt_boxes_for_proposals, proposals)
            localization_loss = torch.nn.functional.smooth_l1_loss(
                input=regression_pred[fg_proposals_indices, fg_cls_labels],
                target=regression_targets[fg_proposals_indices],
                beta=1.0/9.0,
                reduction="sum"
            ) / labels.numel()
            frcnn_output["frcnn_classification_loss"] = classification_loss
            frcnn_output["frcnn_localization_loss"] = localization_loss
        else:
            predicted_boxes = apply_transformations_to_anchors(
                anchors=proposals, 
                transformations=regression_pred
            )
            predicted_boxes = clamp_boxes_to_shape(
                boxes=predicted_boxes, 
                shape=image_shape[-2:]
            )
            classifier_scores = torch.nn.functional.softmax(classifier_scores, dim=-1)
            
            # Creates labels for each prediction
            pred_labels = torch.arange(self.num_classes, device=classifier_scores.device)
            pred_labels = pred_labels.view(1, -1).expand_as(classifier_scores)

            
            # remove predictions with background label
            pred_labels = pred_labels[:, 1:]
            predicted_boxes = predicted_boxes[:, 1:]
            classifier_scores = classifier_scores[:, 1:]
            assert (
                pred_labels.shape == (pred_labels.shape[0], self.num_classes - 1,)
                and predicted_boxes.shape == (pred_labels.shape[0], self.num_classes - 1, 4)
                and classifier_scores.shape == (pred_labels.shape[0], self.num_classes - 1,)
            )

            pred_labels = pred_labels.reshape(-1)
            predicted_boxes = predicted_boxes.reshape(-1, 4)
            classifier_scores = classifier_scores.reshape(-1)
            
            predicted_boxes, pred_labels, classifier_scores = filter_roi_predictions(
                pred_boxes=predicted_boxes,
                pred_labels=pred_labels,
                pred_scores=classifier_scores,
                low_score_threshold=self.low_score_threshold,
                min_size=16,
                nms_threshold=self.nms_threshold,
                topk_detections=self.topK_detections
            )
            frcnn_output["boxes"] = predicted_boxes
            frcnn_output["scores"] = classifier_scores
            frcnn_output["labels"] = pred_labels
        return frcnn_output