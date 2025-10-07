import torch

def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    :param boxes1: (Tensor of shape N x 4)
    :param boxes2: (Tensor of shape M x 4)
    :return: IOU matrix of shape N x M
    """
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
    
    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)  # (N, M)
    union = area1[:, None] + area2 - intersection_area  # (N, M)
    iou = intersection_area / union  # (N, M)
    return iou

import torch
from typing import Optional

def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    :param boxes1: (Tensor of shape N x 4)
    :param boxes2: (Tensor of shape M x 4)
    :return: IOU matrix of shape N x M
    """
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
    
    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)  # (N, M)
    union = area1[:, None] + area2 - intersection_area  # (N, M)
    iou = intersection_area / union  # (N, M)
    return iou


def assign_targets_to_anchors(anchors, gt_boxes):
    r"""
    For each anchor assign a ground truth box based on the IOU.
    Also creates classification labels to be used for training
    label=1 for anchors where maximum IOU with a gtbox > high_iou_threshold
    label=0 for anchors where maximum IOU with a gtbox < low_iou_threshold
    label=-1 for anchors where maximum IOU with a gtbox between (low_iou_threshold, high_iou_threshold)
    :param anchors: (num_anchors_in_image, 4) all anchor boxes
    :param gt_boxes: (num_gt_boxes_in_image, 4) all ground truth boxes
    :return:
        label: (num_anchors_in_image) {-1/0/1}
        matched_gt_boxes: (num_anchors_in_image, 4) coordinates of assigned gt_box to each anchor
            Even background/to_be_ignored anchors will be assigned some ground truth box.
            It's fine, we will use label to differentiate those instances later
    """
    
    # Get (gt_boxes, num_anchors_in_image) IOU matrix
    iou_matrix = get_iou(gt_boxes, anchors)
    
    # For each anchor get the gt box index with maximum overlap
    best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
    # best_match_gt_idx -> (num_anchors_in_image)
    
    # This copy of best_match_gt_idx will be needed later to
    # add low quality matches
    best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()
    
    # Based on threshold, update the values of best_match_gt_idx
    # For anchors with highest IOU < low_threshold update to be -1
    # For anchors with highest IOU between low_threshold & high threshold update to be -2
    below_low_threshold = best_match_iou < 0.3
    between_thresholds = (best_match_iou >= 0.3) & (best_match_iou < 0.7)
    best_match_gt_idx[below_low_threshold] = -1
    best_match_gt_idx[between_thresholds] = -2
    
    # Add low quality anchor boxes, if for a given ground truth box, these are the ones
    # that have highest IOU with that gt box
    
    # For each gt box, get the maximum IOU value amongst all anchors
    best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
    # best_anchor_iou_for_gt -> (num_gt_boxes_in_image)
    
    # For each gt box get those anchors
    # which have this same IOU as present in best_anchor_iou_for_gt
    # This is to ensure if 10 anchors all have the same IOU value,
    # which is equal to the highest IOU that this gt box has with any anchor
    # then we get all these 10 anchors
    gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
    # gt_pred_pair_with_highest_iou -> [0, 0, 0, 1, 1, 1], [8896,  8905,  8914, 10472, 10805, 11138]
    # This means that anchors at the first 3 indexes have an IOU with gt box at index 0
    # which is equal to the highest IOU that this gt box has with ANY anchor
    # Similarly anchor at last three indexes(10472, 10805, 11138) have an IOU with gt box at index 1
    # which is equal to the highest IOU that this gt box has with ANY anchor
    # These 6 anchor indexes will also be added as positive anchors
    
    # Get all the anchors indexes to update
    pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
    
    # Update the matched gt index for all these anchors with whatever was the best gt box
    # prior to thresholding
    best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]
    
    # best_match_gt_idx is either a valid index for all anchors or -1(background) or -2(to be ignored)
    # Clamp this so that the best_match_gt_idx is a valid non-negative index
    # At this moment the -1 and -2 labelled anchors will be mapped to the 0th gt box
    matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]
    
    # Set all foreground anchor labels as 1
    labels = best_match_gt_idx >= 0
    labels = labels.to(dtype=torch.float32)
    
    # Set all background anchor labels as 0
    background_anchors = best_match_gt_idx == -1
    labels[background_anchors] = 0.0
    
    # Set all to be ignored anchor labels as -1
    ignored_anchors = best_match_gt_idx == -2
    labels[ignored_anchors] = -1.0
    # Later for classification we will only pick labels which have > 0 label
    
    return labels, matched_gt_boxes

    
def assign_target_to_regions(
    gt_boxes,
    regions,
    gt_labels: Optional[torch.Tensor] = None,
    low_iou: float = 0.3,
    high_iou: float = 0.7,
    should_match_all_gt_boxes: bool = True,
    labels_type = torch.float32
):
    iou_matrix = get_iou(gt_boxes, regions) # N_box x N_regions
    best_iou_for_regions, best_iou_idx_for_regions = iou_matrix.max(dim=0) # N_regions, finds best gt box for each region
    
    # === Have found best gt_box for each anchor ===
    # Based on IOU value we want to label each passed gt_box either as:
    # Background == 0
    # Ignored == -1
    # Positive == 1
    
    below_low_iou_regions = best_iou_for_regions < low_iou
    between_thresholds_regions = (best_iou_for_regions >= low_iou) & (best_iou_for_regions < high_iou)

    if should_match_all_gt_boxes:
	    # We want to find best anchor for each gt_box. I.e. associate every gt_box with any anchor.
	    # Masks above do not respect this rule - so we make additional adjustments
        # Anchor indices that we dont want to mark as background or ignored
        best_anchor_iou, _ = iou_matrix.max(dim=1)
        # We do this like this instead of using indices above (unnamed `_`), to match all anchors that have the same IOU value.
        anchors_to_unignore = torch.where(iou_matrix == best_anchor_iou[:, None])[1]        
        below_low_iou_regions[anchors_to_unignore] = False
        between_thresholds_regions[anchors_to_unignore] = False

    
    best_iou_idx_for_regions[below_low_iou_regions] = -2
    best_iou_idx_for_regions[between_thresholds_regions] = -1
    
    matched_gt_boxes = gt_boxes[best_iou_idx_for_regions.clamp(0)]
    if gt_labels is not None:
        labels = gt_labels[below_low_iou_regions.clamp(0)].to(dtype=labels_type)
    else:
        labels = (best_iou_idx_for_regions >= 0).to(dtype=labels_type)
            
        
    labels[below_low_iou_regions] = 0.0 # Background
    labels[between_thresholds_regions] = -1.0 # Ignored
    
    return labels, matched_gt_boxes