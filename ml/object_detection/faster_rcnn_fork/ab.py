import torch
from dataclasses import dataclass

@dataclass
class Config:
    is_custom_targets_assignment_rpn_enabled: bool = True
    is_custom_targets_assignment_roi_enabled: bool = False
    
config = Config()

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
    # Negative == 0
    # Ignored == -1
    # Positive == 1
    
    below_low_iou_regions = best_iou_for_regions < low_iou
    between_thresholds_regions = (best_iou_for_regions >= low_iou) & (best_iou_for_regions < high_iou)

    if should_match_all_gt_boxes:
	    # We want to find best anchor for each gt_box. I.e. associate every gt_box with any anchor.
	    # Masks above do not respect this rule - so we make additional adjustments
        # Anchor indices that we dont want to mark as negative or ignored
        best_anchor_iou, _ = iou_matrix.max(dim=1)
        # We do this like this instead of using indices above (unnamed `_`), to match all anchors that have the same IOU value.
        anchors_to_unignore = torch.where(iou_matrix == best_anchor_iou[:, None])[1]        
        below_low_iou_regions[anchors_to_unignore] = False
        between_thresholds_regions[anchors_to_unignore] = False

    
    # best_iou_idx_for_regions[below_low_iou_regions] = -2
    # best_iou_idx_for_regions[between_thresholds_regions] = -1
    
    matched_gt_boxes = gt_boxes[best_iou_idx_for_regions.clamp(0)]
    if gt_labels is not None:
        labels = gt_labels.to(dtype=labels_type)
    else:
        # labels = (best_iou_idx_for_regions >= 0).to(dtype=labels_type)
        labels = torch.ones_like(best_iou_idx_for_regions).to(dtype=labels_type)
            
        
    labels[below_low_iou_regions] = 0.0 # Negative
    labels[between_thresholds_regions] = -1.0 # Ignored
    
    return labels, matched_gt_boxes