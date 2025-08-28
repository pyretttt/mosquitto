import torch
import math


def apply_transformations_to_anchors(
    anchors: torch.Tensor, 
    transformations: torch.Tensor
) -> torch.Tensor:
    r"""Applies transformation computed by regression head to generated anchors.
    Read README.md for calculation explanation
    
    Args:
        anchors (torch.Tensor): H_feat * W_feat * Num_achors x 4
        transformations (torch.Tensor): N * H_feat * W_feat * Num_achors x 4

    Returns:
        torch.Tensor: N * H_feat * W_feat * Num_achors x 4
    """
    batch_size = transformations.size(0) // anchors.size(0)
    transformations = transformations.reshape(batch_size, -1, 4)
    assert transformations.shape == (batch_size, anchors.size(0), 4)
    t_x = transformations[..., 0]
    t_y = transformations[..., 2]
    t_w = transformations[..., 1]
    t_h = transformations[..., 3]
    # t_h -> (N x H_feat * W_feat * Num_achors x 1)
    
    # Prevents super big exponentials
    t_w = torch.clamp(t_w, max=math.log(1000.0, math.e))
    t_h = torch.clamp(t_h, max=math.log(1000.0, math.e))
    
    anchor_widths = anchors[..., 2] - anchors[..., 0]
    anchor_heights = anchors[..., 3] - anchors[..., 1]
    anchor_center_x = anchors[..., 0] + anchor_widths / 2.0
    anchor_center_y = anchors[..., 1] + anchor_heights / 2.0
    assert (
        anchor_widths.shape == (anchors.size(0), ) 
        and anchor_heights.shape == (anchors.size(0), )
        and anchor_center_x == (anchors.size(0), )
        and anchor_center_y == (anchors.size(0), )
    )
    x_0 = t_x * anchor_widths[None, ...] + anchor_center_x[None, ...]
    y_0 = t_y * anchor_heights[None, ...] + anchor_center_y[None, ...]
    
    w = torch.exp(t_w) * anchor_widths[None, ...]
    h = torch.exp(t_h) * anchor_heights[None, ...]
    ## xy wh -> (N x H_feat * W_feat * Num_achors x 1)
    x_1 = x_0 + w
    y_1 = y_0 + h
    
    assert (
        x_0.shape == (batch_size, anchors.size(0), )
        and y_0.shape == (batch_size, anchors.size(0), )
        and x_1.shape == (batch_size, anchors.size(0), )
        and y_1.shape == (batch_size, anchors.size(0), )
    )
    
    return torch.stack((x_0, y_0, x_1, y_1), dim=-1) # (N x anchors.size(0) x 4)


def boxes_to_transformations(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    r"""Applies transformation computed by regression head to generated anchors
    Read README.md for calculation explanation
    
    Args:
        boxes (torch.Tensor): N x 4
        anchors (torch.Tensor): N x 4

    Returns:
        torch.Tensor: N * H_feat * W_feat * Num_achors x 4
    """    
    widths = anchors[..., 2] - anchors[..., 0]
    heights = anchors[..., 3] - anchors[..., 1]
    anchor_center_x = anchors[..., 0] + widths / 2.0
    anchor_center_y = anchors[..., 1] + heights / 2.0
    assert (
        widths.shape == (anchors.size(0), ) 
        and heights.shape == (anchors.size(0), )
        and anchor_center_x.shape == (anchors.size(0), )
        and anchor_center_y.shape == (anchors.size(0), )
    )

    gt_widths = boxes[..., 2] - boxes[..., 0]
    gt_heights = boxes[..., 3] - boxes[..., 1]
    gt_anchor_center_x = boxes[..., 0] + gt_widths / 2.0
    gt_anchor_center_y = boxes[..., 1] + gt_heights / 2.0
    assert (
        gt_widths.shape == (anchors.size(0), ) 
        and gt_heights.shape == (anchors.size(0), )
        and gt_anchor_center_x.shape == (anchors.size(0), )
        and gt_anchor_center_y.shape == (anchors.size(0), )
    )
    
    t_x = (gt_anchor_center_x - anchor_center_x) / widths
    t_y = (gt_anchor_center_y - anchor_center_y) / heights
    t_w = torch.log(gt_widths) - torch.log(widths)
    t_h = torch.log(gt_heights) - torch.log(heights)

    return torch.stack(
        [t_x, t_y, t_w, t_h],
        dim=-1
    )


def assign_targets_to_anchors(
    anchors: torch.Tensor,
    bbox: torch.Tensor,
    low_iou_threshold: float,
    high_iou_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""For each anchor assigns a ground truth box based on IOU

    Args:
        anchors (torch.Tensor): N_achors x 4
        bbox (_type_): N_box x 4

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (N_anchors, N_achors x 4)
    """
    iou_matrix = fn.get_iou_many_to_many(anchors, bbox) # (N_achors, N_box)
    best_match_iou, best_match_iou_idx = iou_matrix.max(dim=0) # (N_anchors)
    best_match_iou_idx_pre_thresholding = best_match_iou_idx.copy()
    
    below_low_threshold = best_match_iou < low_iou_threshold
    between_threshold = (best_match_iou < high_iou_threshold and best_match_iou > low_iou_threshold)
    best_match_iou_idx[below_low_threshold] = -1
    best_match_iou_idx[between_threshold] = -2
    
    best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1) # N_box
    gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
    pred_idxs_to_update = gt_pred_pair_with_highest_iou[1]

    best_match_iou_idx[pred_idxs_to_update] = best_match_iou_idx_pre_thresholding[pred_idxs_to_update]
    
    matched_gt_boxes = bbox[best_match_iou_idx.clamp(min=0)]
    labels = (best_match_iou_idx >= 0).to(dtype=torch.float32)
    background_anchors = best_match_iou_idx == -1
    labels[background_anchors] = 0.0
    ignored_anchors = best_match_iou_idx == -2
    labels[ignored_anchors] = -1
    
    return labels, matched_gt_boxes
