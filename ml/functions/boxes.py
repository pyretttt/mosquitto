import torch
import math

from functions.detections import get_iou_many_to_many

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
    iou_matrix = get_iou_many_to_many(anchors, bbox) # (N_achors, N_box)
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



def generate_anchors(image, feat, scales, aspect_ratios) -> torch.Tensor:
    r"""Computes anchors for each sliding window inside feature map
    For each sliding window inside feature map there're num_scales * num_ratios anchors generated

    Args:
        image (torch.Tensor): (N x C x H x W) 
        feat (torch.Tensor): (N x C_feat x H_feat x W_feat) feature map from backbone
        scales (list[int]): scales of anchors in sq pixels
        aspect_ratios (list[int]): aspect ratios of anchors preserving scale
    """
    H_feat, W_feat = feat.shape[-2:]
    H_image, W_image = image.shape[-2:]
    
    stride_h = torch.tensor(H_image // H_feat, dtype=torch.int64, device=image.device)
    stride_w = torch.tensor(W_image // W_feat, dtype=torch.int64, device=image.device)

    scales = torch.as_tensor(scales, dtype=feat.dtype, device=feat.device) # (num_scales)
    assert scales.shape == (len(scales),)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=feat.dtype, device=feat.device) # (num_ratios)
    assert aspect_ratios.shape == (len(aspect_ratios),)
    
    # Assuming anchors of scale 128 sq pixels
    # For 1:1 it would be (128, 128) -> area=16384
    # For 2:1 it would be (181.02, 90.51) -> area=16384
    # For 1:2 it would be (90.51, 181.02) -> area=16384
    
    # Look explanation inside [README.md](#RPN-Params)
    h_ratios = torch.sqrt(aspect_ratios) # (num_ratios)
    w_ratios = 1 / h_ratios # (num_ratios)
    
    # Compute actual side scales
    h_scales = (h_ratios[:, None] * scales[None, :]).view(-1) # (num_ratios * num_scales)
    w_scales = (w_ratios[:, None] * scales[None, :]).view(-1) # (num_ratios * num_scales)
    assert h_scales.shape == (len(scales) * len(aspect_ratios),)
    assert w_scales.shape == (len(scales) * len(aspect_ratios),)
    
    ## Anchor displacements about it center in image coordinates
    anchor_coords_about_center = torch.stack([-w_scales, -h_scales, w_scales, h_scales], dim=1) / 2 # (num_ratios * num_scales x 4)
    anchor_coords_about_center = anchor_coords_about_center.round() # (num_ratios * num_scales x 4)
    assert anchor_coords_about_center.shape == (len(scales) * len(aspect_ratios), 4)
    
    # Computes anchor centers in image coordinates
    shift_x = torch.arange(0, W_feat, dtype=torch.int32, device=feat.device) * stride_w # (W_feat)
    shift_y = torch.arange(0, H_feat, dtype=torch.int32, device=feat.device) * stride_h # (H_feat)
    assert shift_x.shape == (W_feat, ) and shift_y.shape == (H_feat, )

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij") 
    # both (H_feat, W_feat)
    assert shift_x.shape == (H_feat, W_feat) and shift_y.shape == (H_feat, W_feat)
    shift_y = shift_y.reshape(-1)
    shift_x = shift_x.reshape(-1)
    assert shift_x.shape == (H_feat * W_feat, ) and shift_y.shape == (H_feat * W_feat, )
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1) # (H_feat * W_feat x 4)
    assert shifts.shape == (H_feat * W_feat, 4)
    anchor_coordinates = shifts[:, None, :] + anchor_coords_about_center[None, :, :]
    assert anchor_coordinates.shape == (H_feat * W_feat, len(scales) * len(aspect_ratios), 4)
    anchor_coordinates = anchor_coordinates.reshape(-1, 4)
    assert anchor_coordinates.shape == (H_feat * W_feat * len(scales) * len(aspect_ratios), 4)
    return anchor_coordinates