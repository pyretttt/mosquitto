import torch
from dataclasses import dataclass
import math

@dataclass
class Config:
    is_custom_targets_assignment_rpn_enabled: bool = True
    is_custom_targets_assignment_roi_enabled: bool = True
    use_custom_achor_generator: bool = True
    use_custom_boxes_to_transformations: bool = True
    use_custom_apply_transformations_to_anchors: bool = True
    use_custom_sample_positive_negative: bool = True
    use_custom_clamp_boxes_to_shape: bool = True
    
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
    below_threshold_label = 0.0,
    between_threshold_label = -1.0
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

    
    best_iou_idx_for_regions[below_low_iou_regions] = -2
    best_iou_idx_for_regions[between_thresholds_regions] = -1
    
    matched_gt_boxes = gt_boxes[best_iou_idx_for_regions.clamp(0)]
    if gt_labels is not None:
        # get label for each region, later will assign negative and ignored mask based on IOU
        labels = gt_labels[best_iou_idx_for_regions.clamp(0)]
    else:
        labels = torch.ones_like(best_iou_idx_for_regions)
		
    labels[below_low_iou_regions] = below_threshold_label
    labels[between_thresholds_regions] = between_threshold_label
    
    return labels.to(dtype=torch.int64), matched_gt_boxes



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


def boxes_to_transformations(
	boxes: torch.Tensor, 
	anchors: torch.Tensor
) -> torch.Tensor:
    r"""Applies transformation computed by regression head to generated anchors
    
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


def apply_transformations_to_anchors(
    anchors: torch.Tensor, 
    transformations: torch.Tensor
) -> torch.Tensor:
    r"""Applies transformation computed by regression head to generated anchors.
    Read README.md for calculation explanation
    
    Args:
        anchors (torch.Tensor): H_feat * W_feat * Num_achors x 4
        transformations (torch.Tensor): N * H_feat * W_feat * Num_achors x num_clases x 4

    Returns:
        torch.Tensor: N * H_feat * W_feat * Num_achors x Num_classes x 4
    """
	# force class shape for compatibility between roi head and rpn
    transformations = transformations.reshape(transformations.size(0), -1, 4)
    assert transformations.shape == (transformations.size(0), transformations.size(1), 4)
    t_x = transformations[..., 0]
    t_y = transformations[..., 1]
    t_w = transformations[..., 2]
    t_h = transformations[..., 3]
    # t_h -> (N x H_feat * W_feat * Num_achors x Num_classes x 1)
    assert(t_x.shape == (transformations.size(0), transformations.size(1)))
    
    # Prevents super big exponentials
    t_w = torch.clamp(t_w, max=math.log(1000.0 / 16, math.e))
    t_h = torch.clamp(t_h, max=math.log(1000.0 / 16, math.e))
    
    anchor_widths = anchors[..., 2] - anchors[..., 0]
    anchor_heights = anchors[..., 3] - anchors[..., 1]
    anchor_center_x = anchors[..., 0] + anchor_widths * 0.5
    anchor_center_y = anchors[..., 1] + anchor_heights * 0.5
    assert (
        anchor_widths.shape == (anchors.size(0), ) 
        and anchor_heights.shape == (anchors.size(0), )
        and anchor_center_x.shape == (anchors.size(0), )
        and anchor_center_y.shape == (anchors.size(0), )
    )
    pred_center_x = t_x * anchor_widths[:, None] + anchor_center_x[:, None]
    pred_center_y = t_y * anchor_heights[:, None] + anchor_center_y[:, None]
    assert (
        pred_center_x.shape == (anchors.size(0), transformations.size(1),) 
        and pred_center_y.shape == (anchors.size(0), transformations.size(1),)
    )
    
    w = torch.exp(t_w) * anchor_widths[:, None]
    h = torch.exp(t_h) * anchor_heights[:, None]
    ## xy wh -> (N * H_feat * W_feat * Num_achors x num_classes)
    x_0 = pred_center_x - w * 0.5
    x_1 = pred_center_x + w * 0.5
    y_0 = pred_center_y - h * 0.5
    y_1 = pred_center_y + h * 0.5
    
    assert (
        x_0.shape == (transformations.size(0), transformations.size(1), )
        and y_0.shape == (transformations.size(0), transformations.size(1), )
        and x_1.shape == (transformations.size(0), transformations.size(1), )
        and y_1.shape == (transformations.size(0), transformations.size(1), )
    )
    
    return torch.stack((x_0, y_0, x_1, y_1), dim=2) # (transformations.size(0) x x num_classes x 4)

    
def custom_sample_positive_negative(
    labels: torch.Tensor, 
    pos_count: int,
    total_count: int
) -> torch.Tensor:
    positive = torch.where(labels >= 1)[0] # return indices
    negative = torch.where(labels == 0)[0] # return indices
    num_positive = min(pos_count, positive.numel())
    num_negative = min(total_count - num_positive, negative.numel())
    
    perm_positive_idx = torch.randperm(positive.numel(), device=positive.device)[:num_positive]
    perm_negative_idx = torch.randperm(negative.numel(), device=negative.device)[:num_negative]
    pos_idx, neg_idx = positive[perm_positive_idx], negative[perm_negative_idx]
    sampled_pos_idx_mask, sampled_neg_idx_mask = torch.zeros((2, *labels.shape), dtype=torch.bool)
    
    sampled_pos_idx_mask[pos_idx] = True
    sampled_neg_idx_mask[neg_idx] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask

    
def clamp_boxes_to_shape(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Clamps boxes to shape of image

    Args:
        boxes (torch.Tensor): (N x 4)
        shape (tuple[int, int]): width x height

    Returns:
        torch.Tensor: clamped boxes (N x 4)
    """
    height, width = shape[-2:]
    x_1, y_1, x_2, y_2 = (
        boxes[..., 0],
        boxes[..., 1],
        boxes[..., 2],
        boxes[..., 3],
    )
    x_1 = x_1.clamp(min=0, max=width)
    y_1 = y_1.clamp(min=0, max=height)
    x_2 = x_2.clamp(min=0, max=width)
    y_2 = y_2.clamp(min=0, max=height)
    return torch.cat([        
	    x_1[..., None],
        y_1[..., None],
        x_2[..., None],
        y_2[..., None],
    ], dim=-1)