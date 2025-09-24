import torch


def get_iou(bbox1, bbox2):
    """Computes bounding intersection over union between two boxes

    Args:
        bbox1 (torch.tensor): bounding box (4, )
        bbox2 (torch.tensor): bounding box (4, )
    """
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    x_left = torch.max(bbox1[0], bbox2[0])
    x_right = torch.min(bbox1[2], bbox2[2])
    y_top = torch.max(bbox1[1], bbox2[1])
    y_bottom = torch.min(bbox1[3], bbox2[3])
    
    intersection = (x_right - x_left).clamp(0) * (y_bottom - y_top).clamp(0)
    
    return intersection / (area1 + area2 - intersection)
    

def get_iou_many_to_many(bbox1, bbox2) -> float:
    """
    Computes iou of each rectangle in bbox1 with each rectange in bbox2
    :param bbox1: torch.Tensor Nx4    
    :param bbox2: torch.Tensor Mx4
    return: IOU torch.Tensor NxM
    """
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) # (N, )
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) # (M, )

    # Let's find rightmost leading edge between each rectange in first and second array
    x_left = torch.max(bbox1[..., None, 0], bbox2[..., 0]) # (N x M)
    # Let's find leftmost trailling edge between each rectange in first and second array
    x_right = torch.min(bbox1[..., None, 2], bbox2[..., 2]) # (N x M)
    # Let's find lower top edge between each rectange in first and second array
    y_top = torch.max(bbox1[..., None, 1], bbox2[..., 1]) # (N x M)
    # Let's find upper bottom edge between each rectange in first and second array
    y_bottom = torch.min(bbox1[..., None, 4], bbox2[..., 4]) # (N x M)

    # If x_right < x_left then there's no intersection.
    # The same if y_bottom < y_top there's no intesection.
    
    intersection = (x_right - x_left).clamp(min=0.0) * (y_bottom - y_top).clamp(min=0.0) # (N, M)
    union = area1[:, None] + area2 - intersection # (N, M)
    iou = intersection / union
    return iou


def sample_positive_negative(
    labels: torch.Tensor, 
    pos_count: int,
    total_count: int
) -> torch.Tensor:
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    num_positive = min(pos_count, positive.numel())
    num_negative = min(total_count - num_positive, negative.numel())
    
    perm_positive_idx = torch.randnperm(positive.numel(), device=positive.device)[:num_positive]
    perm_negative_idx = torch.randnperm(negative.numel(), device=num_negative.device)[:num_negative]
    pos_idx, neg_idx = positive[perm_positive_idx], negative[perm_negative_idx]
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = sampled_pos_idx_mask.copy()
    
    sampled_pos_idx_mask[pos_idx] = True
    sampled_neg_idx_mask[neg_idx] = True
    return sampled_pos_idx_mask, sampled_neg_idx_mask


def clamp_boxes_to_shape(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Clamps boxes to shape of image

    Args:
        boxes (torch.Tensor): (N x 4)
        shape (tuple[int, int]): width x height

    Returns:
        torch.Tensor: clamped boxes (N x 4)
    """
    width, height = shape[-2:]
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


def scale_boxes_by_aspect_ratio(
    boxes: torch.Tensor, 
    after_scale_size: tuple[int, int], 
    original_size: tuple[int, int]
) -> torch.Tensor:
    """Applies aspect ratio scale to boxes

    Args:
        boxes (torch.Tensor): (N x 4)
        after_scale_size (tuple[int, int]): Size after scale
        original_size (tuple[int, int]): Size before scale

    Returns:
        torch.Tensor: Boxes scaled according to original size
    """
    width_scale = original_size[0] / after_scale_size[0]
    height_scale = original_size[1] / after_scale_size[1]
    
    x_1 = boxes[..., 0] * width_scale
    y_1 = boxes[..., 1] * height_scale
    x_2 = boxes[..., 2] * width_scale
    y_2 = boxes[..., 3] * height_scale
    
    return torch.cat([
        x_1[..., None],
        y_1[..., None],
        x_2[..., None],
        y_2[..., None]
    ], dim=-1)