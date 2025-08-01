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
    x_left = torch.max(bbox1[..., None, 0], bbox2[..., 0])
    # Let's find leftmost trailling edge between each rectange in first and second array
    x_right = torch.min(bbox1[..., None, 2], bbox2[..., 2])
    # Let's find lower top edge between each rectange in first and second array
    y_top = torch.max(bbox1[..., None, 1], bbox2[..., 1]) 
    # Let's find upper bottom edge between each rectange in first and second array
    y_bottom = torch.min(bbox1[..., None, 4], bbox2[..., 4])

    # If x_right < x_left then there's no intersection.
    # The same if y_bottom < y_top there's no intesection.
    
    intersection = (x_right - x_left).clamp(min=0.0) * (y_bottom - y_top).clamp(min=0.0) # (N, M)
    union = area1[:, None] + area2 - intersection # (N, M)
    iou = intersection / union
    return iou


