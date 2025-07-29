import torch

def get_iou(bbox1, bbox2) -> float:
    """
    :param bbox1: torch.Tensor Nx4    
    :param bbox2: torch.Tensor Mx4
    return: IOU torch.Tensor NxM
    """
    
    intersection = 1 # NxM