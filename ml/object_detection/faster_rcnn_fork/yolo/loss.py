import torch    

def yolo_v1_loss(targets, predictions, num_anchors: int = 1):
    """
    targets: torch.Tensor (N, grid_size, grid_size, depth)
    predictions: torch.Tensor (N, grid_size, grid_size, depth)
    """
    
    # Compute loss only for cells where object is located
    mask = targets[..., 0].as_type(torch.bool) # N, grid_size, grid_size
    targets_obj, predictions_obj = targets[mask], predictions[mask]
    targets_noobj, predictions_noobj = targets[~mask], predictions[~mask]
    
    object_loss = torch.sum((targets_obj[..., 0] - predictions_obj[..., 0]) ** 2)
    noobj_loss = torch.sum((targets_noobj[..., 0] - predictions_noobj[..., 0]) ** 2)
    
    class_loss = torch.sum((targets_obj[..., 5:] - predictions_obj[..., 5:]) ** 2)
    
    localization_loss = torch.sum((targets_obj[..., 1:3] - predictions_obj[..., 1:3]) ** 2)
    
    size_loss = torch.sum((torch.sqrt(targets_obj[..., 3:5])) - torch.sqrt(predictions_obj[..., 3:5]) ** 2)

    return (
        object_loss
        + 0.5 * noobj_loss
        + class_loss,
        + 5 * localization_loss
        + 5 * size_loss
    )