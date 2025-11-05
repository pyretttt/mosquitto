
from dataset.voc import VOCDataset

import torch
from torchvision import transforms
from torch.utils.data import Dataset


def make_yolo_targets(
    targets: dict, 
    grid_size: int, 
    num_classes: int
):
    """
    targets: {bboxes: torch.Tensor (N, 4), labels: List[str] (N, )}
    target intentionally contain only one bounding box insted, so only `5 + num_classes`
    """
    out = torch.zeros((grid_size, grid_size, 5 + num_classes))
    
    labels = targets["labels"]
    bboxes = targets["bboxes"]
    for idx, label in enumerate(labels):
        grid_x = bboxes[..., idx, 0] * grid_size
        grid_y = bboxes[..., idx, 1] * grid_size
        
        row, col = int(grid_y), int(grid_x)
        # gt confidence, center of object inside grid cell, gt width and height
        out[row, col, :5] = torch.as_tensor([1.0, grid_x % 1, grid_y % 1, bboxes[..., idx, 2], bboxes[..., idx, 3]])
        out[row, col, 5 + label] = 1.0 # gt class probabilities

    return out


class VocYoloDatasetAdapter(Dataset):
    def __init__(self, split, im_dir, ann_dir, grid_size: int, im_size: int):
        super().__init__()
        self.voc = VOCDataset(split, im_dir, ann_dir)
        self.grid_size = grid_size
        
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(size=(im_size, im_size)),
        ])
        
    @property
    def num_classes(self):
        return len(self.voc.label2idx)

        
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, index):
        im_tensor, targets, _ = self.voc[index]
        im_info = self.voc.images_info[index]
        bboxes = targets["bboxes"] # (N, 4)
        bboxes_out = torch.zeros_like(bboxes)
        bboxes_out[:, 0] = (0.5 * bboxes[:, 0] + bboxes[:, 2]) / im_info["width"]
        bboxes_out[:, 1] = (0.5 * bboxes[:, 1] + bboxes[:, 3]) / im_info["height"]
        bboxes_out[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / im_info["width"]
        bboxes_out[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / im_info["height"]
        targets["bboxes"] = bboxes_out
        
        yolo_targets = make_yolo_targets(
            targets, 
            grid_size=self.grid_size,
            num_classes=self.num_classes
        )
        im_tensor = self.transform(im_tensor)
        
        return im_tensor, targets, im_info, yolo_targets