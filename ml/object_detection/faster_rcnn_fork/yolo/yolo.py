import torch
from torch import nn

class YOLO(nn.Module):
    def __init__(
        self,
        num_classes: int, 
        num_anchors: int = 3, 
        grid_size: int = 7
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(
                in_features=448 / (2 ** 3), out_features=
            )
        )
        
    def forward(self, x, y,):
        pass