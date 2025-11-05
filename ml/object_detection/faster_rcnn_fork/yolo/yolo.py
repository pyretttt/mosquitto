import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


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
        
        self.backbone = self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
        )
        
        self.head = nn.Conv2d(in_channels=128, out_channels=(num_anchors * 5 + num_classes) * grid_size ** 2, kernel_size=1)
        
    def forward(self, x, y,):
        pass