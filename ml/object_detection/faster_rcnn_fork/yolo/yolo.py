import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0
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
        self.depth = num_anchors * 5 + num_classes
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        
        layers = [
            ConvBlock(3, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
        ]
        
        for _ in range(4):
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        
        layers += [
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        ]
        
        for i in range(2):                                                          # Conv 5
            layers += [
                ConvBlock(1024, 512, kernel_size=1),
                ConvBlock(512, 1024, kernel_size=3, padding=1),
            ]
        layers += [
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        ]

        layers = [
            nn.Flatten(),
            nn.Linear(grid_size * grid_size * 1024, 4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, self.depth * self.grid_size * self.grid_size)
        ]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return (
            self.model(x)
                .reshape(x.size(dim=0), self.grid_size, self.grid_size, self.depth)
        )