import torch
from torch import nn

import torchvision.models as models


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
        num_anchors: int,
        grid_size: int
    ):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        
        self.depth = num_anchors * 5 + num_classes
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        
        self.conv_layers = nn.Sequential(
            ConvBlock(2048, 512, kernel_size=1, padding=0),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            # nn.MaxPool2d(2, 2),
            ConvBlock(512, 512, kernel_size=3, padding=1),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(512 * self.grid_size * self.grid_size, out_features=4096), # Implicitly we know the size of in channels
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, self.grid_size * self.grid_size * self.depth),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.dense_layers(
                self.conv_layers(
                    self.backbone(x)
                ) # N, 512, 14, 14
                .flatten(start_dim=1) # N, 512 * 14 * 14
            ) # N, 14 * 14 * 30