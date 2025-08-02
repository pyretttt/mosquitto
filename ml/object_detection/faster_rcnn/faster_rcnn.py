from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import math

@dataclass
class Config:
    iou_neg_threshold: float
    iou_pos_threshold: float
    extractor_channels: int
    num_ratios: int
    num_scales: int


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.extractor(x)


class RPN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_ratios: int,
        num_scales: int,
        **kwargs
    ):
        self.num_anchors = num_ratios * num_scales
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.regression_head = nn.Conv2d(input_channels, self.num_anchors * 4, kernel_size=1)
        self.classification_head = nn.Conv2d(input_channels, self.num_anchors, kernel_size=1)
        
        
    def init_weights(self):
        for layer in [self.conv1, self.regression_head, self.classification_head]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    
        


class RCNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder



if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 320)
    backbone = Backbone()
    out = backbone(img)

    print(out.shape)
    
    vgg = torchvision.models.vgg16(pretrained=False)
    print(vgg)
    out = vgg.features[:-1](img)
    print(out.shape)