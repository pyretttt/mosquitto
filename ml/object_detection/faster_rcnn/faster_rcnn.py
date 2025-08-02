import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import math

## RPN returns classes and 

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

    def forward(self, x):
        return self.extractor(x)


class RPN(nn.Module):
    def __init__(
        self,
        
    ):
        ...

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