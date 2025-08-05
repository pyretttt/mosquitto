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
    scales: list[int]
    ratios: list[int]
    rpn_nms_threshold: float


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
        scales: list[int],
        aspect_ratios: list[int],
        iou_pos_threshold: float,
        iou_neg_threshold: float,
        rpn_nms_threshold: float,
        **kwargs
    ):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.iou_pos_threshold = iou_pos_threshold
        self.iou_neg_threshold = iou_neg_threshold
        self.rpn_nms_threshold = rpn_nms_threshold
        self.num_anchors = len(scales) * len(aspect_ratios)
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.regression_head = nn.Conv2d(input_channels, self.num_anchors * 4, kernel_size=1)
        self.classification_head = nn.Conv2d(input_channels, self.num_anchors, kernel_size=1)
        
        
    def init_weights(self):
        for layer in [self.conv1, self.regression_head, self.classification_head]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    

    def generate_anchors(self, image, feat):
        """Computes anchors for each sliding window inside feature map
        For each sliding window inside feature map there're num_scales * num_ratios anchors generated

        Args:
            image (torch.Tensor): (N x C x H x W) 
            feat (torch.Tensor): (N x C_feat x H_feat x W_feat) feature map from backbone
        """
        H_feat, W_feat = feat.shape[-2:]
        H_image, W_image = image.shape[-2:]
        
        stride_h = torch.tensor(H_image // H_feat, dtype=torch.int64, device=image.device)
        stride_w = torch.tensor(W_image // W_feat, dtype=torch.int64, device=image.device)

        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device) # (num_scales)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)
        
        # Assuming anchors of scale 128 sq pixels
        # For 1:1 it would be (128, 128) -> area=16384
        # For 2:1 it would be (181.02, 90.51) -> area=16384
        # For 1:2 it would be (90.51, 181.02) -> area=16384
        
        # Look explanation inside [README.md](#RPN-Params)
        h_ratios = torch.sqrt(aspect_ratios) # (num_ratios)
        w_ratios = 1 / h_ratios # (num_ratios)
        
        # Compute actual side scales
        h_scales = (h_ratios[:, None] * scales[None, :]).view(-1) # (num_ratios * num_scales)
        w_scales = (w_ratios[:, None] * scales[None, :]).view(-1) # (num_ratios * num_scales)
        
        ## Anchor displacements about it center in image coordinates
        anchor_coords_about_center = torch.stack([-w_scales, -h_scales, w_scales, h_scales]) / 2
        anchor_coords_about_center = anchor_coords_about_center.round()
        
        # Computes anchor 
        shift_x = torch.arange(0, W_feat, dtype=torch.int32, device=feat.device) * stride_w
        shift_y = torch.arange(0, H_feat, dtype=torch.int32, device=feat.device) * stride_h


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