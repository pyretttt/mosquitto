from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import math

import functions as fn
import functions.boxes as box_fn

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
        rpn_prenms_topk: int,
        **kwargs
    ):
        super().__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.iou_pos_threshold = iou_pos_threshold
        self.iou_neg_threshold = iou_neg_threshold
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_prenms_topk = rpn_prenms_topk
        self.num_anchors = len(scales) * len(aspect_ratios)
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.regression_head = nn.Conv2d(input_channels, self.num_anchors * 4, kernel_size=1)
        self.classification_head = nn.Conv2d(input_channels, self.num_anchors, kernel_size=1)
        self.relu = nn.ReLU()


    def init_weights(self):
        for layer in [self.conv1, self.regression_head, self.classification_head]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)


    def generate_anchors(self, image, feat):
        return box_fn.generate_anchors(image, feat, self.scales, self.aspect_ratios)


    def generate_anchors_(self, image, feat):
        r"""
        Method to generate anchors. First we generate one set of zero-centred anchors
        using the scales and aspect ratios provided.
        We then generate shift values in x,y axis for all featuremap locations.
        The single zero centred anchors generated are replicated and shifted accordingly
        to generate anchors for all feature map locations.
        Note that these anchors are generated such that their centre is top left corner of the
        feature map cell rather than the centre of the feature map cell.
        :param image: (N, C, H, W) tensor
        :param feat: (N, C_feat, H_feat, W_feat) tensor
        :return: anchor boxes of shape (H_feat * W_feat * num_anchors_per_location, 4)
        """
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]
        
        # For the vgg16 case stride would be 16 for both h and w
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)
        
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)
        
        # Assuming anchors of scale 128 sq pixels
        # For 1:1 it would be (128, 128) -> area=16384
        # For 2:1 it would be (181.02, 90.51) -> area=16384
        # For 1:2 it would be (90.51, 181.02) -> area=16384
        
        # The below code ensures h/w = aspect_ratios and h*w=1
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        # Now we will just multiply h and w with scale(example 128)
        # to make h*w = 128 sq pixels and h/w = aspect_ratios
        # This gives us the widths and heights of all anchors
        # which we need to replicate at all locations
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        # Now we make all anchors zero centred
        # So x1, y1, x2, y2 = -w/2, -h/2, w/2, h/2
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        # Get the shifts in x axis (0, 1,..., W_feat-1) * stride_w
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w

        # Get the shifts in x axis (0, 1,..., H_feat-1) * stride_h
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        
        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        # shifts_x -> (H_feat, W_feat)
        # shifts_y -> (H_feat, W_feat)
        
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        # Setting shifts for x1 and x2(same as shifts_x) and y1 and y2(same as shifts_y)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)
        # shifts -> (H_feat * W_feat, 4)
        
        # base_anchors -> (num_anchors_per_location, 4)
        # shifts -> (H_feat * W_feat, 4)
        # Add these shifts to each of the base anchors
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        # anchors -> (H_feat * W_feat, num_anchors_per_location, 4)
        anchors = anchors.reshape(-1, 4)
        # anchors -> (H_feat * W_feat * num_anchors_per_location, 4)
        return anchors

    
    def map_proposals(
        self, 
        proposals: torch.Tensor, 
        cls_scores: torch.Tensor, 
        image_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Maps proposals

        Args:
            proposals (torch.Tensor): N x anchor.size(0) x 4
            cls_scores (torch.Tensor):  N * anchor.size(0) x 1
            image_shape (tuple[int, int]): width x height

        Returns:
            torch.Tensor: _description_
        """
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(min(self.rpn_prenms_topk, len(cls_scores)))
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[..., top_n_idx]
        proposals = fn.clamp_boxes_to_shape(proposals, image_shape)
        
        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep_idx = torch.where(keep)[0]
        proposals = proposals[keep_idx]
        cls_scores = cls_scores[keep_idx]

        # NMS
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, self.rpn_nms_threshold)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]

        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        
        proposals, cls_scores = (
            proposals[post_nms_keep_indices[:self.rpn_topk]],
            cls_scores[post_nms_keep_indices[:self.rpn_topk]],
        )

        return proposals, cls_scores


    def forward(self, image, feat, target=None):
        rpn_feat = self.conv1(feat)
        rpn_feat = self.relu(rpn_feat)
        cls_scores = self.classification_head(rpn_feat) # N x Num_achors x H_feat x W_feat
        cls_scores = cls_scores.permute(0, 2, 3, 1)  # N x H_feat x W_feat x Num_achors
        cls_scores = cls_scores.reshape(-1, 1) # N * H_feat * W_feat * Num_achors x 1
        
        box_tranform_pred = self.regression_head(rpn_feat) # N x Num_achors * 4 x H_feat x W_feat
        assert box_tranform_pred.shape == (image.shape[0], self.num_anchors * 4, feat.shape[-2], feat.shape[-1])
        box_tranform_pred = box_tranform_pred.view(
            box_tranform_pred.size(dim=0),
            self.num_anchors,
            4,
            rpn_feat.shape[-2], 
            rpn_feat.shape[-1]
        ) # N x Num_achors x 4 x H_feat x H_head
        box_tranform_pred = box_tranform_pred.permute(0, 3, 4, 1, 2) # N x H_feat x H_head x Num_achors x 4
        box_tranform_pred = box_tranform_pred.reshape(-1, 4) # N * H_feat * H_head * Num_achors x 4
        
        anchors = self.generate_anchors(image, feat) # H_feat * W_feat * Num_achors x 4
        
        proposals = box_fn.apply_transformations_to_anchors(
            anchors,
            box_tranform_pred.detach()
        ) # N x anchor.size(0) x 4
        proposals, cls_scores = self.map_proposals(proposals, cls_scores, image.shape)
        out = {
            "proposals": proposals,
            "scores": cls_scores
        }

        return out


class RCNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder



if __name__ == "__main__":
    # Check RPN
    img = torch.randn(1, 3, 224, 320)
    backbone = Backbone()
    out = backbone(img)
    print(out.shape)
    
    vgg = torchvision.models.vgg16(pretrained=False)
    print(vgg)
    out = vgg.features[:-1](img)
    print(out.shape)
    
    # Check region proposals
    rpn = RPN(400, (128, 256), (1, 2), 0, 0, 0)
    image = torch.randn((2, 400, 256, 256))
    feat = torch.randn((2, 400, 16, 16))
    my_regions = rpn.generate_anchors(image[0], feat)
    orig_regions = rpn.generate_anchors_(image[0], feat)
    
    assert my_regions.equal(orig_regions)
    
    # out = rpn(image, feat)
    orig_out = rpn
    
    ## Other Tests
    print("Other Tests")
    a = torch.randn((4, 4))
    print(a[None, ..., 0:1].shape)