import torch
import torch.nn as nn

class RCNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
           
    ):
        super().__init__()
        self.encoder = encoder
