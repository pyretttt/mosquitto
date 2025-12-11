import torch
from torch import nn
import numpy as np

def get_spatial_position_embedding(emb_dim, feat_map):
    H, W = feat_map.shape[2], feat_map.shape[3]
    vertical_indices = torch.arange(H, dtype=torch.float32, device=feat_map.device)
    horizontal_indices = torch.arange(W, dtype=torch.float32, device=feat_map.device)
    grid = torch.meshgrid(vertical_indices, horizontal_indices, indexing="ij") # (H,) (W,) -> (H, W), (H, W)
    grid_h = grid[0].reshape(-1).unsqueeze(1) # H * W, 1
    grid_w = grid[1].reshape(-1).unsqueeze(1) # H * W, 1
    assert grid_h.shape == (H * W, 1) and grid_w.shape == (H * W, 1)

    factor = torch.exp(
        -np.log(10000)
        * torch.arange(0, end=emb_dim // 4, device=feat_map.device).float()
        / (emb_dim // 4)
    ) # (emb_dim // 4)
    assert factor.shape == (emb_dim // 4, )

    grid_h_emb = grid_h * factor # H * W, embed_dim // 4
    grid_w_emb = grid_w * factor # H * W, embed_dim // 4
    assert grid_h_emb.shape == grid_w_emb.shape and grid_h_emb.shape == (H * W, emb_dim // 4)

    print(grid_h)

    pos_h = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1) # H * W, embed_dim // 2

    pos_w = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1) # H * W, embed_dim // 2

    pos = torch.cat([
        pos_h,
        pos_w
    ], dim=-1) # H * W, embed_dim

    return pos


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ff_inner_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        