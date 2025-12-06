import torch

def get_spatial_position_embedding(emb_dim, feat_map):
    H, W = feat_map.shape[2], feat_map.shape[3]
    vertical_indices = torch.arange(H, dtype=torch.float32, device=feat_map.device)
    horizontal_indices = torch.arange(W, dtype=torch.float32, device=feat_map.device)
    grid = torch.meshgrid(vertical_indices, horizontal_indices, indexing="ij") # (H,) (W,) -> (H, W), (H, W)
    grid = torch.stack(grid, dim=0) # 2, H, W
    grid_h = grid[0].reshape(-1) # H * W
    grid_w = grid[1].reshape(-1) # H * W

    factor = 10_000
