import torch

def get_spatial_position_embedding(emb_dim, feat_map):
    H, W = feat_map.shape[2], feat_map.shape[3]
    vertical_indices = torch.arange(H, dtype=torch.float32, device=feat_map.device)
    horizontal_indices = torch.arange(W, dtype=torch.float32, device=feat_map.device)
    grid = torch.meshgrid(vertical_indices, horizontal_indices, indexing="ij")
    grid = torch.stack(grid, dim=0) # 2, 