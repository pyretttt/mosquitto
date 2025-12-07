import torch


def get_spatial_position_embedding(emb_dim, feat_map):
    H, W = feat_map.shape[2], feat_map.shape[3]
    vertical_indices = torch.arange(H, dtype=torch.float32, device=feat_map.device)
    horizontal_indices = torch.arange(W, dtype=torch.float32, device=feat_map.device)
    grid = torch.meshgrid(vertical_indices, horizontal_indices, indexing="ij") # (H,) (W,) -> (H, W), (H, W)
    grid = torch.stack(grid, dim=0) # 2, H, W
    grid_h = grid[0].reshape(-1).unsqueeze(1) # H * W, 1
    grid_w = grid[1].reshape(-1).unsqueeze(1) # H * W, 1

    factor = torch.exp(
        -torch.log(10000)
        * torch.arange(0, emb_dim, 2, device=feat_map.device).float()
        / emb_dim
    ) # (emb_dim // 2)

    grid_h_emb = grid_h * factor # H * W, embed_dim // 2
    grid_w_emb = grid_w * factor # H * W, embed_dim // 2

    grid_h_emb = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1) # H * W, embed_dim

    grid_w_emb = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1) # H * W, embed_dim

    pos_emb = torch.cat([
        grid_h_emb,
        grid_w_emb
    ], dim=-1) # H * W, embed_dim

    return pos_emb