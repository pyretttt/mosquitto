import torch
import torch.nn as nn


from scratchers.vanilla_transformer.transformer import TransformerConfig
from scratchers.positional_embeddings import PositionalEmbeddings


class TorchGPT(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.attn_d_k,
            nhead=config.nheads,
            dim_feedforward=config.transformer_proj_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=config.pre_layer_norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.nlayers
        )
        self.projector1 = nn.Linear(config.input_size, config.attn_d_k)
        self.pemb = PositionalEmbeddings(config.max_seq_len, config.attn_d_k)
        self.projector2 = nn.Linear(config.attn_d_k, config.input_size)

    def forward(self, x):
        x = self.projector1(x)
        x = self.pemb(x)
        x = self.encoder.forward(
            x, 
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(-2)), 
            is_causal=True
        )
        return self.projector2(x)
    
    def predict(self, x):
        return self(x)[:, -1:, :]
