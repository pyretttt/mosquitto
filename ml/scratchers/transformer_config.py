from dataclasses import dataclass

@dataclass
class TransformerConfig:
    input_size: int
    attn_d_k: int
    transformer_proj_dim: int
    dropout: float
    nlayers: int
    is_self_attn: bool
    max_seq_len: int
    nheads: int
    pre_layer_norm: bool
