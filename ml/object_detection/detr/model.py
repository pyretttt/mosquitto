from typing import Optional

import torch
from torch import Tensor


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float
    ):
        self.self_attention = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)


    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        src2 = self.norm1(src)
        k = src + pos if pos is not None else pos
        q = src + pos if pos is not None else pos

        src2 = self.self_attention(q, k, values=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(
            self.dropout(
                torch.function.relu(
                    self.linear1(src2)
                )
            )
        )
        src = src + self.dropout2(src2)
        return src


class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers, encoder_layer, norm):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            encoder_layer.deep_copy() for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos
            )
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(dim_feedforward, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = torch.nn.Linear(d_model, dim_feedforward)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.dropout3 = torch.nn.Dropout(p=dropout)


    def forward(
        self,
        tgt,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        # Implement
        tgt2 = self.self_attn(tgt + pos, tgt + pos, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(
            query_pos + tgt,
            pos + memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(
            self.dropout(
                torch.function.relu(
                    self.linear1(tgt)
                )
            )
        )
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(torch.nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            decoder_layer.deep_copy() for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        tgt = tgt
        intermediate = []
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos
            )
        if self.return_intermediate:
            intermediate.append(tgt)

        if self.norm is not None:
            tgt = self.norm(tgt)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return tgt.unsqueeze(0)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        num_encoder_layer: int = 6,
        num_decoder_layers: int = 6,
        dropout=0.1,
        return_intermediates=False
    ):
        super().__init__()
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layer,
            encoder_layer=TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            norm=encoder_norm
        )
        decoder_norm = torch.nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            encoder_layer=TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            norm=decoder_norm
        )
        self._reset_parameters()
        self.return_intermediates = return_intermediates


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) # H*W, N, C
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # # H*W, N, C
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        memory = self.encoder(
            src=src,
            src_key_padding_mask=mask,
            pos=pos_embed
        )
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder.forward(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
