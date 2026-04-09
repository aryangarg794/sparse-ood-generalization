from hydra.utils import instantiate
import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Self

from sparse_generalization.models.mlp import BasicMLP
from sparse_generalization.layers.bern_mha import MultiHeadAttentionBern
from sparse_generalization.layers.oracle import MultiHeadAttentionOracle


class MHABlock(nn.Module):

    def __init__(
        self: Self,
        embed_size: int,
        out_dim: int,
        hidden_dims: List,
        act: nn.Module,
        dropout: int,
        residual: bool,
        mha_layer: nn.Module,
        layernorm: bool,
        num_heads: int,
        *args,
        **kwargs,
    ):
        self.residual = residual
        self.layernorm = layernorm
        super(MHABlock, self).__init__(*args, **kwargs)

        self.mha = mha_layer(
            embed_size, num_heads=num_heads, batch_first=True
        )  # (b, 3, 1) or (b, 3, 2) with pe
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = BasicMLP(
            input_dim=embed_size,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            act=act,
            dropout=dropout,
        )  # (b, 3)

    def forward(self: Self, x: Tensor):
        return self._forward_image(x)

    def _forward_image(self: Self, x: Tensor):
        attn_out, attn_scores = self.mha(x, x, x)
        if self.layernorm:
            if self.residual:
                attn_out = self.ln1(attn_out + x)
                out = self.mlp(attn_out)
                out = self.ln2(out + attn_out)
            else:
                attn_out = self.ln1(attn_out)
                out = self.mlp(attn_out)
                out = self.ln2(out)
        else:
            if self.residual:
                out = self.mlp(attn_out + x)
            else:
                out = self.mlp(attn_out)

        return out, attn_scores

    def mha_parameters(self: Self):
        return (
            list(self.feature_map.parameters()) + list(self.mha.parameters())
            if self.use_grid
            else list(self.mha.parameters())
        )


class MHABlockBern(nn.Module):
    """Basic transformer block for the toy example

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self: Self,
        embed_size: int,
        num_heads: int,  # for the toy example just keep it one
        hidden_dims: list,
        act: nn.Module,
        dropout: int,
        layernorm: bool,
        residual: bool,
        separate_mask: bool = False,
        alpha_res: bool = False,
        *args,
        **kwargs,
    ):
        super(MHABlockBern, self).__init__(*args, **kwargs)
        self.residual = residual
        self.layernorm = layernorm

        if alpha_res:
            self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.alpha_res = alpha_res

        self.mha = MultiHeadAttentionBern(
            embed_size,
            num_heads=num_heads,
            dropout=dropout,
            separate_mask=separate_mask,
            residual=residual,
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = BasicMLP(
            input_dim=embed_size,
            out_dim=embed_size,
            hidden_dims=hidden_dims,
            act=act,
            dropout=dropout,
        )

    def forward(self: Self, x: Tensor):
        return self._forward_image(x)

    def _forward_image(self: Self, x: Tensor):
        attn_out, attn_masks, attn_scores = self.mha(x, x, x)
        if self.layernorm:
            if self.residual:
                attn_out = self.ln1(attn_out + x)
                out = self.mlp(attn_out)
                out = self.ln2(out + attn_out)
            else:
                attn_out = self.ln1(attn_out)
                out = self.mlp(attn_out)
                out = self.ln2(out)
        else:
            if self.residual:
                out = self.mlp(attn_out + x)
            else:
                out = self.mlp(attn_out)

        return out, attn_masks, attn_scores

    def mha_parameters(self: Self):
        return (
            list(self.feature_map.parameters()) + list(self.mha.parameters())
            if self.use_grid
            else list(self.mha.parameters())
        )


class MHABlockOracle(MHABlockBern):

    def __init__(
        self,
        embed_size: int,
        use_grid: bool,
        num_heads: int,
        act: nn.Module,
        hidden_dims: list,
        dropout: int,
        residual: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            embed_size,
            use_grid,
            num_heads,
            hidden_dims,
            act,
            dropout,
            residual,
            *args,
            **kwargs,
        )

        self.mha = MultiHeadAttentionOracle(
            embed_size, num_heads=num_heads, dropout=dropout, residual=residual
        )

    def forward(self: Self, x: Tensor, edges: List):
        if self.use_grid:
            return self._forward_image(x, edges)
        else:
            return self._forward_basic(x, edges)

    def _forward_image(self, x, true_edges):
        assert self.use_grid

        attn_out, attn_masks, attn_scores = self.mha(x, x, x, true_edges)
        if self.residual:
            attn_out = self.ln1(attn_out + x)  # (b, l, d)
            out = self.mlp(attn_out)
            out = self.ln2(out + attn_out)
        else:
            out = self.ln1(attn_out)
            out = self.mlp(out)
            out = self.ln2(out)

        return out, attn_masks, attn_scores
