from hydra.utils import instantiate
import torch
import torch.nn as nn
import zuko

from torch import Tensor
from typing import List, Self

from sparse_generalization.models.mlp import BasicMLP
from sparse_generalization.layers.bern_mha import MultiHeadAttentionBern
from sparse_generalization.layers.oracle import MultiHeadAttentionOracle
from sparse_generalization.layers.gen_mha import FlowMasking, FlowMHA
from sparse_generalization.layers.film_attn import FiLMAttention, FiLMLayer


class MHABlock(nn.Module):

    def __init__(
        self: Self,
        embed_size: int,
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
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self: Self, x: Tensor):
        return self._forward_image(x)

    def _forward_image(self: Self, x: Tensor):

        if self.layernorm:
            x_ln = self.ln1(x)
            attn_out, attn_scores = self.mha(x_ln, x_ln, x_ln)
            if self.residual:
                attn_out = attn_out + x
                out = self.mlp(self.ln2(attn_out))
                out = out + attn_out
            else:
                out = self.mlp(self.ln2(attn_out))
        else:
            attn_out, attn_scores = self.mha(x, x, x)
            if self.residual:
                out = self.mlp(attn_out + x)
                out = out + attn_out
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
        act: nn.Module,
        dropout: int,
        layernorm: bool,
        residual: bool,
        zeros: bool,
        mask_res: bool = False,
        separate_mask: bool = False,
        alpha_res: bool = False,
        *args,
        **kwargs,
    ):
        super(MHABlockBern, self).__init__(*args, **kwargs)
        self.residual = residual
        self.layernorm = layernorm
        self.mask_res = mask_res
        if mask_res:
            assert num_heads == 1, "heads needs to be 1 else residual doesnt make sense"

        if alpha_res:
            self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.alpha_res = alpha_res

        self.mha = MultiHeadAttentionBern(
            embed_size,
            num_heads=num_heads,
            dropout=dropout,
            zeros=zeros,
            mask_res=mask_res,
            separate_mask=separate_mask,
            residual=residual,
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self: Self, x: Tensor, forced_expl: bool = False):
        return self._forward_image(x, forced_expl)

    def _forward_image(self: Self, x: Tensor, forced_expl: bool = False):

        if self.layernorm:
            x_ln = self.ln1(x)
            attn_out, attn_masks, masked_attn_scores, attn_scores = self.mha(
                x_ln, x_ln, x_ln, forced_expl=forced_expl
            )
            if self.mask_res:
                diags = attn_masks.diagonal(dim1=-2, dim2=-1)
                x = x * diags.unsqueeze(dim=-1)  # (B, L, L) * (B, L, D)

            if self.residual:
                attn_out = attn_out + x
                out = self.mlp(self.ln2(attn_out))
                out = out + attn_out
            else:
                out = self.mlp(self.ln2(attn_out))
        else:
            attn_out, attn_masks, masked_attn_scores, attn_scores = self.mha(x, x, x)
            if self.residual:
                out = self.mlp(attn_out + x)
                out = out + attn_out
            else:
                out = self.mlp(attn_out)

        return out, attn_masks, masked_attn_scores, attn_scores

    def mha_parameters(self: Self):
        return (
            list(self.feature_map.parameters()) + list(self.mha.parameters())
            if self.use_grid
            else list(self.mha.parameters())
        )


class MHABlockGen(nn.Module):

    def __init__(
        self: Self,
        embed_size: int,
        act: nn.Module,
        seq_len: int,
        dropout: int,
        layernorm: bool,
        base_dist: zuko.lazy.LazyDistribution,
        residual: bool,
        separate_mask: bool,
        num_heads: int = 1,
        force_vae_gaussian: bool = False,
        use_mask: bool = False,
        mha_layer: nn.Module = FlowMasking,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        device: str = "cuda",
        *args,
        **kwargs,
    ):
        super(MHABlockGen, self).__init__(*args, **kwargs)
        self.residual = residual
        self.layernorm = layernorm
        self.per_mask_prior = per_mask_prior

        self.mha = mha_layer(
            embed_size,
            seq_len=seq_len,
            num_heads=num_heads,
            base_dist=base_dist,
            separate_mask=separate_mask,
            prior_params=prior_params,
            layernorm=layernorm,
            use_mask=use_mask,
            flow_params=flow_params,
            force_vae_gaussian=force_vae_gaussian,
            device=device,
            per_mask_prior=per_mask_prior,
            prior_type=prior_type,
        )

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self: Self, x: Tensor):
        return self._forward_image(x)

    def _forward_image(self: Self, x: Tensor):

        if self.layernorm:
            x_ln = self.ln1(x)
            if self.training:
                attn_out, attn_masks, attn_scores, prior, ladj = self.mha(
                    x_ln, x_ln, x_ln
                )
            else:
                attn_out, attn_masks, attn_scores = self.mha(x_ln, x_ln, x_ln)
            if self.residual:
                attn_out = attn_out + x
                out = self.mlp(self.ln2(attn_out))
                out = out + attn_out
            else:
                out = self.mlp(self.ln2(attn_out))
        else:
            if self.training:
                attn_out, attn_masks, attn_scores, prior, ladj = self.mha(x, x, x)
            else:
                attn_out, attn_masks, attn_scores = self.mha(x, x, x)
            if self.residual:
                out = self.mlp(attn_out + x)
                out = out + attn_out
            else:
                out = self.mlp(attn_out)

        if self.training:
            return out, attn_masks, attn_scores, prior, ladj
        else:
            return (
                out,
                attn_masks,
                attn_scores,
            )


class MHABlockCond(nn.Module):
    def __init__(
        self: Self,
        embed_size: int,
        context_dim: int,
        act: nn.Module,
        dropout: int,
        layernorm: bool,
        residual: bool,
        num_heads: int = 1,
        temp: float = 1.0,
        num_layers_film: int = 2,
        film_mlp: bool = False,
        device: str = "cuda",
        *args,
        **kwargs,
    ):
        super(MHABlockCond, self).__init__(*args, **kwargs)
        self.residual = residual
        self.layernorm = layernorm
        self.film_mlp = film_mlp

        self.mha = FiLMAttention(
            embed_size,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            temp=temp,
            act=act,
            num_layers_film=num_layers_film,
            residual=residual,
        )

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        if film_mlp:
            self.first_mlp = nn.Sequential(
                nn.Linear(embed_size, 4 * embed_size),
                nn.Dropout(dropout),
            )
            self.film_layer = FiLMLayer(
                4 * embed_size, context_dim, num_layers_film, act
            )
            self.second_mlp = nn.Sequential(
                act(),
                nn.Linear(4 * embed_size, embed_size),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_size, 4 * embed_size),
                nn.Dropout(dropout),
                act(),
                nn.Linear(4 * embed_size, embed_size),
            )

    def _mlp(self: Self, x: Tensor, context: Tensor):
        if self.film_mlp:
            out = self.first_mlp(x)  # (b, m, l, 4d)
            out = self.film_layer(out, context, per_mode=True)
            return self.second_mlp(out)
        return self.mlp(x)

    def forward(self: Self, x: Tensor, context: Tensor, avg_heads: bool = True):
        return self._forward_image(x, context, avg_heads)

    def _forward_image(self: Self, x: Tensor, context: Tensor, avg_heads: bool = True):

        if self.layernorm:
            x_ln = self.ln1(x)
            attn_out, attn_masks, masked_attn_scores, attn_scores = self.mha(
                x_ln, x_ln, x_ln, context, avg_attn_heads=avg_heads, avg_mask=avg_heads
            )
            if self.residual:
                attn_out = attn_out + x
                out = self._mlp(self.ln2(attn_out), context)
                out = out + attn_out
            else:
                out = self._mlp(self.ln2(attn_out), context)
        else:
            attn_out, attn_masks, masked_attn_scores, attn_scores = self.mha(
                x, x, x, context, avg_attn_heads=avg_heads, avg_mask=avg_heads
            )
            if self.residual:
                out = self._mlp(attn_out + x, context)
                out = out + attn_out
            else:
                out = self._mlp(attn_out, context)

        return out, attn_masks, masked_attn_scores, attn_scores


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
