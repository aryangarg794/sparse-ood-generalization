import torch
import torch.nn as nn
import zuko
import numpy as np
import math
import torch.nn.functional as F

from zuko.flows import Flow
from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax
from typing import Self

from sparse_generalization.layers.priors import LaplacePrior, NormalPrior
from sparse_generalization.layers.vae import FlowVAE


class FlowMasking(nn.Module):
    def __init__(
        self,
        embed_size: int,
        seq_len: int,
        base_dist: zuko.lazy.LazyDistribution,
        num_heads: int = 1,
        flow_params: dict = {"n_flows": 3, "hidden_features": [128, 128]},
        prior_params: dict = {"n_flows": 3, "hidden_features": [128, 128]},
        residual: bool = False,
        bias: float = 0.5,
        prior_type: str = "laplace",
        device: str = "cuda",
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        per_mask_prior: bool = False,
        force_vae_gaussian: bool = False, 
        *args,
        **kwargs,
    ):
        super(FlowMasking, self).__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.residual = residual
        self.bias = bias
        self.per_mask_prior = per_mask_prior

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.v = nn.Parameter(torch.randn(seq_len, seq_len))
        nn.init.xavier_uniform_(self.v)

        self.prior_type = prior_type

        assert not (
            self.prior_type == "a_laplace" and per_mask_prior
        ), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == "nf":
            self.prior = zuko.flows.MAF(
                features=seq_len,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == "laplace":
            self.prior = LaplacePrior()
        elif self.prior_type == "normal":
            self.prior = NormalPrior()
        else:
            self.prior = nn.Identity()

        self.param_flow = FlowVAE(
            input_dim=embed_size,
            output_dim=seq_len,
            base_dist=base_dist,
            num_heads=num_heads,
            encoder_heads=True,
            use_encoder=True,
            layernorm=layernorm,
            device=device,
            flow_params=flow_params,
            use_mask=use_mask,
            force_vae_gaussian=force_vae_gaussian, 
            separate_mask=separate_mask,
        )

    def _split_heads(self: Self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.heads, self.dk)
            .transpose(1, 2)
            .reshape(batch_size * self.heads, seq_len, self.dk)
        )

    def _merge_heads(self: Self, x: Tensor):
        batch_size, _, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, self.heads, seq_len, self.dk)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.dk * self.heads)
        )

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        queries_split = self._split_heads(queries)  # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        batch_heads = self.heads * batch_size
        g, ladj = self.param_flow(x)
        v_dir = F.normalize(self.v, dim=-1).unsqueeze(0).expand(batch_heads, -1, -1)
        mask_weights_raw = g.view(-1, seq_len, 1) * v_dir

        edges_logit = mask_weights_raw.view(batch_heads, -1) + self.bias
        edges_logit = torch.stack([torch.zeros_like(edges_logit), edges_logit], dim=-1)

        A = gumbel_softmax(edges_logit, tau=1.0, hard=True)
        A = A[:, :, -1].view(batch_heads, seq_len, seq_len)

        attention_repr, mask_per_head, attn_per_head = self._attention(
            queries_split, keys_split, values_split, A
        )

        if self.prior_type == "laplace" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(A.sum(dim=(-2, -1)))
        elif self.prior_type == "normal" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g).sum(dim=-1)
        elif self.prior_type == "nf" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g)
        elif self.prior_type == "uniform" and self.training and self.per_mask_prior:
            prior = torch.tensor([1.0], device=queries.device).expand_as(ladj)

        attention_repr = self._merge_heads(attention_repr)
        attention_repr = self.projection(attention_repr)

        if avg_attn_heads:
            adjacency = attn_per_head.sum(dim=1)
        else:
            adjacency = attn_per_head

        if avg_mask:
            mask = mask_per_head.sum(dim=1)
        else:
            mask = mask_per_head

        if self.training:
            return attention_repr, mask, adjacency, prior, ladj

        return attention_repr, mask, adjacency

    def _attention(self: Self, query: Tensor, key: Tensor, value: Tensor, A: Tensor):
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.dk
        )
        attention_probs = F.softmax(attention_logits, dim=-1)

        _, seq_len, _ = query.shape

        masked_attention_probs = A * attention_probs
        hidden_repr = torch.matmul(masked_attention_probs, value)

        if self.residual and not getattr(self, "mask_res", False):
            eye = torch.eye(seq_len, device=A.device).view(1, 1, seq_len, seq_len)
            A = A + eye

        return (
            hidden_repr.view(-1, self.heads, seq_len, self.dk),
            A.view(-1, self.heads, seq_len, seq_len),
            masked_attention_probs.view(-1, self.heads, seq_len, seq_len),
        )
class FlowMHA(nn.Module):

    def __init__(
        self,
        embed_size: int,
        base_dist: zuko.lazy.LazyDistribution,
        seq_len: int = 25,
        num_heads: int = 1,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        device: str = "cuda",
        layernorm: bool = True,
        separate_mask: bool = False,
        force_vae_gaussian: bool = False,
        use_mask: bool = False,
        *args,
        **kwargs,
    ):

        super(FlowMHA, self).__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.residual = residual
        self.per_mask_prior = per_mask_prior
        self.force_vae_gaussian = force_vae_gaussian

        self.Wq = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(embed_size, embed_size))
        )
        self.Wk = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(embed_size, embed_size))
        )
        self.Wv = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(embed_size, embed_size))
        )
        self.Wo = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(embed_size, embed_size))
        )

        self.prior_type = prior_type
        assert self.prior_type != "laplace", "FlowMHA doesn't support priors"
        assert not (
            self.prior_type == "a_laplace" and per_mask_prior
        ), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == "nf":
            self.prior = zuko.flows.MAF(
                features=4 * embed_size,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == "laplace":
            self.prior = LaplacePrior()
        elif self.prior_type == "normal":
            self.prior = NormalPrior()
        else:
            self.prior = nn.Identity()

        self.param_flow = FlowVAE(
            input_dim=embed_size,
            output_dim=4 * embed_size,
            base_dist=base_dist,
            num_heads=num_heads,
            encoder_heads=False,
            use_encoder=False,
            layernorm=layernorm,
            device=device,
            flow_params=flow_params,
            use_mask=use_mask,
            separate_mask=separate_mask,
        )

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()

        batch_heads = self.heads * batch_size
        g, ladj = self.param_flow(x)
        gq, gk, gv, go = torch.chunk(g, chunks=4, dim=-1)
        vq_dir = F.normalize(self.Wq, dim=-1)
        vk_dir = F.normalize(self.Wk, dim=-1)
        vv_dir = F.normalize(self.Wv, dim=-1)
        vo_dir = F.normalize(self.Wo, dim=-1)

        Wq = gq.view(self.embed_size, 1) * vq_dir
        Wk = gk.view(self.embed_size, 1) * vk_dir
        Wv = gv.view(self.embed_size, 1) * vv_dir
        Wo = go.view(self.embed_size, 1) * vo_dir

        attention_repr, attn_per_head = self._attention(
            queries, keys, values, Wq, Wk, Wv, Wo
        )

        mask_per_head = torch.ones((batch_size, self.heads, seq_len, seq_len))

        if self.prior_type == "normal" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g).sum(dim=-1)
        elif self.prior_type == "nf" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g)
        elif self.prior_type == "uniform" and self.training and self.per_mask_prior:
            prior = torch.tensor([1.0], device=queries.device).expand_as(ladj)

        if avg_attn_heads:
            adjacency = attn_per_head.sum(dim=1)

        if avg_mask:
            mask = mask_per_head.sum(dim=1)

        if self.training:
            return attention_repr, mask, adjacency, prior, ladj

        return attention_repr, mask, adjacency

    def _split_heads(self: Self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.heads, self.dk)
            .transpose(1, 2)
            .reshape(batch_size * self.heads, seq_len, self.dk)
        )

    def _merge_heads(self: Self, x: Tensor):
        batch_size, _, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, self.heads, seq_len, self.dk)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.dk * self.heads)
        )

    def _attention(
        self: Self, query: Tensor, key: Tensor, value: Tensor, query_mat: Tensor, key_mat: Tensor, 
        value_mat: Tensor, proj_mat: Tensor
    ):
        _, seq_len, _ = query.shape
        queries = query @ query_mat  # (b, l, k) @ (k, k)
        keys = key @ key_mat
        values = value @ value_mat

        queries_split = self._split_heads(queries)  # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_logits = torch.bmm(
            queries_split, keys_split.transpose(1, 2)
        ) / np.sqrt(
            self.dk
        )  # (b*h, l, l)

        attention_probs = softmax(attention_logits, dim=-1)
        attention_probs = torch.clamp(attention_probs, min=0.001, max=0.999)
        hidden_repr = torch.bmm(attention_probs, values_split)

        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.heads, seq_len, self.dk)
        )
        attention_repr = attention_repr @ proj_mat

        return (
            attention_repr,
            attention_probs.view(-1, self.heads, seq_len, seq_len)
        )


class FlowDirectA(nn.Module):

    def __init__(
        self,
        embed_size: int,
        base_dist: zuko.lazy.LazyDistribution,
        seq_len: int = 25,
        force_vae_gaussian: bool = False, 
        num_heads: int = 1,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        device: str = "cuda",
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        *args,
        **kwargs,
    ):

        super(FlowDirectA, self).__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.residual = residual
        self.per_mask_prior = per_mask_prior

        self.attention_weights = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(seq_len, seq_len))
        )
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.prior_type = prior_type
        assert self.prior_type != "laplace", "FlowMHA doesn't support priors"
        assert not (
            self.prior_type == "a_laplace" and per_mask_prior
        ), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == "nf":
            self.prior = zuko.flows.MAF(
                features=seq_len,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == "laplace":
            self.prior = LaplacePrior()
        elif self.prior_type == "normal":
            self.prior = NormalPrior()
        else:
            self.prior = nn.Identity()

        self.param_flow = FlowVAE(
            input_dim=embed_size,
            output_dim=seq_len,
            base_dist=base_dist,
            num_heads=num_heads,
            encoder_heads=True,
            use_encoder=True,
            layernorm=layernorm,
            device=device,
            flow_params=flow_params,
            use_mask=use_mask,
            separate_mask=separate_mask,
        )

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        
        batch_heads = self.heads * batch_size
        g, ladj = self.param_flow(x)
        attn_dir = (
            F.normalize(self.attention_weights, dim=-1)
            .unsqueeze(0)
            .expand(batch_heads, -1, -1)
        )
        attention_logits = g.view(-1, seq_len, 1) * attn_dir

        attention_repr, mask_per_head, attn_per_head = self._attention(
            queries, keys, values, attention_logits
        )

        if self.prior_type == "normal" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g).sum(dim=-1)
        elif self.prior_type == "nf" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g)
        elif self.prior_type == "uniform" and self.training and self.per_mask_prior:
            prior = torch.tensor([1.0], device=queries.device).expand_as(ladj)

        if avg_attn_heads:
            adjacency = attn_per_head.sum(dim=1)

        if avg_mask:
            mask = mask_per_head.sum(dim=1)

        if self.training:
            return attention_repr, mask, adjacency, prior, ladj

        return attention_repr, mask, adjacency

    def _split_heads(self: Self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.heads, self.dk)
            .transpose(1, 2)
            .reshape(batch_size * self.heads, seq_len, self.dk)
        )

    def _merge_heads(self: Self, x: Tensor):
        batch_size, _, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, self.heads, seq_len, self.dk)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.dk * self.heads)
        )

    def _attention(
        self: Self, query: Tensor, key: Tensor, value: Tensor, attn_logits: Tensor
    ):
        batch_size, seq_len, _ = query.size()

        values = self.values(value)
        values_split = self._split_heads(values)

        attention_probs = softmax(attn_logits, dim=-1)
        attention_probs = torch.clamp(attention_probs, min=0.001, max=0.999)
        hidden_repr = torch.bmm(attention_probs, values_split)

        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.heads, seq_len, self.dk)
        )

        attention_repr = self.projection(attention_repr)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, seq_len, seq_len)),
            attention_probs.view(-1, self.heads, seq_len, seq_len),
        )


class FlowOnlyQK(nn.Module):

    def __init__(
        self,
        embed_size: int,
        base_dist: zuko.lazy.LazyDistribution,
        seq_len: int = 25,
        num_heads: int = 1,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        force_vae_gaussian: bool = False, 
        per_mask_prior: bool = False,
        device: str = "cuda",
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        *args,
        **kwargs,
    ):

        super(FlowOnlyQK, self).__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.residual = residual
        self.per_mask_prior = per_mask_prior

        self.Wq = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(embed_size, embed_size))
        )
        self.Wk = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(embed_size, embed_size))
        )

        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.prior_type = prior_type
        assert self.prior_type != "laplace", "FlowMHA doesn't support priors"
        assert not (
            self.prior_type == "a_laplace" and per_mask_prior
        ), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == "nf":
            self.prior = zuko.flows.MAF(
                features=2 * embed_size,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == "laplace":
            self.prior = LaplacePrior()
        elif self.prior_type == "normal":
            self.prior = NormalPrior()
        else:
            self.prior = nn.Identity()

        self.param_flow = FlowVAE(
            input_dim=embed_size,
            output_dim=2 * embed_size,
            base_dist=base_dist,
            num_heads=num_heads,
            encoder_heads=False,
            use_encoder=False,
            layernorm=layernorm,
            device=device,
            flow_params=flow_params,
            use_mask=use_mask,
            separate_mask=separate_mask,
        )

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        
        g, ladj = self.param_flow(x)
        gq, gk = torch.chunk(g, chunks=2, dim=-1)
        vq_dir = F.normalize(self.Wq, dim=-1)
        vk_dir = F.normalize(self.Wk, dim=-1)

        Wq = gq.view(self.embed_size, 1) * vq_dir
        Wk = gk.view(self.embed_size, 1) * vk_dir

        attention_repr, mask_per_head, attn_per_head = self._attention(
            queries, keys, values, Wq, Wk
        )

        if self.prior_type == "normal" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g).sum(dim=-1)
        elif self.prior_type == "nf" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(g)
        elif self.prior_type == "uniform" and self.training and self.per_mask_prior:
            prior = torch.tensor([1.0], device=queries.device).expand_as(ladj)

        if avg_attn_heads:
            adjacency = attn_per_head.sum(dim=1)

        if avg_mask:
            mask = mask_per_head.sum(dim=1)

        if self.training:
            return attention_repr, mask, adjacency, prior, ladj

        return attention_repr, mask, adjacency

    def _split_heads(self: Self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.heads, self.dk)
            .transpose(1, 2)
            .reshape(batch_size * self.heads, seq_len, self.dk)
        )

    def _merge_heads(self: Self, x: Tensor):
        batch_size, _, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, self.heads, seq_len, self.dk)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.dk * self.heads)
        )

    def _attention(
        self: Self, query: Tensor, key: Tensor, value: Tensor, query_mat: Tensor, key_mat: Tensor
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = query.size()        

        queries = query @ query_mat  # (b, l, k) @ (k, k)
        keys = key @ key_mat
        values = self.values(value)

        queries_split = self._split_heads(queries)  # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_logits = torch.bmm(
            queries_split, keys_split.transpose(1, 2)
        ) / np.sqrt(
            self.dk
        )  # (b*h, l, l)

        attention_probs = softmax(attention_logits, dim=-1)
        attention_probs = torch.clamp(attention_probs, min=0.001, max=0.999)
        hidden_repr = torch.bmm(attention_probs, values_split)

        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.heads, seq_len, self.dk)
        )
        attention_repr = self.projection(attention_repr)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, seq_len, seq_len)),
            attention_probs.view(-1, self.heads, seq_len, seq_len)
        )
