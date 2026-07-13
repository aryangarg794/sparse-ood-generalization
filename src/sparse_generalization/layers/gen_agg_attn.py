import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax
from typing import Self
from zuko.flows import Flow

from sparse_generalization.layers.vae import FlowVAE
from sparse_generalization.layers.priors import LaplacePrior


class AggregationFlowMask(nn.Module):
    def __init__(
        self,
        embed_size: int,
        seq_len: int,
        base_dist: zuko.lazy.LazyDistribution,
        num_heads: int = 1,
        out_dim: int = 1,
        separate_mask: bool = False,
        use_mask: bool = False,
        dropout: float = 0.0,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        act: nn.Module = nn.ReLU,
        layernorm: bool = True,
        device: str = "cuda",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        assert num_heads == 1, "Only 1 head implemented"

        self.embed_size = embed_size
        self.heads = num_heads
        self.dk = embed_size // num_heads
        self.layernorm = layernorm
        self.per_mask_prior = per_mask_prior

        self.query = nn.Parameter(torch.zeros((1, embed_size)))
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

        self.v = nn.Parameter(torch.randn(1, seq_len))
        nn.init.xavier_uniform_(self.v)

        self.prior_type = prior_type
        if self.prior_type == "nf":
            self.prior = zuko.flows.NSF(
                features=1,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == "laplace":
            self.prior = LaplacePrior()
        else:
            self.prior = nn.Identity()

        self.param_flow = FlowVAE(
            input_dim=embed_size,
            output_dim=1,
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

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, out_dim),
        )

    def forward(self, x: Tensor, sum_heads: bool = True):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = x.size()

        query_expanded = self.query.expand(batch_size, 1, -1)
        queries = self.queries(query_expanded)
        keys = self.keys(x)
        values = self.values(x)

        queries_split = queries.view(batch_size, 1, self.heads, self.dk).permute(
            0, 2, 1, 3
        )
        keys_split = keys.view(batch_size, seq_len, self.heads, self.dk).permute(
            0, 2, 1, 3
        )
        values_split = values.view(batch_size, seq_len, self.heads, self.dk).permute(
            0, 2, 1, 3
        )

        batch_heads = self.heads * batch_size
        g, ladj = self.param_flow(x)
        v_dir = F.normalize(self.v, dim=-1).unsqueeze(0).expand(batch_heads, -1, -1)
        mask_weights_raw = g.view(-1, 1, 1) * v_dir

        edges_logit = mask_weights_raw.view(batch_heads, -1) + self.bias
        edges_logit = torch.stack([torch.zeros_like(edges_logit), edges_logit], dim=-1)
        A = gumbel_softmax(edges_logit, tau=1.0, hard=True)
        A = A[:, :, -1].view(batch_size, self.heads, 1, seq_len)

        attention_repr, masks, attention_probs = self._attention(
            queries_split, keys_split, values_split, A
        )

        attention_repr = attention_repr.permute(0, 2, 1, 3).reshape(
            batch_size, 1, self.embed_size
        )

        if self.prior_type == "laplace" and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(A.sum(dim=(-2, -1)))
        elif self.prior_type == "uniform" and self.training and self.per_mask_prior:
            prior = torch.tensor([1.0]).expand_as(ladj)

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        out = self.mlp(attention_repr.squeeze(dim=1))

        if self.training:
            return out, masks, attention_probs, prior, ladj
        return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        A: Tensor,
    ):

        batch_size, heads, seq_len, _ = key.size()
        batch_heads = batch_size * heads

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.dk
        )
        attention_probs = F.softmax(attention_logits, dim=-1)

        attention_probs = A * attention_probs
        hidden_repr = torch.matmul(attention_probs, value)

        return (
            hidden_repr,
            A,
            attention_probs,
        )


class AggregationFlowMHA(nn.Module):

    def __init__(
        self,
        embed_size: int,
        seq_len: int,
        latent_dim: int = 16,
        num_heads: int = 1,
        out_dim: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        act: nn.Module = nn.ReLU,
        layernorm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        assert num_heads == 1, "Only 1 head implemented"

        self.embed_size = embed_size
        self.heads = num_heads
        self.dk = embed_size // num_heads
        self.layernorm = layernorm

        self.query = nn.Parameter(torch.zeros((1, embed_size)))
        nn.init.uniform_(self.query)
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

        self.encoder_lstm = nn.LSTM(
            embed_size,
            latent_dim // 2 if bidirectional else latent_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 4 * embed_size)
        )

        self.Wq = nn.init.uniform_(nn.Parameter(torch.zeros(embed_size, embed_size)))
        self.Wk = nn.init.uniform_(nn.Parameter(torch.zeros(embed_size, embed_size)))
        self.Wv = nn.init.uniform_(nn.Parameter(torch.zeros(embed_size, embed_size)))
        self.Wo = nn.init.uniform_(nn.Parameter(torch.zeros(embed_size, embed_size)))

        self.prior_type = prior_type
        assert self.prior_type != "laplace", "FlowMHA doesn't support priors"
        assert not (
            self.prior_type == "a_laplace" and per_mask_prior
        ), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == "nf":
            self.prior = zuko.flows.NSF(
                features=latent_dim,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        else:
            self.prior = nn.Identity()

        base_flow = zuko.flows.NSF(
            features=latent_dim,
            context=latent_dim,
            transforms=flow_params["n_flows"],
            hidden_features=flow_params["hidden_features"],
        )
        self.normalizing_flow = Flow(base_flow.transform.inv, base_flow.base)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, out_dim),
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

    def forward(self, x: Tensor, sum_heads: bool = True):
        batch_size, seq_len, _ = x.size()

        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        encoding = self.encoder(out[:, -1, :]).view(batch_size, -1)

        attention_repr, masks, attention_probs, ladj, prior = self._attention(
            self.query.repeat(batch_size, 1, 1), x, x, encoding
        )

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        out = self.mlp(attention_repr.squeeze(dim=1))
        if self.training:
            return out, masks, attention_probs, prior, ladj

        return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        encoding: Tensor,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = key.size()
        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == "nf":
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        gq, gk, gv, go = torch.chunk(self.decoder(latent_nf), chunks=4, dim=-1)
        vq_dir = F.normalize(self.Wq, dim=-1)
        vk_dir = F.normalize(self.Wk, dim=-1)
        vv_dir = F.normalize(self.Wv, dim=-1)
        vo_dir = F.normalize(self.Wo, dim=-1)

        Wq = gq.view(-1, self.embed_size, 1) * vq_dir
        Wk = gk.view(-1, self.embed_size, 1) * vk_dir
        Wv = gv.view(-1, self.embed_size, 1) * vv_dir
        Wo = go.view(-1, self.embed_size, 1) * vo_dir

        queries = torch.bmm(query, Wq)  # (b, l, k) @ (b, k, k)
        keys = torch.bmm(key, Wk)
        values = torch.bmm(value, Wv)

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

        attention_repr = self._merge_heads(hidden_repr.view(-1, self.heads, 1, self.dk))
        attention_repr = torch.bmm(attention_repr, Wo)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, 1, seq_len)),
            attention_probs.view(-1, self.heads, 1, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )


class AggregationFlowDirectA(nn.Module):

    def __init__(
        self,
        embed_size: int,
        seq_len: int,
        latent_dim: int = 16,
        num_heads: int = 1,
        out_dim: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        act: nn.Module = nn.ReLU,
        layernorm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        assert num_heads == 1, "Only 1 head implemented"

        self.embed_size = embed_size
        self.heads = num_heads
        self.dk = embed_size // num_heads
        self.layernorm = layernorm

        self.query = nn.Parameter(torch.zeros((1, embed_size)))
        nn.init.uniform_(self.query)

        self.encoder_lstm = nn.LSTM(
            embed_size,
            latent_dim // 2 if bidirectional else latent_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder = nn.Linear(latent_dim, latent_dim * num_heads)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.attention_weights = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(1, seq_len))
        )
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.prior_type = prior_type
        assert self.prior_type != "laplace", "FlowMHA doesn't support priors"
        assert not (
            self.prior_type == "a_laplace" and per_mask_prior
        ), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == "nf":
            self.prior = zuko.flows.NSF(
                features=latent_dim,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        else:
            self.prior = nn.Identity()

        base_flow = zuko.flows.NSF(
            features=latent_dim,
            context=latent_dim,
            transforms=flow_params["n_flows"],
            hidden_features=flow_params["hidden_features"],
        )
        self.normalizing_flow = Flow(base_flow.transform.inv, base_flow.base)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, out_dim),
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

    def forward(self, x: Tensor, sum_heads: bool = True):
        batch_size, seq_len, _ = x.size()
        # encoder
        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        encoding = self.encoder(out[:, -1, :]).view(batch_size, self.heads, -1)
        encoding = encoding.reshape(batch_size * self.heads, -1)

        attention_repr, masks, attention_probs, ladj, prior = self._attention(
            self.query.repeat(batch_size, 1, 1), x, x, encoding
        )

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        out = self.mlp(attention_repr.squeeze(dim=1))
        if self.training:
            return out, masks, attention_probs, prior, ladj

        return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        encoding: Tensor,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = key.size()
        batch_heads = batch_size * self.heads
        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == "nf":
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        g = self.decoder(latent_nf)
        attn_dir = (
            F.normalize(self.attention_weights, dim=-1)
            .unsqueeze(0)
            .expand(batch_heads, -1, -1)
        )
        attention_logits = g.view(-1, 1, 1) * attn_dir

        values = self.values(value)
        values_split = self._split_heads(values)

        attention_probs = softmax(attention_logits, dim=-1)
        attention_probs = torch.clamp(attention_probs, min=0.001, max=0.999)
        hidden_repr = torch.bmm(attention_probs, values_split)

        attention_repr = self._merge_heads(hidden_repr.view(-1, self.heads, 1, self.dk))
        attention_repr = self.projection(attention_repr)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, 1, seq_len)),
            attention_probs.view(-1, self.heads, 1, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )


class AggregationFlowOnlyQK(nn.Module):

    def __init__(
        self,
        embed_size: int,
        seq_len: int,
        latent_dim: int = 16,
        num_heads: int = 1,
        out_dim: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        act: nn.Module = nn.ReLU,
        layernorm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        assert num_heads == 1, "Only 1 head implemented"

        self.embed_size = embed_size
        self.heads = num_heads
        self.dk = embed_size // num_heads
        self.layernorm = layernorm

        self.query = nn.Parameter(torch.zeros((1, embed_size)))
        nn.init.uniform_(self.query)

        self.encoder_lstm = nn.LSTM(
            embed_size,
            latent_dim // 2 if bidirectional else latent_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 2 * embed_size)
        )
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
            self.prior = zuko.flows.NSF(
                features=latent_dim,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        else:
            self.prior = nn.Identity()

        base_flow = zuko.flows.NSF(
            features=latent_dim,
            context=latent_dim,
            transforms=flow_params["n_flows"],
            hidden_features=flow_params["hidden_features"],
        )
        self.normalizing_flow = Flow(base_flow.transform.inv, base_flow.base)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, out_dim),
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

    def forward(self, x: Tensor, sum_heads: bool = True):
        batch_size, seq_len, _ = x.size()
        # encoder
        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        encoding = self.encoder(out[:, -1, :])

        attention_repr, masks, attention_probs, ladj, prior = self._attention(
            self.query.repeat(batch_size, 1, 1), x, x, encoding
        )

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        out = self.mlp(attention_repr.squeeze(dim=1))
        if self.training:
            return out, masks, attention_probs, prior, ladj

        return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        encoding: Tensor,
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = key.size()
        batch_heads = batch_size * self.heads
        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == "nf":
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        gq, gk = torch.chunk(self.decoder(latent_nf), chunks=2, dim=-1)
        vq_dir = F.normalize(self.Wq, dim=-1)
        vk_dir = F.normalize(self.Wk, dim=-1)

        Wq = gq.view(-1, self.embed_size, 1) * vq_dir
        Wk = gk.view(-1, self.embed_size, 1) * vk_dir

        queries = torch.bmm(query, Wq)  # (b, l, k) @ (b, k, k)
        keys = torch.bmm(key, Wk)
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

        attention_repr = self._merge_heads(hidden_repr.view(-1, self.heads, 1, self.dk))
        attention_repr = self.projection(attention_repr)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, 1, seq_len)),
            attention_probs.view(-1, self.heads, 1, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )
