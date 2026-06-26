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

from sparse_generalization.layers.priors import LaplacePrior
from sparse_generalization.utils.util_funcs import vae_log_prob, reparametrize

class FlowMasking(nn.Module):
    def __init__(
        self,
        embed_size: int,
        seq_len: int,
        latent_dim: int = 16,
        num_heads: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        bias: float = 0.5,
        prior_type: str = 'laplace',
        per_mask_prior: bool = False, 
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

        self.encoder_lstm = nn.LSTM(
            embed_size,
            latent_dim // 2 if bidirectional else latent_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder = nn.Linear(latent_dim, latent_dim * num_heads)

        self.v = nn.Parameter(torch.randn(seq_len, seq_len))
        nn.init.xavier_uniform_(self.v)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, seq_len)
        )

        self.prior_type = prior_type

        assert not (self.prior_type == 'a_laplace' and per_mask_prior), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == 'nf':
            self.prior = zuko.flows.NSF(
                features=latent_dim,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == 'laplace':
            self.prior = LaplacePrior()
        else:
            self.prior = nn.Identity()

        base_flow = zuko.flows.NSF(
            features=latent_dim,
            context=latent_dim, 
            transforms=flow_params["n_flows"],
            hidden_features=flow_params["hidden_features"],
        )
        self.normalizing_flow = Flow(base_flow.transform.inv, base_flow.base)

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        queries_split = queries.view(batch_size, seq_len, self.heads, self.dk).permute(0, 2, 1, 3)
        keys_split = keys.view(batch_size, seq_len, self.heads, self.dk).permute(0, 2, 1, 3)
        values_split = values.view(batch_size, seq_len, self.heads, self.dk).permute(0, 2, 1, 3)

        out, _ = self.encoder_lstm(x)
        encoding = self.encoder(out[:, -1, :]).view(batch_size, self.heads, -1)
        encoding = encoding.reshape(batch_size * self.heads, -1)

        attention_repr, mask_per_head, attn_per_head, ladj, prior = self._attention(
            queries_split, keys_split, values_split, encoding
        )

        attention_repr = attention_repr.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_size)
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

    def _attention(
        self: Self, query: Tensor, key: Tensor, value: Tensor, encoding: Tensor
    ):
        ladj, prior = 0, 0
        batch_size, heads, seq_len, _ = query.size()
        batch_heads = batch_size * heads

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dk)
        attention_probs = F.softmax(attention_logits, dim=-1)

        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == 'nf':
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        g = self.decoder(latent_nf)
        v_dir = F.normalize(self.v, dim=-1).unsqueeze(0).expand(batch_heads, -1, -1)
        mask_weights_raw = g.view(-1, seq_len, 1) * v_dir
        edges_logit = mask_weights_raw.view(batch_heads, -1) + self.bias
        
        edges_logit = torch.stack([torch.zeros_like(edges_logit), edges_logit], dim=-1)
        
        A = gumbel_softmax(edges_logit, tau=1.0, hard=True)  
        A = A[:, :, -1].view(batch_size, heads, seq_len, seq_len) 

        if self.prior_type == 'laplace' and self.training and self.per_mask_prior:
            prior = self.prior().log_prob(A.sum(dim=(-2, -1)))
        elif self.prior_type == 'uniform' and self.training and self.per_mask_prior:
            prior = torch.tensor([1.0], device=query.device).expand_as(ladj)

        masked_attention_probs = A * attention_probs
        hidden_repr = torch.matmul(masked_attention_probs, value)
        
        if self.residual and not getattr(self, 'mask_res', False):
            eye = torch.eye(seq_len, device=A.device).view(1, 1, seq_len, seq_len)
            A = A + eye

        return hidden_repr, A, masked_attention_probs, ladj, prior
    
class FlowMHA(nn.Module):

    def __init__(
        self,
        embed_size: int,
        seq_len: int = 25,
        latent_dim: int = 16,
        num_heads: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = 'laplace',
        per_mask_prior: bool = False, 
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

        # vae encoder-decoder
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
        assert not (self.prior_type == 'a_laplace' and per_mask_prior), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == 'nf':
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

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        # encoder
        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        encoding = self.encoder(out[:, -1, :]).view(batch_size, -1)

        attention_repr, mask_per_head, attn_per_head, ladj, prior = self._attention(
            queries, keys, values, encoding
        )

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
        self: Self, query: Tensor, key: Tensor, value: Tensor, encoding: Tensor
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = query.size()
        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == 'nf':
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        gq, gk, gv, go = torch.chunk(
            self.decoder(latent_nf), chunks=4, dim=-1
        )
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

        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.heads, seq_len, self.dk)
        )
        attention_repr = torch.bmm(attention_repr, Wo)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, seq_len, seq_len)),
            attention_probs.view(-1, self.heads, seq_len, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )


class FlowDirectA(nn.Module):

    def __init__(
        self,
        embed_size: int,
        seq_len: int = 25,
        latent_dim: int = 16,
        num_heads: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = 'laplace',
        per_mask_prior: bool = False, 
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

        # vae encoder-decoder
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
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128,  seq_len)
        )
        self.attention_weights = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(seq_len, seq_len))
        )
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.prior_type = prior_type
        assert self.prior_type != "laplace", "FlowMHA doesn't support priors"
        assert not (self.prior_type == 'a_laplace' and per_mask_prior), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == 'nf':
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

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        # encoder
        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        encoding = self.encoder(out[:, -1, :]).view(batch_size, self.heads, -1)
        encoding = encoding.reshape(batch_size * self.heads, -1)

        attention_repr, mask_per_head, attn_per_head, ladj, prior = self._attention(
            queries, keys, values, encoding
        )

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
        self: Self, query: Tensor, key: Tensor, value: Tensor, encoding: Tensor
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = query.size()
        batch_heads = batch_size * self.heads
        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == 'nf':
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        g = self.decoder(latent_nf)
        attn_dir = F.normalize(self.attention_weights, dim=-1).unsqueeze(0).expand(batch_heads, -1, -1)
        attention_logits = g.view(-1, seq_len, 1) * attn_dir

        values = self.values(value)
        values_split = self._split_heads(values)

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
            attention_probs.view(-1, self.heads, seq_len, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )


class FlowOnlyQK(nn.Module):

    def __init__(
        self,
        embed_size: int,
        seq_len: int = 25,
        latent_dim: int = 16,
        num_heads: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        prior_type: str = 'laplace',
        per_mask_prior: bool = False, 
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

        # vae encoder-decoder
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
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128,  2 * embed_size)
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
        assert not (self.prior_type == 'a_laplace' and per_mask_prior), "Cant have both adaptive laplace and per mask prior"
        if self.prior_type == 'nf':
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

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        batch_size, seq_len, _ = queries.size()
        x = queries.clone()
        # encoder
        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        encoding = self.encoder(out[:, -1, :])
    
        attention_repr, mask_per_head, attn_per_head, ladj, prior = self._attention(
            queries, keys, values, encoding
        )

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
        self: Self, query: Tensor, key: Tensor, value: Tensor, encoding: Tensor
    ):
        ladj, prior = 0, 0
        batch_size, seq_len, _ = query.size()
        transform = self.normalizing_flow(encoding)

        if self.training:
            latent_nf, ladj = transform.rsample_and_log_prob()
            if self.prior_type == 'nf':
                prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform.sample()

        gq, gk = torch.chunk(
            self.decoder(latent_nf), chunks=2, dim=-1
        )
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

        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.heads, seq_len, self.dk)
        )
        attention_repr = self.projection(attention_repr)

        return (
            attention_repr,
            torch.ones((batch_size, self.heads, seq_len, seq_len)),
            attention_probs.view(-1, self.heads, seq_len, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )
