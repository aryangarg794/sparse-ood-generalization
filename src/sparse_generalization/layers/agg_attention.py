import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax
from typing import Self
from zuko.flows import Flow

from sparse_generalization.utils.util_funcs import vae_log_prob, reparametrize


class AggregationAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        out_dim: int,
        dropout: float = 0.0,
        residual: bool = True,
        act: nn.Module = nn.ReLU,
        device: str = "cuda",
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        bias: float = 0.5,
        temp: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.embed_size = embed_size
        self.heads = num_heads
        self.dk = embed_size // num_heads
        self.residual = residual
        self.layernorm = layernorm
        self.use_mask = use_mask
        self.temp = temp
        self.bias = bias

        self.query = nn.Parameter(torch.zeros((1, embed_size), device=device))
        nn.init.uniform_(self.query)

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

        self.separate_mask = separate_mask
        if self.separate_mask:
            self.queries_mask = nn.Linear(embed_size, embed_size)
            self.keys_mask = nn.Linear(embed_size, embed_size)
            self.query_mask = nn.Parameter(torch.rand((1, embed_size), device=device))

        self.ln = nn.LayerNorm(embed_size)

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

    def forward(self, x: Tensor, sum_heads: bool = True, forced_expl: bool = True):
        batch_size, seq_len, _ = x.size()
        queries = self.queries(self.query.repeat(batch_size, 1, 1))  # (b, 1, d)
        keys = self.keys(x)  # (b, l, d)
        values = self.values(x)  # (b, l, d)

        queries_split = self._split_heads(queries)  # (b * h, 1, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        if self.separate_mask:
            queries_mask = self.queries_mask(self.query_mask.repeat(batch_size, 1, 1))
            queries_mask_split = self._split_heads(queries_mask)
            keys_mask = self.keys_mask(x)
            keys_mask_split = self._split_heads(keys_mask)

        attention_repr, masks, masked_probs, attention_probs = self._attention(
            queries_split,
            keys_split,
            values_split,
            queries_mask_split if self.separate_mask else None,
            keys_mask_split if self.separate_mask else None,
            forced_expl=forced_expl
        )

        attention_repr = self._merge_heads(attention_repr)  # (b, 1, d)

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        # if self.residual:
        #     pooled_x = x.max(dim=1, keepdim=True)[0]
        #     out = pooled_x + attention_repr

        if self.layernorm:
            attention_repr = self.ln(attention_repr)

        out = self.mlp(attention_repr.squeeze(dim=1))

        return out, masks, masked_probs, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor,
        keys_mask: Tensor,
        forced_expl: bool = False
    ):
        batch_heads, seq_len, _ = key.size()
        attention_logits = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(
            self.dk
        )  # (bh, 1, dk) @ (bh, dk, l)

        attention_probs = softmax(attention_logits, dim=-1)
        attention_probs = torch.clamp(attention_probs, min=0.001, max=0.999)

        if self.use_mask:
            if self.separate_mask:
                mask_logits = torch.bmm(
                    query_mask, keys_mask.transpose(1, 2)
                ) / np.sqrt(self.dk)
                mask_logits = mask_logits.view(batch_heads, -1)
                edges_logit = torch.stack(
                    [torch.zeros_like(mask_logits), mask_logits + self.bias], dim=-1
                )
                A = gumbel_softmax(edges_logit, tau=self.temp, hard=True)
                A = A[:, :, -1].reshape(batch_heads, 1, seq_len)
            elif self.training and forced_expl:
                A = torch.randint(0, 2, size=(batch_heads, 1, seq_len), device=query.device).float()
            else:
                edges_logit = attention_logits.view(batch_heads, -1)  # (b*h, l*l)
                edges_logit = torch.stack(
                    [torch.zeros_like(edges_logit), edges_logit + self.bias], dim=-1
                )
                A = gumbel_softmax(edges_logit, tau=self.temp, hard=True)
                A = A[:, :, -1].reshape(batch_heads, 1, seq_len)

            masked_attention_probs = A * attention_probs
        else:
            masked_attention_probs = attention_probs 

        # (bh, 1, l)
        hidden_repr = torch.bmm(
            masked_attention_probs, value
        )  # (bh, 1, l) @ (bh, l, dk)

        return (
            hidden_repr.view(-1, self.heads, 1, self.dk),
            A.view(-1, self.heads, 1, seq_len),
            masked_attention_probs.view(-1, self.heads, 1, seq_len),
            attention_probs.view(-1, self.heads, 1, seq_len),
        )

    def temp_decay(self, step, total_steps, start_temp=3.0, end_temp=0.1):
        if step >= total_steps:
            return end_temp

        temp_range = start_temp - end_temp
        current_temp = start_temp - (step / total_steps) * temp_range

        self.temp = max(current_temp, end_temp)


class AggregationFlow(nn.Module):

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
        nf_prior: bool = True,
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
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

        self.encoder_lstm = nn.LSTM(
            self.dk,
            latent_dim // 2 if bidirectional else latent_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder = nn.Linear(latent_dim, 2 * latent_dim)

        self.v = nn.Parameter(torch.randn(1, seq_len))
        nn.init.xavier_uniform_(self.v)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.nf_prior = nf_prior
        if self.nf_prior:
            self.prior = zuko.flows.NSF(
                features=latent_dim,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )

        base_flow = zuko.flows.NSF(
            features=latent_dim,
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
        queries = self.queries(self.query.repeat(batch_size, 1, 1))  # (b, 1, d)
        keys = self.keys(x)  # (b, l, d)
        values = self.values(x)  # (b, l, d)

        queries_split = self._split_heads(queries)  # (b * h, 1, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        out, _ = self.encoder_lstm(x)  # assume self-attn (b, k)
        lstm_encoding = self.encoder(out[:, -1, :]).view(batch_size, 2, self.heads, -1)
        mu, sig = torch.chunk(lstm_encoding, chunks=2, dim=1)  # (b, k*h)
        mu = mu.permute(0, 2, 1, 3).squeeze().reshape(batch_size * self.heads, -1)
        sig = sig.permute(0, 2, 1, 3).squeeze().reshape(batch_size * self.heads, -1)
        encoding = reparametrize(mu, sig)

        attention_repr, masks, attention_probs, ladj, prior = self._attention(
            queries_split, keys_split, values_split, encoding
        )

        attention_repr = self._merge_heads(attention_repr)  # (b, 1, d)

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        out = self.mlp(attention_repr.squeeze(dim=1))
        if self.training:
            mha_loss = prior - vae_log_prob(encoding, mu, sig) + ladj
            return out, masks, attention_probs, mha_loss
        else:
            return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        encoding: Tensor,
    ):
        batch_heads, seq_len, _ = key.size()
        attention_logits = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(
            self.dk
        )  # (bh, 1, dk) @ (bh, dk, s)

        attention_probs = softmax(attention_logits, dim=-1)

        transform = self.normalizing_flow().transform

        if self.training:
            latent_nf, ladj = transform.call_and_ladj(encoding)
            prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform(encoding)

        g = self.decoder(latent_nf).squeeze()
        v = self.v.repeat(batch_heads, 1, 1)
        v_dir = F.normalize(v, dim=-1)
        mask_weights_raw = g.view(-1, 1, 1) * v_dir
        mask_weights = F.sigmoid(mask_weights_raw)
        hard_mask = (mask_weights >= 0.5).float()
        A = hard_mask - mask_weights.detach() + mask_weights

        attention_probs = A * attention_probs

        # (bh, 1, l)
        hidden_repr = torch.bmm(attention_probs, value)  # (bh, 1, l) @ (bh, l, dk)

        return (
            hidden_repr.view(-1, self.heads, 1, self.dk),
            A.view(-1, self.heads, 1, seq_len),
            attention_probs.view(-1, self.heads, 1, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )


class AggregationQKV(nn.Module):

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
        nf_prior: bool = True,
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
        self.encoder = nn.Linear(latent_dim, 2 * latent_dim)

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

        self.nf_prior = nf_prior
        if self.nf_prior:
            self.prior = zuko.flows.NSF(
                features=latent_dim,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )

        base_flow = zuko.flows.NSF(
            features=latent_dim,
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
        mu, sig = torch.chunk(self.encoder(out[:, -1, :]), chunks=2, dim=-1)
        encoding = reparametrize(mu, sig)

        attention_repr, masks, attention_probs, ladj, prior = self._attention(
            self.query.repeat(batch_size, 1, 1), x, x, encoding
        )

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        out = self.mlp(attention_repr.squeeze(dim=1))
        if self.training:
            mha_loss = prior - vae_log_prob(encoding, mu, sig) + ladj
            return out, masks, attention_probs, mha_loss
        else:
            return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        encoding: Tensor,
    ):
        batch_size, seq_len, _ = key.size()
        transform = self.normalizing_flow().transform

        if self.training:
            latent_nf, ladj = transform.call_and_ladj(encoding)
            prior = self.prior().log_prob(latent_nf)
        else:
            latent_nf = transform(encoding)

        gq, gk, gv, go = torch.chunk(
            self.decoder(latent_nf).squeeze(), chunks=4, dim=-1
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
        batch_heads = queries_split.size(0)

        attention_logits = torch.bmm(
            queries_split, keys_split.transpose(1, 2)
        ) / np.sqrt(
            self.dk
        )  # (b*h, l, l)

        attention_probs = softmax(attention_logits, dim=-1)
        mask_probs = F.sigmoid(attention_logits)
        hard_mask = (mask_probs >= 0.5).float()
        A = hard_mask - mask_probs.detach() + mask_probs

        masked_attention_probs = A * attention_probs
        hidden_repr = torch.bmm(masked_attention_probs, values_split)

        attention_repr = self._merge_heads(hidden_repr.view(-1, self.heads, 1, self.dk))
        attention_repr = torch.bmm(hidden_repr, Wo)

        return (
            attention_repr,
            A.view(-1, self.heads, 1, seq_len),
            masked_attention_probs.view(-1, self.heads, 1, seq_len),
            ladj if self.training else None,
            prior if self.training else None,
        )
