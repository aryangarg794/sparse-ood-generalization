import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax
from typing import Self


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
        use_mlp: bool = True,
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
        self.use_mlp = use_mlp

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

        if self.use_mlp:
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
            forced_expl=forced_expl,
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

        if self.use_mlp:
            out = self.mlp(attention_repr.squeeze(dim=1))
        else:
            out = attention_repr.squeeze(dim=1)

        return out, masks, masked_probs, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor,
        keys_mask: Tensor,
        forced_expl: bool = False,
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
                A = torch.randint(
                    0, 2, size=(batch_heads, 1, seq_len), device=query.device
                ).float()
            else:
                edges_logit = attention_logits.view(batch_heads, -1)  # (b*h, l*l)
                edges_logit = torch.stack(
                    [torch.zeros_like(edges_logit), edges_logit + self.bias], dim=-1
                )
                A = gumbel_softmax(edges_logit, tau=self.temp, hard=True)
                A = A[:, :, -1].reshape(batch_heads, 1, seq_len)

            masked_attention_probs = A * attention_probs
        else:
            A = torch.ones(batch_heads, 1, seq_len)
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
