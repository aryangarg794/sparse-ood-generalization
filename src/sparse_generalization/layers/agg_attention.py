import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax
from typing import Self

from sparse_generalization.models.mlp import BasicMLP


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

        self.query = nn.Parameter(torch.rand((1, embed_size), device=device))
        nn.init.xavier_uniform_(self.query)

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
            nn.Linear(embed_size, 4*embed_size), 
            nn.Dropout(dropout), 
            act(),
            nn.Linear(4*embed_size, out_dim)
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

        if self.separate_mask:
            queries_mask = self.queries_mask(self.query_mask.repeat(batch_size, 1, 1))
            queries_mask_split = self._split_heads(queries_mask)
            keys_mask = self.keys_mask(x)
            keys_mask_split = self._split_heads(keys_mask)

        attention_repr, masks, attention_probs = self._attention(
            queries_split,
            keys_split,
            values_split,
            queries_mask_split if self.separate_mask else None,
            keys_mask_split if self.separate_mask else None,
        )

        attention_repr = self._merge_heads(attention_repr)  # (b, 1, d)

        if sum_heads:
            masks = masks.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        if self.layernorm:
            attention_repr = self.ln(attention_repr)

        out = self.mlp(attention_repr.squeeze(dim=1))

        return out, masks, attention_probs

    def _attention(
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor,
        keys_mask: Tensor,
    ):
        batch_heads, seq_len, _ = key.size()
        attention_logits = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(
            self.dk
        )  # (bh, 1, dk) @ (bh, dk, l)

        attention_probs = softmax(attention_logits, dim=-1)
        A = torch.zeros((batch_heads, seq_len, seq_len))

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
            else:
                edges_logit = attention_logits.view(batch_heads, -1)  # (b*h, l*l)
                edges_logit = torch.stack(
                    [torch.zeros_like(edges_logit), edges_logit], dim=-1
                )
                A = gumbel_softmax(edges_logit, tau=self.temp, hard=True)
                A = A[:, :, -1].reshape(batch_heads, 1, seq_len)
            
            attention_probs = A * attention_probs

        # (bh, 1, l)
        hidden_repr = torch.bmm(attention_probs, value)  # (bh, 1, l) @ (bh, l, dk)

        return hidden_repr.view(-1, self.heads, 1, self.dk), A.view(-1, self.heads, 1, seq_len), attention_probs.view(
            -1, self.heads, 1, seq_len
        )