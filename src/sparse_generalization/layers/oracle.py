import math
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.functional import gumbel_softmax, softmax
from typing import Self, Callable


class MultiHeadAttentionOracle(nn.Module):
    """Implements oracle MHA, currently set to a single head.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self: Self,
        embed_size: int,
        num_heads: int = 1,
        bias: int = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        residual: bool = False,
        size: int = 10,
        *args,
        **kwargs,
    ):
        super(MultiHeadAttentionOracle, self).__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual

        self.queries = nn.Linear(embed_size, embed_size, bias=bias)
        self.keys = nn.Linear(embed_size, embed_size, bias=bias)
        self.values = nn.Linear(embed_size, embed_size, bias=bias)
        self.projection = nn.Linear(embed_size, embed_size, bias=bias)
        self.height = size
        self.width = size

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        oracle_edges: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        queries = self.queries(queries)  # (b, l, d)
        keys = self.keys(keys)
        values = self.values(values)

        queries_split = self._split_heads(queries)  # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_repr, mask_per_head, adjacency_per_head = self._attention(
            queries_split, keys_split, values_split, oracle_edges
        )

        attention_repr = self._merge_heads(attention_repr)  # (b, l, d)
        attention_repr = self.projection(attention_repr)

        if avg_attn_heads:
            adjacency = adjacency_per_head.mean(dim=1)

        if avg_mask:
            mask = mask_per_head.mean(dim=1)

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
        self: Self, query: Tensor, key: Tensor, value: Tensor, oracle_edges: Tensor
    ):
        batch_heads, seq_len, _ = query.size()
        batch = int(batch_heads / self.heads)
        # attention_logits = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.dk) # (b*h, l, l)

        # attention_probs = softmax(attention_logits, dim=-1)
        A = torch.zeros((batch_heads, seq_len, seq_len), device=query.device)
        # convert the nodes, coords to get nodes, 1

        for i, item in enumerate(oracle_edges):  # (num_paths, num_edges, 2, 2)
            num_paths, num_edges, _, _ = item.shape
            item = item.to(query.device)
            # NOTE: right now we check ALL paths but maybe we should change this to check a random one
            item = item.view(num_paths * num_edges, 2, 2)
            item = item[:, :, 1] * self.width + item[:, :, 0]  # (paths * edges, 2)
            for h in range(self.heads):
                A[i + batch * h, item[:, 0], item[:, 1]] = 1

        masked_attention_probs = A  # (b*h, l, l)

        hidden_repr = torch.bmm(masked_attention_probs, value)  # (b*h, l, d)

        if self.residual:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0).repeat(
                batch_heads, 1, 1
            )

        return (
            hidden_repr.view(-1, self.heads, seq_len, self.dk),
            A.view(-1, self.heads, seq_len, seq_len),
            masked_attention_probs.view(-1, self.heads, seq_len, seq_len),
        )
