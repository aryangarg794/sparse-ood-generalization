import torch 
import torch.nn as nn
import numpy as np


from torch import Tensor
from torch.nn.functional import gumbel_softmax, softmax
from typing import Self, Callable

from sparse_generalization.layers.film_attn import FiLMLayer, FiLMMLP
from sparse_generalization.utils.util_funcs import get_device


class FiLMAggAttention(nn.Module):
    def __init__(
        self: Self,
        embed_size: int,
        context_dim: int,
        out_dim: int,
        num_modes: int | None = None, # None -> one query shared across modes
        num_heads: int = 1,
        dropout: float = 0.0,
        temp: float = 1.0,
        hard: bool = True,
        layernorm: bool = False,
        device: str | None = None,
        act: nn.Module = nn.ReLU, 
        num_layers_film: int = 2, 
        residual: bool = False,
        agg_residual: bool = False,
        agg_res_coeff: float = 1.0,
        train_query: bool = True,
        *args,
        **kwargs,
    ):
        device = get_device(device)
        super().__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.num_modes = num_modes
        self.dropout = nn.Dropout(p=dropout)
        self.temp = temp
        self.hard = hard
        self.residual = residual
        self.layernorm = layernorm
        self.agg_residual = agg_residual
        self.agg_res_coeff = agg_res_coeff
        self.bias = 1.0

        self.queries_mask = FiLMLayer(embed_size, context_dim, num_layers_film, act)
        self.keys_mask = FiLMLayer(embed_size, context_dim, num_layers_film, act)

        self.query = nn.Parameter(torch.zeros((num_modes, embed_size), device=device), requires_grad=train_query)
        nn.init.xavier_uniform_(self.query)

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.mlp = FiLMMLP(embed_size, context_dim, out_dim, act, dropout, num_layers_film)

        if agg_residual:
            self.res_proj = nn.Linear(embed_size, embed_size)

        if layernorm:
            self.ln = nn.LayerNorm(embed_size)


    def forward(
        self: Self,
        x: Tensor, # (b, m, l, d) -- already repeated/expanded over modes
        context: Tensor, # (m, context_dim)
        sum_heads: bool = True,
    ):
        batch_size, num_modes, seq_len, _ = x.size()

        # learned aggregation query (per mode, or one shared across modes), shared across the batch
        queries = self.queries(self.query)  # (m, d) or (1, d)
        queries = queries.view(1, -1, 1, self.embed_size).expand(batch_size, num_modes, -1, -1)  # (b, m, 1, d)
        keys = self.keys(x)  # (b, m, l, d)
        values = self.values(x)

        queries_mask = self.queries_mask(queries, context, per_mode=True) # (b, m, 1, d)
        keys_mask = self.keys_mask(keys, context, per_mode=True) # (b, m, l, d)

        queries_split = self._split_heads(queries)  # (b * h, m, 1, d_k)
        keys_split = self._split_heads(keys)  # (b * h, m, l, d_k)
        values_split = self._split_heads(values)

        q_mask_split = self._split_heads(queries_mask) # (b * h, m, 1, d_k)
        k_mask_split = self._split_heads(keys_mask) # (b * h, m, l, d_k)

        attention_repr, masks, masked_probs, attention_probs = self._attention(
            queries_split,
            keys_split,
            values_split,
            q_mask_split, 
            k_mask_split
        )

        attention_repr = self._merge_heads(attention_repr)  # (b, m, 1, d)
        attention_repr = self.projection(attention_repr).squeeze(dim=2)  # (b, m, d)

        if sum_heads:
            masks = masks.sum(dim=1) # (b, m, 1, l)
            masked_probs = masked_probs.sum(dim=1)
            attention_probs = attention_probs.sum(dim=1)

        if self.agg_residual:
            pooled = x.max(dim=2)[0]  # (b, m, d)
            attention_repr = attention_repr + self.agg_res_coeff * pooled

        if self.layernorm:
            attention_repr = self.ln(attention_repr)

        out = self.mlp(attention_repr, context)  # (b, m, out_dim)

        return out, masks.squeeze(-2), masked_probs.squeeze(-2), attention_probs.squeeze(-2)

    def _split_heads(self: Self, x: Tensor):
        # (b, m, l, d) -> (b * h, m, l, d_k), keeping b*h as the leading dim
        batch_size, num_modes, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, num_modes, seq_len, self.heads, self.dk)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch_size * self.heads, num_modes, seq_len, self.dk)
        )

    def _merge_heads(self: Self, x: Tensor):
        # (b, h, m, l, d_k) -> (b, m, l, d)
        batch_size, _, num_modes, seq_len, _ = x.size()
        return (
            x.permute(0, 2, 3, 1, 4)
            .reshape(batch_size, num_modes, seq_len, self.dk * self.heads)
        )

    def _attention(
        self: Self,
        query: Tensor, # (b*h, m, 1, d_k)
        key: Tensor, # (b*h, m, l, d_k)
        value: Tensor, # (b*h, m, l, d_k)
        query_mask: Tensor, # (b*h, m, 1, d_k)
        keys_mask: Tensor, # (b*h, m, l, d_k)
    ):
        batch_heads, num_modes, seq_len, _ = key.size()

        # (b*h, m, 1, d_k) @ (b*h, m, d_k, l) -> (b*h, m, 1, l)
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.dk)

        attention_probs = softmax(attention_logits, dim=-1)
        # (b*h, m, 1, d_k) @ (b*h, m, d_k, l) -> (b*h, m, 1, l)
        mask_logits = torch.matmul(query_mask, keys_mask.transpose(-2, -1)) / np.sqrt(self.dk)
        mask_logits = mask_logits.reshape(-1, seq_len) # (b*h*m, l)
        edges_logit = torch.stack(
            [torch.zeros_like(mask_logits), mask_logits + self.bias], dim=-1
        )
        A = gumbel_softmax(edges_logit, tau=self.temp, hard=self.hard) # (b*h*m, l, 2)
        A = A[:, :, -1].view(batch_heads, num_modes, 1, seq_len)

        masked_attention_probs = A * attention_probs # (b*h, m, 1, l)

        # (b*h, m, 1, l) @ (b*h, m, l, d_k) -> (b*h, m, 1, d_k)
        hidden_repr = torch.matmul(masked_attention_probs, value)

        return (
            hidden_repr.view(-1, self.heads, num_modes, 1, self.dk),
            A.view(-1, self.heads, num_modes, 1, seq_len),
            masked_attention_probs.view(-1, self.heads, num_modes, 1, seq_len),
            attention_probs.view(-1, self.heads, num_modes, 1, seq_len),
        )
