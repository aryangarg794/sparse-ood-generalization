import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.functional import gumbel_softmax, softmax
from typing import Self


class MaskNet(nn.Module):
    """Hypernetwork m(x): token features -> one (heads, L, L) attention mask per layer.

    The flattened input tokens (b, L * d), positions included via the PE they carry,
    are mapped by an MLP to mask logits for every attention layer and, optionally, an
    (heads, L) mask for an aggregation layer. Edges are binarised with hard
    straight-through Gumbel-softmax in training (P(edge) = sigmoid(logit + bias), same
    construction as FiLMAttention) and thresholded deterministically in eval.
    """

    def __init__(
        self,
        seq_len: int,
        embed_size: int,
        num_layers: int = 1,
        num_heads: int = 1,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        temp: float = 1.0,
        hard: bool = True,
        bias: float = 0.5,
        agg_layer: bool = False,
        act: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.temp = temp
        self.hard = hard
        self.bias = bias
        self.agg_layer = agg_layer

        self.layer_size = num_layers * num_heads * seq_len * seq_len
        self.agg_size = num_heads * seq_len if agg_layer else 0

        layers = [nn.Linear(seq_len * embed_size, hidden_dim), act()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, self.layer_size + self.agg_size))
        self.net = nn.Sequential(*layers)

    def logits(self, x: Tensor):
        """x: (b, L, d) -> layer logits (b, num_layers, h, L, L), agg logits (b, h, L) | None"""
        out = self.net(x.flatten(1))
        layer_logits = out[:, : self.layer_size].view(
            -1, self.num_layers, self.num_heads, self.seq_len, self.seq_len
        )
        agg_logits = None
        if self.agg_layer:
            agg_logits = out[:, self.layer_size :].view(-1, self.num_heads, self.seq_len)
        return layer_logits, agg_logits

    def _binarise(self, logits: Tensor, deterministic: bool) -> Tensor:
        logits = logits + self.bias
        if deterministic:
            return (logits > 0).to(logits.dtype)
        two_class = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return gumbel_softmax(two_class, tau=self.temp, hard=self.hard)[..., 1]

    def forward(self, x: Tensor, deterministic: bool = None):
        """Returns (layer_masks, agg_mask, layer_logits, agg_logits).

        layer_masks: (b, num_layers, h, L, L); agg_mask: (b, h, L) or None.
        `deterministic` defaults to `not self.training`.
        """
        if deterministic is None:
            deterministic = not self.training
        layer_logits, agg_logits = self.logits(x)
        layer_masks = self._binarise(layer_logits, deterministic)
        agg_mask = self._binarise(agg_logits, deterministic) if agg_logits is not None else None
        return layer_masks, agg_mask, layer_logits, agg_logits

    def edge_probs(self, x: Tensor):
        """Per-edge on-probabilities for x (no binarisation), useful for logging/inspection."""
        layer_logits, agg_logits = self.logits(x)
        layer_p = torch.sigmoid(layer_logits + self.bias)
        agg_p = torch.sigmoid(agg_logits + self.bias) if agg_logits is not None else None
        return layer_p, agg_p


class HyperMaskAttention(nn.Module):
    """Multi-head attention gated by an externally supplied (b, m, h, L, L) mask.

    Returns the same tuple layout as FiLMAttention so the model bookkeeping is shared:
    (repr (b, m, l, d), mask (b, [h,] m, l, l), masked probs, probs).
    """

    def __init__(
        self: Self,
        embed_size: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"embed_size {embed_size} not divisible by num_heads {num_heads}")
        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

    def _split_heads(self: Self, x: Tensor):
        # (b, m, l, d) -> (b, h, m, l, d_k)
        b, m, l, _ = x.size()
        return x.view(b, m, l, self.heads, self.dk).permute(0, 3, 1, 2, 4)

    def forward(
        self: Self,
        x: Tensor,  # (b, m, l, d)
        mask: Tensor,  # (b, m, h, l, l) in {0, 1}
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        b, m, l, _ = x.size()
        q = self._split_heads(self.queries(x))  # (b, h, m, l, d_k)
        k = self._split_heads(self.keys(x))
        v = self._split_heads(self.values(x))

        logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dk)  # (b, h, m, l, l)
        probs = softmax(logits, dim=-1)

        A = mask.transpose(1, 2)  # (b, h, m, l, l)
        masked_probs = A * probs
        hidden = torch.matmul(self.dropout(masked_probs), v)  # (b, h, m, l, d_k)
        repr = hidden.permute(0, 2, 3, 1, 4).reshape(b, m, l, self.embed_size)
        repr = self.projection(repr)

        if self.residual:
            eye = torch.eye(l, device=A.device, dtype=A.dtype).view(1, 1, 1, l, l)
            A = A + eye

        if avg_attn_heads:
            masked_probs = masked_probs.sum(dim=1)  # (b, m, l, l)
            probs = probs.sum(dim=1)
        if avg_mask:
            A = A.sum(dim=1)  # (b, m, l, l)

        return repr, A, masked_probs, probs


class HyperMaskAggAttention(nn.Module):
    """Learned-query aggregation over tokens, gated by a (b, m, h, L) mask, then linear."""

    def __init__(self: Self, embed_size: int, out_dim: int, num_heads: int = 1, residual: bool = False):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"embed_size {embed_size} not divisible by num_heads {num_heads}")
        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.residual = residual
        self.query = nn.Parameter(torch.randn(1, embed_size) / np.sqrt(embed_size))
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, out_dim)

    def forward(self: Self, x: Tensor, mask: Tensor):
        # x: (b, m, l, d), mask: (b, m, h, l) -> out (b, m, o), mask (b, m, l), masked probs, probs (b, m, l)
        b, m, l, _ = x.size()
        q = self.query.view(1, self.heads, 1, 1, self.dk)  # (1, h, 1, 1, d_k)
        k = self.keys(x).view(b, m, l, self.heads, self.dk).permute(0, 3, 1, 2, 4)  # (b, h, m, l, d_k)
        v = self.values(x).view(b, m, l, self.heads, self.dk).permute(0, 3, 1, 2, 4)
        logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dk)  # (b, h, m, 1, l)
        probs = softmax(logits, dim=-1)
        A = mask.transpose(1, 2).unsqueeze(3)  # (b, h, m, 1, l)
        masked_probs = A * probs
        hidden = torch.matmul(masked_probs, v).squeeze(3)  # (b, h, m, d_k)
        pooled = self.projection(hidden.permute(0, 2, 1, 3).reshape(b, m, self.embed_size))
        out = self.out(pooled)  # (b, m, o)
        A = A.sum(dim=1).squeeze(2)  # (b, m, l)
        return out, A, masked_probs.sum(dim=1).squeeze(2), probs.sum(dim=1).squeeze(2)
