import torch 
import torch.nn as nn
import numpy as np


from torch import Tensor
from torch.nn.functional import gumbel_softmax, softmax
from typing import Self, Callable

class FiLMLayer(nn.Module):

    def __init__(
        self, 
        inp_dim: int, 
        context_dim: int, 
        num_layers: int = 2, 
        act: nn.Module = nn.ReLU, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if num_layers == 1:
            self.film_mlp = nn.Linear(context_dim, 2*inp_dim)
        else:
            hidden_dim = context_dim
            layers = [nn.Linear(context_dim, hidden_dim), act()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), act()]
            layers.append(nn.Linear(hidden_dim, 2*inp_dim))
            self.film_mlp = nn.Sequential(*layers)

        self.projection = nn.Linear(inp_dim, inp_dim)

    def forward(self, x: Tensor, c: Tensor, per_mode: bool = False):
        mu, delta = self.film_mlp(c).chunk(2, dim=-1)   # (m, d) each
        sig = 1.0 + delta
        out = self.projection(x) # (b, l, d) or (b, m, d)
        if not per_mode:
            out = out.unsqueeze(1)                     # (b, 1, *, d)
        shape = (1, mu.size(0)) + (1,) * (out.dim() - 3) + (mu.size(-1),)  # (1, m, 1..., d)
        return sig.view(shape) * out + mu.view(shape)  # (b, m, *, d)


class FiLMMLP(nn.Module):

    def __init__(
        self,
        inp_dim: int,
        context_dim: int,
        out_dim: int,
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        num_layers_film: int = 2,
    ):
        super().__init__()
        self.first_mlp = nn.Sequential(
            nn.Linear(inp_dim, 4 * inp_dim),
            nn.Dropout(dropout),
        )
        self.film_layer = FiLMLayer(4 * inp_dim, context_dim, num_layers_film, act)
        self.second_mlp = nn.Sequential(
            act(),
            nn.Linear(4 * inp_dim, out_dim),
        )

    def forward(self, x: Tensor, context: Tensor):
        # x: (b, m, *, d), context: (m, context_dim)
        out = self.first_mlp(x)
        out = self.film_layer(out, context, per_mode=True)
        return self.second_mlp(out)


class FiLMHead(nn.Module):
    
    def __init__(
        self,
        embed_size: int,
        context_dim: int,
        out_dim: int,
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        num_layers_film: int = 2,
        layernorm: bool = False,
        pool: str = "mean",  # 'mean' | 'max' | 'concat'
        seq_len: int = None,  # required for 'concat'
    ):
        super().__init__()
        if pool not in ("mean", "max", "concat"):
            raise ValueError(f"pool must be 'mean', 'max' or 'concat', got {pool!r}")
        if pool == "concat" and seq_len is None:
            raise ValueError("pool='concat' needs seq_len")
        self.pool = pool
        head_dim = embed_size * seq_len if pool == "concat" else embed_size
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(head_dim)
        self.film = FiLMLayer(head_dim, context_dim, num_layers_film, act)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(head_dim, out_dim)

    def forward(self, x: Tensor, context: Tensor):
        if self.pool == "mean":
            pooled = x.mean(dim=2)  # (b, m, d)
        elif self.pool == "max":
            pooled = x.max(dim=2)[0]  # (b, m, d)
        else:
            pooled = x.flatten(2)  # (b, m, l * d)
        if self.layernorm:
            pooled = self.ln(pooled)
        pooled = self.film(pooled, context, per_mode=True)  # (b, m, d)
        return self.out(self.dropout(pooled))  # (b, m, out_dim)


class FiLMAttention(nn.Module):
    def __init__(
        self: Self,
        embed_size: int,
        context_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        temp: float = 1.0,
        hard: bool = True,
        act: nn.Module = nn.ReLU, 
        num_layers_film: int = 2, 
        residual: bool = False,
        film_values: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )

        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        self.temp = temp
        self.hard = hard
        self.bias = 0.5
        self.residual = residual

        self.queries_mask = FiLMLayer(embed_size, context_dim, num_layers_film, act)
        self.keys_mask = FiLMLayer(embed_size, context_dim, num_layers_film, act)

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        # with film_values the modes get their own value space (FiLMLayer already
        # includes the linear projection); otherwise values are shared across modes
        self.film_values = film_values
        if film_values:
            self.values = FiLMLayer(embed_size, context_dim, num_layers_film, act)
        else:
            self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(
        self: Self,
        queries: Tensor, # (b, m, l, d) 
        keys: Tensor,
        values: Tensor,
        context: Tensor, # (m, context_dim)
        avg_attn_heads: bool = True,
        avg_mask: bool = True
    ):
        queries = self.queries(queries)  # (b, m, l, d)
        keys = self.keys(keys)
        if self.film_values:
            values = self.values(values, context, per_mode=True)  # (b, m, l, d)
        else:
            values = self.values(values)
        queries_mask = self.queries_mask(queries, context, per_mode=True) # (b, m, l, d)
        keys_mask = self.keys_mask(keys, context, per_mode=True)

        queries_split = self._split_heads(queries)  # (b * h, m, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        q_mask_split = self._split_heads(queries_mask) # (b * h, m, l, d_k)
        k_mask_split = self._split_heads(keys_mask)

        attention_repr, mask_per_head, mask_attn_per_head, attn_per_head = (
            self._attention(
                queries_split,
                keys_split,
                values_split,
                q_mask_split, 
                k_mask_split
            )
        )

        attention_repr = self._merge_heads(attention_repr)  # (b, m, l, d)
        attention_repr = self.projection(attention_repr)

        mask = mask_per_head # (b, h, m, l, l)
        adjacency = attn_per_head # (b, h, m, l, l)

        if avg_attn_heads:
            mask_attn_per_head = mask_attn_per_head.sum(dim=1) # (b, m, l, l)
            adjacency = attn_per_head.sum(dim=1) # (b, m, l, l)

        if avg_mask:
            mask = mask_per_head.sum(dim=1) # (b, m, l, l)

        return attention_repr, mask, mask_attn_per_head, adjacency

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
        query: Tensor, # (b*h, m, l, d_k)
        key: Tensor,
        value: Tensor,
        query_mask: Tensor, # (b*h, m, l, d_k)
        keys_mask: Tensor, 
    ):
        batch_heads, num_modes, seq_len, _ = query.size()

        # (b*h, m, l, d_k) @ (b*h, m, d_k, l) -> (b*h, m, l, l)
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(
            self.dk
        )

        attention_probs = softmax(attention_logits, dim=-1)

        mask_logits = torch.matmul(query_mask, keys_mask.transpose(-2, -1)) / np.sqrt(
            self.dk
        ) # (b*h, m, l, l)
        mask_logits = mask_logits.reshape(-1, seq_len ** 2) # (b*h*m, l*l)
        # mask_logits = mask_logits.clamp(min=-5)
        edges_logit = torch.stack(
            [torch.zeros_like(mask_logits), mask_logits + self.bias], dim=-1
        )
        A = gumbel_softmax(
            edges_logit, tau=self.temp, hard=self.hard
        )  # (b*h*m, l*l, 2)
        A = A[:, :, -1]

        A = A.view(batch_heads, num_modes, seq_len, seq_len)

        masked_attention_probs = A * attention_probs # (b*h, m, l, l)

        # (b*h, m, l, l) @ (b*h, m, l, d_k) -> (b*h, m, l, d_k)
        hidden_repr = torch.matmul(masked_attention_probs, value)

        if self.residual:
            eye = torch.eye(seq_len, device=A.device, dtype=A.dtype).view(1, 1, seq_len, seq_len)
            A = A + eye

        return (
            hidden_repr.view(-1, self.heads, num_modes, seq_len, self.dk),
            A.view(-1, self.heads, num_modes, seq_len, seq_len),
            masked_attention_probs.view(-1, self.heads, num_modes, seq_len, seq_len),
            attention_probs.view(-1, self.heads, num_modes, seq_len, seq_len),
        )
