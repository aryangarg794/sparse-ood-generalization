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
        # nn.init.trunc_normal_(self.query, std=0.02)

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

        # if self.residual:
        #     pooled_x = x.max(dim=1, keepdim=True)[0]
        #     out = pooled_x + attention_repr

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
                    [torch.zeros_like(edges_logit), edges_logit + self.bias], dim=-1
                )
                A = gumbel_softmax(edges_logit, tau=self.temp, hard=True)
                A = A[:, :, -1].reshape(batch_heads, 1, seq_len)
            
            attention_probs = A * attention_probs

        # (bh, 1, l)
        hidden_repr = torch.bmm(attention_probs, value)  # (bh, 1, l) @ (bh, l, dk)

        return hidden_repr.view(-1, self.heads, 1, self.dk), A.view(-1, self.heads, 1, seq_len), attention_probs.view(
            -1, self.heads, 1, seq_len
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
        flow_params: dict = {'n_flows' : 2, 'hidden_features' : (128, 128)},
        prior_params: dict = {'n_flows' : 3, 'hidden_features' : (256, 256)},
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
            bidirectional=bidirectional
        )
        self.encoder = nn.Linear(latent_dim, 2 * latent_dim)

        self.v = nn.Parameter(torch.randn(1, seq_len))
        nn.init.xavier_uniform_(self.v)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.nf_prior = nf_prior
        if self.nf_prior:
            self.prior = zuko.flows.MAF(
                features=latent_dim,
                transforms=prior_params['n_flows'],
                hidden_features=prior_params['hidden_features'],
            )

        base_flow = zuko.flows.NSF(
            features=latent_dim,
            transforms=flow_params['n_flows'],
            hidden_features=flow_params['hidden_features']
        )
        self.normalizing_flow = Flow(base_flow.transform.inv, base_flow.base)

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

        out, _ = self.encoder_lstm(x) # assume self-attn (b, k)
        mu, sig = torch.chunk(self.encoder(out[:, -1, :]), chunks=2, dim=-1) 
        encoding = reparametrize(mu, sig)

        attention_repr, masks, attention_probs, ladj, prior = self._attention(
            queries_split,
            keys_split,
            values_split,
            encoding
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

        return hidden_repr.view(-1, self.heads, 1, self.dk), A.view(-1, self.heads, 1, seq_len), attention_probs.view(
            -1, self.heads, 1, seq_len
        ), ladj if self.training else None, prior if self.training else None