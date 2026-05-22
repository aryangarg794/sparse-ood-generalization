import torch
import torch.nn as nn
import zuko
import numpy as np
import math
import torch.nn.functional as F

from zuko.flows import Flow
from torch import Tensor
from torch.nn.functional import softmax
from typing import Self

from sparse_generalization.utils.util_funcs import vae_log_prob, reparametrize

class AttentionFlowLatent(nn.Module):

    def __init__(
            self,  
            embed_size: int,
            seq_len: int, 
            latent_dim: int = 16, 
            num_heads: int = 1,
            lstm_layers: int = 1, 
            dropout: float = 0.0, 
            bidirectional: bool = True, 
            flow_params: dict = {'n_flows' : 2, 'hidden_features' : (128, 128)},
            prior_params: dict = {'n_flows' : 3, 'hidden_features' : (256, 256)},
            residual: bool = False,
            nf_prior: bool = True, 
            *args, 
            **kwargs
        ):
        
        super(AttentionFlowLatent, self).__init__(*args, **kwargs)

        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )
        
        assert num_heads == 1, "Temporarily can only be 1 head used"

        self.dk = embed_size // num_heads
        self.heads = num_heads # NOTE: has to 1 right now since idk how to do multiheaded 
        self.embed_size = embed_size
        self.residual = residual  

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        # vae encoder-decoder
        self.encoder_lstm = nn.LSTM(
            self.dk, 
            latent_dim // 2 if bidirectional else latent_dim, 
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.encoder = nn.Linear(latent_dim, 2 * latent_dim)

        self.v = nn.Parameter(torch.randn(seq_len, seq_len))
        nn.init.xavier_uniform_(self.v)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len)
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

    def forward(
        self: Self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        avg_attn_heads: bool = True,
        avg_mask: bool = True,
    ):
        x = queries.clone()
        queries = self.queries(queries)  
        keys = self.keys(keys)
        values = self.values(values)

        queries_split = self._split_heads(queries)  # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values) 

        # encoder
        out, _ = self.encoder_lstm(x) # assume self-attn (b, k)
        mu, sig = torch.chunk(self.encoder(out[:, -1, :]), chunks=2, dim=-1) 
        encoding = reparametrize(mu, sig)

        attention_repr, mask_per_head, attn_per_head, ladj, prior = self._attention(
            queries_split,
            keys_split,
            values_split,
            encoding
        )

        attention_repr = self._merge_heads(attention_repr)  
        attention_repr = self.projection(attention_repr)

        if avg_attn_heads:
            adjacency = attn_per_head.sum(dim=1)

        if avg_mask:
            mask = mask_per_head.sum(dim=1)

        if self.training:
            mha_loss = prior - vae_log_prob(encoding, mu, sig) + ladj
            return attention_repr, mask, adjacency, mha_loss
        else:
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
        self: Self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        encoding: Tensor
    ):
        batch_heads, seq_len, _ = query.size()
        attention_logits = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(
            self.dk
        )  # (b*h, l, l)

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
        mask_weights_raw = g.view(-1, seq_len, 1) * v_dir
        mask_weights = F.sigmoid(mask_weights_raw)
        hard_mask = (mask_weights >= 0.5).float()
        A = hard_mask - mask_weights.detach() + mask_weights

        masked_attention_probs = A * attention_probs
        hidden_repr = torch.bmm(masked_attention_probs, value)
        if self.residual and not self.mask_res:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0).repeat(
                batch_heads, 1, 1
            )

        return (
            hidden_repr.view(-1, self.heads, seq_len, self.dk),
            A.view(-1, self.heads, seq_len, seq_len),
            masked_attention_probs.view(-1, self.heads, seq_len, seq_len),
            ladj if self.training else None,
            prior if self.training else None
        )
    
