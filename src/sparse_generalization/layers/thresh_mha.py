import numpy as np
import torch 
import torch.nn as nn

from torch import Tensor
from torch.nn.functional import gumbel_softmax, softmax
from typing import Self

class MultiHeadAttentionThresh(nn.Module):
    """Implements MHA with thresholding, all attention values lower than 

    Args:
        nn (_type_): _description_
    """
    
    def __init__(
        self: Self, 
        embed_size: int, 
        num_heads: int, 
        bias: int = True, 
        dropout: float = 0.0, 
        batch_first: bool = True, 
        threshold: float = 0.1,
        *args, 
        **kwargs
    ):
        super(MultiHeadAttentionThresh, self).__init__(*args, **kwargs)
        
        if embed_size % num_heads != 0:
            raise SyntaxError(f'Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}')
        
        self.dk = embed_size // num_heads
        self.heads = num_heads
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        self.threshold = threshold
        
        self.queries = nn.Linear(embed_size, embed_size, bias=bias)
        self.keys = nn.Linear(embed_size, embed_size, bias=bias)
        self.values = nn.Linear(embed_size, embed_size, bias=bias)
        self.projection = nn.Linear(embed_size, embed_size, bias=bias)
    
        
    def forward(self: Self, queries: Tensor, keys: Tensor, values: Tensor, avg_attn_heads: bool = True): 
        queries = self.queries(queries) # (b, l, d)
        keys = self.keys(keys)
        values = self.values(values)
        
        queries_split = self._split_heads(queries) # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)
        
        attention_repr, adjacency_per_head = self._attention(queries_split,
                                                             keys_split, 
                                                             values_split)

        attention_repr = self._merge_heads(attention_repr) # (b, l, d)
        attention_repr = self.projection(attention_repr)
        
        if avg_attn_heads:
            adjacency_per_head = adjacency_per_head.mean(dim=1)
        
        return attention_repr, adjacency_per_head
    
    def _split_heads(self: Self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        return x.reshape(batch_size, seq_len, self.heads, self.dk).transpose(1, 2).reshape(
            batch_size * self.heads, seq_len, self.dk)
        
    def _merge_heads(self: Self, x: Tensor):
        batch_size, _, seq_len, _ = x.size()
        return x.reshape(batch_size, self.heads, seq_len, self.dk).transpose(1, 2).reshape(
            batch_size, seq_len, self.dk * self.heads) 
    
    def _attention(self: Self, query: Tensor, key: Tensor, value: Tensor, return_mask: bool = False):
        batch_heads, seq_len, _ = query.size()
        attention_logits = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.dk) # (b*h, l, l)
        
        attention_probs = softmax(attention_logits, dim=-1)
        attention_probs_mask = (attention_probs > self.threshold).float()
        thresholded_probs = attention_probs_mask * attention_probs 
        thresholded_probs = thresholded_probs - attention_probs.detach() + attention_probs # ste 
        
        hidden_repr = torch.bmm(thresholded_probs, value) # (b*h, l, d)    
        
        if return_mask:
            return hidden_repr.view(-1, self.heads, seq_len, self.dk), attention_probs.view(-1, self.heads, seq_len, seq_len)
        else:
            return hidden_repr.view(-1, self.heads, seq_len, self.dk), attention_probs.view(-1, self.heads, seq_len, seq_len)
        
        
          