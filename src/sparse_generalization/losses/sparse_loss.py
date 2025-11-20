import torch 
import torch.nn as nn

from torch import Tensor
from typing import Self

class L1Sparsity(nn.Module):
    """L1 sparsity loss for attention matrices, forcing the bottom k 
    weights to be 0 using L1 loss. 

    Args:
        
    """
    
    def __init__(
        self: Self, 
        k: int = 5, # the lowest-k that we want to sparsify
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        
    def forward(self: Self, weights: Tensor):
        batch_size = weights.size(0)
        low_vals, _ = torch.sort(weights.view(weights.size(0), -1), dim=1) 
        low_vals = low_vals[:, :self.k]
        l1_loss = low_vals.sum(dim=-1).sum() / batch_size
        return l1_loss
    
    
class L1SparsityAdjacency(nn.Module):
    """L1 sparsity loss for attention matrices, forcing attention adjaceny to be low: 
    https://arxiv.org/pdf/2411.06890

    Args:
        
    """
    
    def __init__(
        self: Self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
    def forward(self: Self, A: Tensor):
        batch_size, _, _ = A.size()
        return A.sum(dim=(1, 2)).mean() 