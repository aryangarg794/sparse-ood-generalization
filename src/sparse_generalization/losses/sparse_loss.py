import torch 
import torch.nn as nn

from typing import Self

class L1Sparsity(nn.Module):
    """L1 sparsity loss for attention matrices, forcing the bottom k 
    weights to be 0 using L1 loss. 

    Args:
        
    """
    
    def __init__(
        self: Self, 
        k: int, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        