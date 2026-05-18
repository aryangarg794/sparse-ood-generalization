import torch
import torch.nn as nn
import zuko

from zuko.flows import LazyInverse

class GenMHA(nn.Module):

    def __init__(
            self, 
            embed_size: int,
            num_heads: int,
            residual: bool = False,
            *args, 
            **kwargs
        ):
        super().__init__(
            *args, 
            **kwargs
        )