import lightning as pl
import torch 
import torch.nn as nn

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self

class BasicCNN(nn.Module):
    """Basic CNN class, based

    Args:
        input_dim (int): Size of Input.
        out_dim (int): Size of output.
        hidden_dims (List): Model architecture. 
        act (nn.Module): Activation function.
    """
    
    def __init__(
        self: Self, 
        input_dim: int, 
        out_dim: int,
        hidden_channels: List,
        act: nn.Module,
        dropout: float, 
        *args, 
        **kwargs
    ):
        super(BasicCNN, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential()
        self.layers.extend([
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_channels[0], kernel_size=3, padding=1), 
            act(),
        ])
        
        for ch1, ch2 in zip(hidden_channels[:-1], hidden_channels[1:]):
            self.layers.extend([
                nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=3, padding=1), 
                act(),
            ])
            
        self.layers.extend([
            nn.MaxPool2d(kernel_size=3, stride=1), 
            act(),
        ])
        
        self.ffn = nn.Sequential()
        self.ffn.extend([
            nn.Linear(in_features=8*8*hidden_channels[-1], out_features=100),
            act(),
            nn.Dropout(dropout), 
            nn.Linear(in_features=100, out_features=64),
            act(),
            nn.Linear(in_features=64, out_features=out_dim),
        ])
        
    def forward(self: Self, x: Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        x = self.ffn(x)
        return x
