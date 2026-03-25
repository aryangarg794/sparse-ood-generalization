import torch 

from torch import Tensor
from torch.utils.data import Dataset
from typing import Self

class BasicDataset(Dataset):
    def __init__(self: Self, x: Tensor, y: Tensor, one_hot: bool):
        super().__init__()
        self.x = x.float()
        self.y = y.float()
        self.one_hot = one_hot
        
        
        if len(self.y.shape) < 2:
            self.y = self.y.unsqueeze(dim=-1)
    
    def __len__(self: Self):
        return self.x.size(0)
    
    def __getitem__(self: Self, index: int):
        if self.one_hot:
            x = torch.nn.functional.one_hot(self.x[index].squeeze().long(), 64).float()
        else:
            x = self.x[index]
        return x, self.y[index]
    
class EdgeDataset(Dataset):
    def __init__(self: Self, x: Tensor, y: Tensor, edges: list):
        super().__init__()
        self.x = x.float()
        self.y = y.float()
        self.edges = edges
        
        if len(self.y.shape) < 2:
            self.y = self.y.unsqueeze(dim=-1)
    
    def __len__(self: Self):
        return self.x.size(0)
    
    def __getitem__(self: Self, index: int):
        return self.x[index], self.y[index], self.edges[index]