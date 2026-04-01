import torch 

from torch import Tensor
from torch.utils.data import Dataset
from typing import Self
from torch.nn.functional import one_hot

class BasicDataset(Dataset):
    def __init__(self: Self, x: Tensor, y: Tensor):
        super().__init__()
        self.x = x.float()
        self.y = y.float()
        
        
        if len(self.y.shape) < 2:
            self.y = self.y.unsqueeze(dim=-1)
    
    def __len__(self: Self):
        return self.x.size(0)
    
    def __getitem__(self: Self, index: int):
        return self.x[index], self.y[index]
class ShapesDataset(Dataset):
    def __init__(self: Self, x: Tensor, y: Tensor, one_hot: bool = False, size: int = 5):
        super().__init__()
        self.x = x.float()
        self.y = y.float()
        self.one_hot = one_hot
        self.size = size
        
        if len(self.y.shape) < 2:
            self.y = self.y.unsqueeze(dim=-1)
    
    def __len__(self: Self):
        return self.x.size(0)
    
    def __getitem__(self: Self, index: int):
        if self.one_hot:
            x = one_hot(self.x[index].squeeze().long(), self.size**2).float()
        else:
            x = self.x[index]
        return x, self.y[index]

class OneHotBoxWorld(Dataset):
    def __init__(self: Self, x: Tensor, y: Tensor):
        super().__init__()
        self.x = x.long()
        self.y = y.float()

        if len(self.y.shape) < 2:
            self.y = self.y.unsqueeze(dim=-1)

    def __len__(self: Self):
        return self.x.size(0)
    
    def __getitem__(self: Self, index: int):
        x = self.x[index] # (10, 10, 3)
        objects = one_hot(x[:, :, 0], 14)
        colors = one_hot(x[:, :, 1], 60)
        return torch.cat([objects, colors], dim=-1).float(), self.y[index]


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
    

