import torch 

from torch import Tensor
from torch.utils.data import Dataset
from typing import Self

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