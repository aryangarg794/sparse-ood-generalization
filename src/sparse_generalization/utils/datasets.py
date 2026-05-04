import torch

from torch import Tensor
from torch.utils.data import Dataset
from typing import Self
from torch.nn.functional import one_hot

from sparse_generalization.data.shapes.constants import SHAPES_TO_IDX
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
    def __init__(
        self: Self, x: Tensor, y: Tensor, one_hot: bool = False, size: int = 5, masks: Tensor = None, 
    ):
        super().__init__()
        self.x = x.float()
        self.y = y.float()
        self.one_hot = one_hot
        self.size = size
        if masks is not None:
            self.mask = True
            self.masks = masks
        else:
            self.mask = False

        if len(self.y.shape) < 2:
            self.y = self.y.unsqueeze(dim=-1)

    def __len__(self: Self):
        return self.x.size(0)

    def __getitem__(self: Self, index: int):
        if self.one_hot:
            x = one_hot(self.x[index].squeeze().long(), self.size**2).float()
        else:
            x = self.x[index]
        
        if self.mask:
            return x, self.y[index], self.masks[index] 
        else:
            return x, self.y[index]

    def get_flat_idx(self, x, y, height):
        return (x * height) + y
    
    def get_mask_idxs(self, x):
        star_idx = SHAPES_TO_IDX["star"]
        heart_idx = SHAPES_TO_IDX["heart"]
        circle_idx = SHAPES_TO_IDX["circle"]
        square_idx = SHAPES_TO_IDX["square"]
        coords_star = torch.where(x == star_idx)
        coords_heart = torch.where(x == heart_idx)
        coords_circle = torch.where(x == circle_idx)
        coords_square = torch.where(x == square_idx)

        # heart_idx = self.get_flat_idx(coords_heart[0], coords_heart[1], x.size(0))
        # star_idx = self.get_flat_idx(coords_star[0], coords_star[1], x.size(0))
        # circle_idx = self.get_flat_idx(coords_circle[0], coords_circle[1], x.size(0))
        # square_idx = self.get_flat_idx(coords_square[0], coords_square[1], x.size(0))
        return coords_star, coords_heart, coords_circle, coords_square

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
        x = self.x[index]  # (10, 10, 3)
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
