import lightning as pl
import torch 
import torch.nn as nn

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self

from sparse_generalization.models.cnn import BasicCNN

class BasicMLP(nn.Module):
    """Basic MLP class

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
        hidden_dims: List,
        act: nn.Module,
        dropout: float, 
        *args, 
        **kwargs
    ):
        super(BasicMLP, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential()
        self.layers.extend([nn.Linear(input_dim, hidden_dims[-1]), act()])
        
        for dim1, dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(dim1, dim2), act(), nn.Dropout(dropout)])
        self.layers.append(nn.Linear(hidden_dims[-1], out_dim))
        
    def forward(self: Self, x: Tensor):
        return self.layers(x)
    
class BasicMLPLit(pl.LightningModule):
    """Lighting Model for the basic MLP

    Args:
        input_dim (int): Size of Input.
        out_dim (int): Size of output.
        hidden_dims (List): Model architecture. Default [64, 128, 64].
        act (nn.Module): Activation function. Default is ReLU.
        lr: (float): Learning rate for Adam.
        loss: (nn.Module): Loss function for optimization.
    """
    def __init__(
        self: Self, 
        input_dim: int, 
        out_dim: int,
        hidden_dims: List = list([64, 128, 64]),
        act: nn.Module = nn.ReLU,
        lr: float = 1e-3,
        dropout: float = 0.1, 
        loss: nn.Module = nn.BCEWithLogitsLoss,
        module: nn.Module = BasicMLP
    ):
        super().__init__()
        self.loss = loss()
        self.lr = lr
        
        self.model = module(
            input_dim,
            out_dim,
            hidden_dims,
            act,
            dropout
        )
        
        self.accuracy = BinaryAccuracy()
        
    def _get_loss_acc(self: Self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._get_loss_acc(batch)
        self.log(
            "train_loss", 
            loss, 
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_acc", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def validation_step(self, batch):
        loss, acc = self._get_loss_acc(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_acc", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def test_step(self, batch):
        loss, acc = self._get_loss_acc(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_acc", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
