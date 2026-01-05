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
            nn.Linear(in_features=6*6*hidden_channels[-1], out_features=100),
            act(),
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
    
class BasicCNNLit(pl.LightningModule):
    """Lighting Model for the basic CNN

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
        hidden_channels: List = list([16, 32, 64]),
        act: nn.Module = nn.ReLU,
        lr: float = 1e-3,
        loss: nn.Module = nn.BCEWithLogitsLoss
    ):
        super().__init__()
        self.loss = loss()
        self.lr = lr
        
        self.model = BasicCNN(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_channels=hidden_channels,
            act=act
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
