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
        self.layers.extend([nn.Linear(input_dim, hidden_dims[0]), act()])
        
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
        wd: float = 1e-3, 
        val_to_name: dict = {
            0: 'id', 
            1: 'col', 
            2: 'pair', 
            3: 'dist', 
            4: 'comb'
        }, 
        embedding_inp: bool = True, 
        num_embeddings: int = 64, 
        model_dim: int = 32, 
        dropout: float = 0.1, 
        loss: nn.Module = nn.BCEWithLogitsLoss,
        module: nn.Module = BasicMLP,
        inp_size: int = 10, 
    ):
        super().__init__()
        self.loss = loss()
        self.lr = lr
        self.wd = wd
        
        self.model = module(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            act=act,
            dropout=dropout, 
            num_embeddings=num_embeddings,
            embedding_inp=embedding_inp,
            model_dim=model_dim,
            inp_size=inp_size
        )
        
        self.accuracy = BinaryAccuracy()
        
        self.losses = []
        self.accs = []
        self.running_loss = 0.0
        self.running_acc = 0.0
        self.val_to_name = val_to_name

        self.running_loss_test = {i:0.0 for i in val_to_name.keys()}
        self.running_acc_test = {i:0.0 for i in val_to_name.keys()}
        
        self.losses_test = {i:[] for i in val_to_name.values()}
        self.accs_test = {i:[] for i in val_to_name.values()}
        
        
    def _get_loss_acc(self: Self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._get_loss_acc(batch)
        self.log(
            "train/loss", 
            loss, 
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/acc", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.running_loss += loss.item()
        self.running_acc += acc.item()
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, acc = self._get_loss_acc(batch)
        name = self.val_to_name[dataloader_idx]
        self.log(
            f"val/loss_{name}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False
        )
        self.log(
            f"val/acc_{name}", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False
        )
        
        self.running_loss_test[dataloader_idx] += loss.item()
        self.running_acc_test[dataloader_idx] += acc.item()
        
        return loss

    def test_step(self, batch):
        loss, acc = self._get_loss_acc(batch)
        self.log(
            f"test/loss_{self.test_name}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"test/acc_{self.test_name}", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def on_train_epoch_end(self):
        epoch_loss = self.running_loss / self.num_train_batches
        epoch_acc = self.running_acc / self.num_train_batches
        
        self.losses.append(epoch_loss)
        self.accs.append(epoch_acc)
        
        self.running_loss = 0.0
        self.running_acc = 0.0
        
    def on_validation_epoch_end(self):
        for idx, name in self.val_to_name.items():
            epoch_loss = self.running_loss_test[idx] / self.num_val_batches
            epoch_acc = self.running_acc_test[idx] / self.num_val_batches
            self.losses_test[name].append(epoch_loss)
            self.accs_test[name].append(epoch_acc)
            
            self.running_loss_test[idx] = 0.0
            self.running_acc_test[idx] = 0.0
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer
