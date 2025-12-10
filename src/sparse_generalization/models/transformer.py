import matplotlib.pyplot as plt
import lightning as pl
import seaborn as sns
import torch 
import torch.nn as nn
import wandb

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self

from sparse_generalization.models.mlp import BasicMLP
from sparse_generalization.losses.sparse_loss import L1Sparsity, L1SparsityAdjacency

class MHABlock(nn.Module):
    """Basic transformer block for the toy example 

    Args:
        nn (_type_): _description_
    """
    
    def __init__(
        self: Self, 
        embed_size: int, 
        input_dim: int, 
        out_dim: int,
        num_heads: int, # for the toy example just keep it one
        hidden_dims: List,
        positional_encoding: bool, 
        act: nn.Module,
        dropout: int,
        residual: bool, 
        mha_layer: nn.Module, 
        *args, 
        **kwargs
    ):
        super(MHABlock, self).__init__(*args, **kwargs)
        self.residual = residual
        embed_size = embed_size + 1 if positional_encoding else embed_size
        self.pe = positional_encoding
         
        self.mha = mha_layer(embed_size, num_heads=num_heads, dropout=dropout, batch_first=True) # (b, 3, 1) or (b, 3, 2) with pe
        # self.norm = nn.LayerNorm(embed_size) # layer norm does not work for toy example
        self.mlp = BasicMLP(input_dim=embed_size, out_dim=out_dim, hidden_dims=hidden_dims, act=act) # (b, 3) 
    
    def forward(self: Self, x: Tensor):
        x = x.unsqueeze(dim=-1) # so we treat each input as a node in the graph with dim 1
        if self.pe:
            batch_size, seq_len, _ = x.size()
            device = x.device
            indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
            indices = indices.expand(batch_size, seq_len, 1)
            x = torch.cat([x, indices], dim=-1)
        attn_out, attn_scores = self.mha(x, x, x)
        # out = self.norm(attn_out)
        if self.residual:
            out = self.mlp((attn_out + x).max(dim=1)[0])
        else:
            out = self.mlp(attn_out.max(dim=1)[0])
        return out, attn_scores
    
class TransformerLit(pl.LightningModule):
    """Lighting Model for the basic MHA block

    Args:

    """
    def __init__(
        self: Self, 
        embed_size: int, 
        input_dim: int, 
        out_dim: int,
        num_heads: int, # for the toy example just keep it one
        hidden_dims: List,
        mha_layer: nn.Module = nn.MultiheadAttention, 
        act: nn.Module = nn.ReLU,
        dropout: int = 0.0, 
        lr: float = 1e-3,
        residual: bool = True,
        include_sparsity: bool = False,
        sparse_loss: nn.Module = L1Sparsity, 
        l1_weight: float = 0.1,  
        positional_encoding: bool = True, 
        loss: nn.Module = nn.BCEWithLogitsLoss,
    ):
        super().__init__()
        self.loss = loss()
        self.sparse = include_sparsity
        if include_sparsity:
            self.l1_loss = sparse_loss()
            self.l1_weight = l1_weight
        self.lr = lr
        
        self.model = MHABlock(
            embed_size=embed_size,
            input_dim=input_dim, 
            out_dim=out_dim,
            num_heads=num_heads,
            residual=residual,  
            hidden_dims=hidden_dims,
            act=act,
            positional_encoding=positional_encoding,
            dropout=dropout,
            mha_layer=mha_layer
        )
        
        self.accuracy = BinaryAccuracy()
        self.test_attn_matrices = []
        self.test_name = 'placeholder'
        
    def _get_loss_acc(self: Self, batch):
        x, y = batch
        y_hat, attn = self.model(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc, attn

    def training_step(self, batch, batch_idx):
        loss, acc, attn = self._get_loss_acc(batch)
        if self.sparse:
            sparse_loss = self.l1_weight * self.l1_loss(attn)
            loss = loss + sparse_loss
            self.log(
                "sparse_loss", 
                sparse_loss, 
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
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
        loss, acc, _ = self._get_loss_acc(batch)
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
        loss, acc, attn = self._get_loss_acc(batch)
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
        self.test_attn_matrices.append(attn)
        return loss
    
    def on_test_epoch_end(self):
        all_attn = torch.cat(self.test_attn_matrices, dim=0) 
        avg_attn = all_attn.mean(dim=0)  
        
        table = wandb.Table(columns=["From\\To", "Token 0", "Token 1", "Token 2"])
        for i in range(3):
            table.add_data(f"Token {i}", float(avg_attn[i,0]), float(avg_attn[i,1]), float(avg_attn[i,2]))
        
        self.logger.experiment.log({"avg_attention_table": table})
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        sns.heatmap(avg_attn.detach().cpu().numpy(), annot=True, cmap="viridis", xticklabels=[0,1,2], yticklabels=[0,1,2], ax=ax)
        plt.title(f'Average Attention Matrix {self.test_name}')
        plt.xlabel("Key")
        plt.ylabel("Query")
        self.logger.experiment.log({f'Heatmap Attn {self.test_name}': wandb.Image(fig)})
        plt.close()
        

        self.test_attn_matrices.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer