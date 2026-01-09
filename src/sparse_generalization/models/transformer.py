import matplotlib.pyplot as plt
import numpy as np
import lightning as pl
import seaborn as sns
import torch 
import torch.nn as nn
import wandb

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self

from sparse_generalization.models.mlp import BasicMLP
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency, L1SparsityWeights

class MHABlock(nn.Module):
    """Basic transformer block for the toy example 

    Args:
        nn (_type_): _description_
    """
    
    def __init__(
        self: Self, 
        embed_size: int, 
        use_grid: bool,
        model_dim: int, 
        num_feature_layers: int, 
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
        self.use_grid = use_grid
        self.model_dim = model_dim

        if use_grid:
            self.feature_map = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=model_dim, kernel_size=1),
                act()
            )
            for _ in range(num_feature_layers):
                self.feature_map.extend([
                    nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=1),
                    act()
                ])                        
            
        if positional_encoding and not use_grid:
            embed_size += 1
        elif positional_encoding and use_grid:
            model_dim += 2  
            embed_size = model_dim  
        
        self.embed_size = embed_size
        self.pe = positional_encoding
         
        self.mha = mha_layer(embed_size, num_heads=num_heads, dropout=dropout, batch_first=True) # (b, 3, 1) or (b, 3, 2) with pe
        # self.norm = nn.LayerNorm(embed_size) # layer norm does not work for toy example
        self.mlp = BasicMLP(input_dim=embed_size, out_dim=out_dim, hidden_dims=hidden_dims, act=act) # (b, 3) 
    
    def forward(self: Self, x: Tensor):
        if self.use_grid:
            return self._forward_image(x)
        else:
            return self._forward_basic(x)
        
        
    def _forward_basic(self: Self, x: Tensor):
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
    
    def _forward_image(self: Self, x: Tensor):
        # x is (w, h, 3)
        assert self.use_grid
        batch_size, width, height, _ = x.size()
        x_features = self.feature_map(x.permute(0, 3, 1, 2)) # (b, 3, w, h)
        x_features = x_features.permute(0, 2, 3, 1)
        x_attn = x_features.view(-1, width*height, self.model_dim)
        
        if self.pe:
            device = x.device
            xs = torch.arange(width, device=device)
            ys = torch.arange(height, device=device)
            coords = torch.cartesian_prod(xs, ys)
            coords = coords.expand(batch_size, width*height, 2)
            x_attn = torch.cat([x_attn, coords], dim=-1)
        
        attn_out, attn_scores = self.mha(x_attn, x_attn, x_attn)
        if self.residual:
            out = self.mlp((attn_out + x_attn).max(dim=1)[0])
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
        use_grid: bool, 
        model_dim: int, 
        out_dim: int,
        num_heads: int, # for the toy example just keep it one
        hidden_dims: List,
        num_feature_layers: int = 3, 
        mha_layer: nn.Module = nn.MultiheadAttention, 
        act: nn.Module = nn.ReLU,
        dropout: int = 0.0, 
        lr: float = 1e-3,
        residual: bool = True,
        include_sparsity: bool = False,
        sparse_loss: nn.Module = L1SparsityWeights, 
        l1_weight: float = 0.1,  
        positional_encoding: bool = True,
        k: int = 5,  
        loss: nn.Module = nn.BCEWithLogitsLoss,
        lagrangian: bool = False, 
        target_loss: float = 0.05, 
        start_lambda: float = 1e7,
        step_size: float = 1e-1, 
        cma: float = 0.9, 
    ):
        super().__init__()
        self.loss = loss()
        self.sparse = include_sparsity
        if include_sparsity:
            if isinstance(sparse_loss, L1SparsityWeights):
                self.l1_loss = sparse_loss(k=k)
            else:
                self.l1_loss = sparse_loss()    
            
            self.l1_weight = l1_weight
        else:
            self.l1_loss = None
            self.l1_weight = None
            
        self.lr = lr
        
        self.model = MHABlock(
            embed_size=embed_size,
            out_dim=out_dim,
            model_dim=model_dim, 
            use_grid=use_grid,
            num_heads=num_heads,
            residual=residual,  
            hidden_dims=hidden_dims,
            act=act,
            num_feature_layers=num_feature_layers, 
            positional_encoding=positional_encoding,
            dropout=dropout,
            mha_layer=mha_layer
        )
        
        self.accuracy = BinaryAccuracy()
        self.test_attn_matrices = []
        self.train_attn_matrices = []
        self.test_name = 'placeholder'
        self.automatic_optimization = False
        self.lagrangian = lagrangian
        if lagrangian:
            self.lambd = start_lambda
            self.target_loss = target_loss
            self.step_size = step_size
            self.ema_step = cma
        
    def _get_loss_acc(self: Self, batch):
        x, y = batch
        y_hat, attn = self.model(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc, attn

    def training_step(self, batch, batch_idx):
        rec_loss, acc, attn = self._get_loss_acc(batch)
        opt = self.optimizers()
        if self.sparse:
            if self.lagrangian:
                sparse_loss = self.l1_loss(attn)
                loss = rec_loss + sparse_loss/self.lambd
                if self.global_step == 0:
                    self.ema_loss = (rec_loss - self.target_loss).detach()
                else:
                    self.ema_loss = self.ema_step * self.ema_loss + (1- self.ema_step) \
                        * (rec_loss-self.target_loss).detach()
            else:
                sparse_loss = self.l1_weight * self.l1_loss(attn)
                loss = rec_loss + sparse_loss
                
            self.log(
                "sparse_loss", 
                sparse_loss, 
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        if self.lagrangian:
            self.lambd = torch.exp(self.step_size*self.ema_loss) * self.lambd
            self.lambd = torch.clamp(self.lambd, min=1e-20, max=1e30)
            
        self.log(
            "train_loss", 
            loss, 
            on_step=False,
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
        
        if self.lagrangian:
            self.log(
                "log_lambda", 
                self.lambd.log().item(), 
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )
        
        
        self.train_attn_matrices.append(attn)
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
    
    
    def on_train_epoch_end(self):
        all_attn = torch.cat(self.train_attn_matrices, dim=0) 
        num_attn = self._compute_attn_mean(all_attn)
        
        self.log(
            "avg_num_edges_train",
            num_attn,
            on_step=False,
            on_epoch=True,
        )

        self.train_attn_matrices.clear()  
    
    def on_test_epoch_end(self):
        all_attn = torch.cat(self.test_attn_matrices, dim=0) 
        avg_attn = all_attn.mean(dim=0)  
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        sns.heatmap(avg_attn.detach().cpu().numpy(), annot=True, cmap="viridis", xticklabels=[0,1,2], yticklabels=[0,1,2], ax=ax)
        plt.title(f'Average Attention Matrix {self.test_name}')
        plt.xlabel("Key")
        plt.ylabel("Query")
        self.logger.experiment.log({f'Heatmap Attn {self.test_name}': wandb.Image(fig)})
        plt.close()
        
        num_attn = self._compute_attn_mean(all_attn)
        self.log(
            "avg_num_edges_test",
            num_attn,
            on_step=False,
            on_epoch=True,
        )

        self.test_attn_matrices.clear()
    
    def _compute_attn_mean(self: Self, all_attn: Tensor):
        if self.l1_loss is None or isinstance(self.l1_loss, L1SparsityWeights):
            return (all_attn > 0.01).float().sum(dim=(1, 2)).mean().item()
        else:
            return all_attn.sum(dim=(1, 2)).mean().item()
            
    
    def configure_optimizers(self: Self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer