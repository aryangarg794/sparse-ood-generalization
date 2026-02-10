import matplotlib.pyplot as plt
import numpy as np
import lightning as pl
import seaborn as sns
import torch 
import torch.nn as nn
import torch.nn.utils as utils
import wandb

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self

from sparse_generalization.models.mlp import BasicMLP
from sparse_generalization.layers.bern_mha import MultiHeadAttentionBern
from sparse_generalization.layers.thresh_mha import MultiHeadAttentionThresh
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency, L1SparsityWeights
from sparse_generalization.utils.util_funcs import noise_scheduler
from sparse_generalization.models.blocks import MHABlock

    
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
        dropout: int = 0.1, 
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
        noisy_grads: bool = False,
        eta: float = 1.0, 
        gamma: float = 0.55,
        noisy_bern: bool = False,
        beta1: float = 0.99, 
        beta2: float = 0.999,
        foopt: bool = False,
        eps: float = 1e-3,
        var: float = 1.0
    ):
        self.betas = (beta1, beta2)
        super().__init__()
        self.loss = loss()
        self.sparse = include_sparsity
        self.noisy_grads = noisy_grads
        self.eta = eta
        self.gamma = gamma
        self.noisy_bern = noisy_bern
        self.foopt = foopt
        self.eps = eps
        self.var = var
        
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
            mha_layer=mha_layer,
            noisy_bern=noisy_bern
        )
        
        self.accuracy = BinaryAccuracy()
        self.test_attn_matrices = []
        self.train_attn_matrices = []
        self.test_name = 'placeholder'
        self.automatic_optimization = False
        self.lagrangian = lagrangian

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
        if self.noisy_bern:
            bern_var = self.model.mha.noise_scheduler(self.global_step)
            self.log(
                "bern_noise", 
                bern_var, 
                on_step=False,
                on_epoch=True,
            )
            
        rec_loss, acc, attn = self._get_loss_acc(batch)
        opt = self.optimizers()
        if self.sparse:
            if self.lagrangian or self.foopt:
                if self.global_step == 0:
                    self.ema_loss = (rec_loss - self.target_loss).detach()
                else:
                    self.ema_loss = self.ema_step * self.ema_loss + (1- self.ema_step) \
                        * (rec_loss-self.target_loss).detach()
                        
            
            if self.lagrangian:
                sparse_loss = self.l1_loss(attn)
                loss = rec_loss + sparse_loss/self.lambd
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
        else:
            loss = rec_loss  
            
        opt.zero_grad()
        self.manual_backward(loss)
    
        if self.foopt:
            total_norm = utils.clip_grad_norm_(self.model.mha_parameters(), max_norm=float('inf'))
            if total_norm <= self.eps and rec_loss >= self.target_loss:
                with torch.no_grad():
                    for param in self.parameters():
                        noise = torch.randn_like(param) * self.var
                        param.grad.data.add_(noise)
                        
            self.var = torch.exp(self.step_size*self.ema_loss) * self.var
            self.var = torch.clamp(self.var, min=1e-3, max=1e3)
        
        if self.noisy_grads:
            with torch.no_grad():
                var = noise_scheduler(self.eta, self.global_step, self.gamma)
                for param in self.parameters():
                    noise = torch.randn_like(param) * var
                    param.grad.data.add_(noise)
            
            self.log(
                "noise", 
                var, 
                on_step=False,
                on_epoch=True,
            )

        opt.step()

        if self.lagrangian:
            self.lambd = torch.exp(self.step_size*self.ema_loss) * self.lambd
            self.lambd = torch.clamp(self.lambd, min=1e2, max=1e15)
            
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
            f"test_loss_{self.test_name}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"test_acc_{self.test_name}", 
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
        if self.l1_loss is None or isinstance(self.model.mha, MultiHeadAttentionThresh) \
            or isinstance(self.model.mha, torch.nn.MultiheadAttention):
            return (all_attn > 0.01).float().sum(dim=(1, 2)).mean().item()
        else:
            return all_attn.sum(dim=(1, 2)).mean().item()
            
    
    def configure_optimizers(self: Self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        return optimizer