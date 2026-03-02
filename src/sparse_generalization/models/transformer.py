import matplotlib.pyplot as plt
import numpy as np
import lightning as pl
import seaborn as sns
import torch 
import torch.nn as nn
import torch.nn.utils as utils
import wandb

from hydra.utils import instantiate
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self

from sparse_generalization.layers.thresh_mha import MultiHeadAttentionThresh
from sparse_generalization.losses.sparse_loss import L1SparsityWeights
from sparse_generalization.utils.util_funcs import noise_scheduler
from sparse_generalization.models.blocks import MHABlock
from sparse_generalization.layers.agg_attention import AggregationAttention

    
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
        agg_pool: bool, 
        layernorm: bool, 
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
        num_layers: int = 4, 
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
        self.agg_pool = agg_pool
        
        if include_sparsity:
            self.l1_loss = sparse_loss()  
            self.l1_weight = l1_weight
        else:
            self.l1_loss = None
            self.l1_weight = None
            
        self.lr = lr
        
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
        
        self.layers = nn.ModuleList()
        self.layers.append(MHABlock(
                embed_size=self.embed_size,
                out_dim=self.embed_size,
                residual=residual,  
                hidden_dims=hidden_dims,
                act=act,
                dropout=dropout,
                use_grid=use_grid, 
                mha_layer=mha_layer,
                num_heads=num_heads,
                layernorm=layernorm,  
            )
        )
        
        num_layers = num_layers-1 if self.agg_pool else num_layers-2
        for _ in range(num_layers):
            self.layers.append(MHABlock(
                embed_size=self.embed_size,
                out_dim=self.embed_size,
                residual=residual,  
                hidden_dims=hidden_dims,
                act=act,
                dropout=dropout,
                use_grid=use_grid,
                mha_layer=mha_layer,
                num_heads=num_heads,
                layernorm=layernorm,
            ))
        
        
        if self.agg_pool:
            self.out = AggregationAttention(
                num_heads=num_heads, 
                embed_size=embed_size, 
                out_dim=out_dim, 
                residual=residual, 
                hidden_dims=hidden_dims, 
                act=act,
                dropout=dropout,
                layernorm=layernorm
            )
        else:
            self.out = nn.Linear(self.embed_size, out_dim)
        
        self.accuracy = BinaryAccuracy()

        self.val_to_name = {
            0: 'id', 
            1: 'col', 
            2: 'pair', 
            3: 'dist', 
            4: 'comb'
        }
        self.test_attn_matrices = []
        self.val_attn_matrices = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.train_attn_matrices = []
        self.test_name = 'placeholder'
        self.automatic_optimization = False
        self.lagrangian = lagrangian

        self.lambd = start_lambda
        self.target_loss = target_loss
        self.step_size = step_size
        self.ema_step = cma
        
        self.running_loss = 0.0
        self.running_sparse = 0.0
        self.running_acc = 0.0
        self.losses = []
        self.accs = []
        self.sparses = []
        self.masks = []
        
        self.running_loss_test = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.running_acc_test = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.masks_test = {'id': [], 'col': [], 'pair': [], 'dist': [], 'comb': []}
        self.losses_test = {'id': [], 'col': [], 'pair': [], 'dist': [], 'comb': []}
        self.accs_test = {'id': [], 'col': [], 'pair': [], 'dist': [], 'comb': []}
        
        
    def _get_loss_acc(self: Self, batch):
        x, y = batch
        attn_matrices = []
        
        batch_size, width, height, _ = x.size()
        x_features = self.feature_map(x.permute(0, 3, 1, 2)) # (b, 3, w, h)
        x_features = x_features.permute(0, 2, 3, 1)
        
        if self.pe:
            device = x.device
            xs = torch.arange(width, device=device)
            ys = torch.arange(height, device=device)
            coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
            coords = coords.expand(batch_size, width, height, 2)
            x_attn = torch.cat([x_features, coords], dim=-1)
            x_attn = x_attn.view(-1, width*height, self.model_dim+2)
        
        for layer in self.layers:
            x_attn, attn = layer(x_attn)
            attn_matrices.append(attn)

        if self.agg_pool:
            y_hat, attn = self.out(x_attn)
        else:
            y_hat = self.out(x_attn.max(dim=1)[0]) 
        
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        path_matrix = self._compute_thresh_path(attn_matrices)
        return loss, acc, path_matrix # (b, l, h, h)

    def training_step(self, batch, batch_idx):
        if self.noisy_bern: #NOTE: doesnt work
            bern_var = self.model.mha.noise_scheduler(self.global_step)
            self.log(
                "train/bern_noise", 
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
                "train/sparse_loss", 
                sparse_loss, 
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            
            self.running_sparse += sparse_loss.item()
        else:
            loss = rec_loss  
            
        opt.zero_grad()
        self.manual_backward(loss)
    
        if self.foopt: #NOTE: doesnt work
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
                "train/noise", 
                var, 
                on_step=False,
                on_epoch=True,
            )

        opt.step()

        if self.lagrangian:
            self.lambd = torch.exp(self.step_size*self.ema_loss) * self.lambd
            self.lambd = torch.clamp(self.lambd, min=1e2, max=1e15)
            
        self.log(
            "train/loss", 
            loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        self.running_loss += loss.item()
        
        self.log(
            "train/acc", 
            acc, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        self.running_acc += acc.item()
        
        if self.lagrangian:
            self.log(
                "train/log_lambda", 
                self.lambd.log().item(), 
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )
        
        
        self.train_attn_matrices.append(attn)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, acc, attn = self._get_loss_acc(batch)
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
        
        
        self.val_attn_matrices[dataloader_idx].append(attn.detach().cpu())
        self.running_loss_test[dataloader_idx] += loss.item()
        self.running_acc_test[dataloader_idx] += acc.item()
        return loss
    
    def test_step(self, batch):
        loss, acc, attn = self._get_loss_acc(batch)
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
        self.test_attn_matrices.append(attn)
        return loss
    
    
    def on_train_epoch_end(self):
        all_attn = torch.cat(self.train_attn_matrices, dim=0) 
        num_attn = self._compute_attn_mean(all_attn)
        
        self.log(
            "train/num_edges",
            num_attn,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        self.masks.append(num_attn)
        
        self.sparses.append(self.running_sparse / self.num_train_batches)
        self.losses.append(self.running_loss / self.num_train_batches)
        self.accs.append(self.running_acc / self.num_train_batches)
        
        self.running_loss = 0.0
        self.running_sparse = 0.0
        self.running_acc = 0.0

        self.train_attn_matrices.clear()
        
        
    def on_validation_epoch_end(self):
        for idx, name in self.val_to_name.items():
            all_attn_id = torch.cat(self.val_attn_matrices[idx], dim=0) 
            num_attn_id = self._compute_attn_mean(all_attn_id)
            
            self.log(
                f"val/num_edges_{name}",
                num_attn_id,
                on_step=False,
                on_epoch=True,
            )
            
            epoch_loss = self.running_loss_test[idx] / self.num_val_batches
            epoch_acc = self.running_acc_test[idx] / self.num_val_batches
            self.losses_test[name].append(epoch_loss)
            self.accs_test[name].append(epoch_acc)
            
            self.running_loss_test[idx] = 0.0
            self.running_acc_test[idx] = 0.0
            self.masks_test[name].append(num_attn_id)
            self.val_attn_matrices[idx].clear() 
    
    def on_test_epoch_end(self):
        all_attn = torch.cat(self.test_attn_matrices, dim=0) 
        avg_attn = all_attn.mean(dim=0)  
        
        # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        # sns.heatmap(avg_attn.detach().cpu().numpy(), annot=True, cmap="viridis", xticklabels=[0,1,2], yticklabels=[0,1,2], ax=ax)
        # plt.title(f'Average Attention Matrix {self.test_name}')
        # plt.xlabel("Key")
        # plt.ylabel("Query")
        # self.logger.experiment.log({f'Heatmap Attn {self.test_name}': wandb.Image(fig)})
        # plt.close()
        
        num_attn = self._compute_attn_mean(all_attn)
        self.log(
            f"test/num_edges_{self.test_name}",
            num_attn,
            on_step=False,
            on_epoch=True,
        )

        self.test_attn_matrices.clear()
    
    
    def _compute_thresh_path(self: Self, attn_list: List):
        thresh_list = [(attn > 0.01).float() for attn in attn_list]
        batch_size, seq_len, _ = thresh_list[0].size()
        path = torch.eye(seq_len, device=self.device).repeat(batch_size, 1, 1)
        for attn in reversed(thresh_list):
            path = path @ attn
        
        return path
    
    def _compute_attn_mean(self: Self, all_attn: Tensor):
        if self.l1_loss is None or isinstance(self.layers[0].mha, MultiHeadAttentionThresh) \
            or isinstance(self.layers[0].mha, torch.nn.MultiheadAttention):
            return all_attn.float().sum(dim=(1, 2)).mean().item()
        else:
            return all_attn.sum(dim=(1, 2)).mean().item()
            
    
    def configure_optimizers(self: Self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer