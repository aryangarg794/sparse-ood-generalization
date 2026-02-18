import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch 
import torch.nn as nn
import torch.nn.utils as utils
import wandb

from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from sparse_generalization.models.blocks import MHABlockBern, MHABlockOracle
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency

class SPARTAN(nn.Module):
    
    def __init__(
        self,  
        embed_size: int, 
        out_dim: int, 
        hidden_dims_ffn: list, 
        model_dim: int, 
        num_feature_layers: int, 
        num_heads: int, 
        num_layers: int, 
        residual: bool, 
        include_sparsity: bool, 
        path_sparsity: bool,
        alpha_res: bool, 
        l1_weight: float = 0.1, 
        lagrangian: bool = False, 
        target_loss: float = 0.05, 
        start_lambda: float = 1e7, 
        step_size: float = 1e-1, 
        cma: float = 0.9, 
        pe: bool = True,
        lr: float = 1e-3, 
        dropout: float = 0.1, 
        use_grid: bool = True, 
        act: nn.Module = nn.ReLU, 
        logger: WandbLogger = None, 
        device: str = 'cuda', 
        threshold: float = 0.01, 
        separate_mask: bool = False, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.logger = logger
        self.model_dim = model_dim
        
        self.feature_map = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=model_dim, kernel_size=1),
            act()
        )
        for _ in range(num_feature_layers):
            self.feature_map.extend([
                nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=1),
                act()
            ]) 
                                   
            
        if pe and not use_grid:
            embed_size += 1
        elif pe and use_grid:
            model_dim += 2  
            embed_size = model_dim  
        
        self.embed_size = embed_size
        self.pe = pe
        
        self.layers = nn.ModuleList()
        self.layers.append(MHABlockBern(
            embed_size=self.embed_size, 
            use_grid=use_grid, 
            num_heads=num_heads, 
            hidden_dims=hidden_dims_ffn, 
            residual=residual, 
            dropout=dropout, 
            separate_mask=separate_mask,
            act=act,
            alpha_res=alpha_res
        ))
        
        for _ in range(num_layers-1):
            self.layers.append(MHABlockBern(
                embed_size=self.embed_size, 
                use_grid=use_grid, 
                num_heads=num_heads, 
                hidden_dims=hidden_dims_ffn,
                residual=residual, 
                dropout=dropout, 
                separate_mask=separate_mask,
                act=act,
                alpha_res=alpha_res
            ))
            
        self.ffn = nn.Linear(self.embed_size, out_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.threshold = threshold
        
        self.sparse_loss = L1SparsityAdjacency()
        self.path_sparsity = path_sparsity
        self.lagrangian = lagrangian
        self.include_sparsity = include_sparsity if not self.lagrangian else True
        self.l1_weight = l1_weight
        self.lambd = start_lambda
        self.target_loss = target_loss
        self.step_size = step_size
        self.ema_step = cma
        
        self.alpha_res = alpha_res
        
        
    def forward(self, x: Tensor):
        attn_matrices = []
        
        batch_size, width, height, _ = x.size()
        x_features = self.feature_map(x.permute(0, 3, 1, 2)) # (b, 3, w, h)
        x_features = x_features.permute(0, 2, 3, 1)
        masks = torch.eye(width*height, device=self.device).repeat(batch_size, 1, 1) if self.path_sparsity else []
        
        if self.pe:
            device = x.device
            xs = torch.arange(width, device=device)
            ys = torch.arange(height, device=device)
            coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
            coords = coords.expand(batch_size, width, height, 2)
            x_attn = torch.cat([x_features, coords], dim=-1)
            x_attn = x_attn.view(-1, width*height, self.model_dim+2)
        
        for layer in self.layers:
            x_attn, mask, attn = layer(x_attn)
            attn_matrices.append(attn)
            if self.path_sparsity:
                masks = torch.bmm(mask, masks)
            else:
                masks.append(mask)
            
        x_attn = x_attn.max(dim=1)[0]
        out = self.ffn(x_attn)
        
        return out, masks, attn_matrices # (b, k, l, l)
    
    def fit(self, dataloader: DataLoader, num_epochs: int):
        
        for step in (pbar := tqdm(range(1, num_epochs+1))): 
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_sparse = 0.0
            attn_running = 0.0
            mask_running = 0.0
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, attns = self(x) # list of (b, l, l)
                rec_loss = self.loss(out, y)
                if self.path_sparsity: 
                    path_matrix = masks
                else:
                    path_matrix = torch.stack(masks, dim=1).mean(dim=1)
                
                if self.include_sparsity:
                    if self.lagrangian:
                        if self.global_step == 0:
                            self.ema_loss = (rec_loss - self.target_loss).detach()
                        else:
                            self.ema_loss = self.ema_step * self.ema_loss + (1- self.ema_step) \
                                * (rec_loss-self.target_loss).detach()
                                
                    if self.lagrangian:
                        sparse_loss = self.sparse_loss(path_matrix)
                        loss = rec_loss + sparse_loss/self.lambd
                    else:
                        sparse_loss = self.l1_weight * self.sparse_loss(path_matrix)
                        loss = rec_loss + sparse_loss
                        
                    epoch_sparse += sparse_loss.item()
                              
                else:
                    loss = rec_loss
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.lagrangian:
                    self.lambd = torch.exp(self.step_size*self.ema_loss) * self.lambd
                    self.lambd = torch.clamp(self.lambd, min=1e1, max=1e15)
                
                epoch_loss += loss.item()
                with torch.no_grad():
                    acc = self.accuracy(out, y)
                    epoch_acc += acc.item()

                    attn_running += self._compute_attn_mean(attns)
                    mask_running += self._compute_mask_mean(path_matrix)
                
                self.global_step += 1
            
            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            epoch_sparse /= len(dataloader)
            attn_running /= len(dataloader)
            mask_running /= len(dataloader)

            
            postfix = {
                "loss": epoch_loss,
                "acc": epoch_acc,
            }
            pbar.set_description(f'Epoch: {step}')
            
            self.logger.log_metrics(
                {'train/loss_epoch': epoch_loss},
                step=step
            )
            
            self.logger.log_metrics(
                {'train/acc_epoch': epoch_acc},
                step=step
            )
            
            if self.include_sparsity:
                self.logger.log_metrics(
                    {"train/sparse_loss" : epoch_sparse},
                    step=step
                )  
                postfix["sparse_loss"] = epoch_sparse
            
            if self.lagrangian:
                log_lam = self.lambd.log().item()
                self.logger.log_metrics(
                    {"train/log_lambda": log_lam}, 
                    step=step
                )
                postfix["lambd"] = log_lam
            
            if self.alpha_res:
                with torch.no_grad():
                    for i, layer in enumerate(self.layers):
                        postfix[f"alpha_lay{i}"] = nn.functional.sigmoid(layer.alpha).item()
            
            self.logger.log_metrics(
                {f'train/attn_edges_train': attn_running},
                step=self.global_step
            )
            
            self.logger.log_metrics(
                {f'train/mask_edges_train': mask_running},
                step=self.global_step
            )    
            
            postfix['edges'] = mask_running
            
            pbar.set_postfix(postfix)

            
            
    def test(self, name: str, dataloader: DataLoader):
        self.eval()
        attn_running = 0.0
        mask_running = 0.0
        epoch_acc = 0.0
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            out, mask, attn = self(x)
            loss = self.loss(out, y)
            
            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                attn_running += self._compute_attn_mean(attn)
                mask_running += self._compute_mask_mean(mask)
                
        
        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)
        
        self.logger.log_metrics(
            {f'test/loss_epoch_{name}': epoch_loss},
            step=self.global_step
        )
        
        self.logger.log_metrics(
            {f'test/acc_epoch_{name}': epoch_acc},
            step=self.global_step
        )
    
        
        self.logger.log_metrics(
            {f'test/attn_edges_{name}': attn_running},
            step=self.global_step
        )
        
        self.logger.log_metrics(
            {f'test/mask_edges_{name}': mask_running},
            step=self.global_step
        )
        
    
        self.train()
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def _compute_attn_mean(self, all_attn: Tensor):
        thresh_list = [(attn > self.threshold).float() for attn in all_attn] # list of (b, l, l)
        batch_size, seq_len, _ = thresh_list[0].size()
        path = torch.eye(seq_len, device=self.device).repeat(batch_size, 1, 1)
        for attn in reversed(thresh_list):
            path = path @ attn
        
        return path.sum(dim=(1, 2)).mean().item()
    
    def _compute_mask_mean(self, all_masks: Tensor):
        return all_masks.sum(dim=(1, 2)).mean().item()
        
        
class OracleTransformer(nn.Module):
    
    def __init__(
        self, 
        embed_size: int, 
        out_dim: int, 
        hidden_dims_ffn: list, 
        model_dim: int, 
        num_feature_layers: int, 
        num_heads: int, 
        num_layers: int, 
        residual: bool, 
        pe: bool = True,
        lr: float = 1e-3, 
        dropout: float = 0.1, 
        use_grid: bool = True, 
        act: nn.Module = nn.ReLU, 
        device: str = 'cuda', 
        logger: WandbLogger = None, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logger = logger
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
            
        if pe and not use_grid:
            embed_size += 1
        elif pe and use_grid:
            model_dim += 2  
            embed_size = model_dim  
        
        self.embed_size = embed_size
        self.pe = pe
        
        self.layers = nn.ModuleList()
        self.layers.append(MHABlockOracle(
            embed_size=self.embed_size, 
            use_grid=use_grid, 
            num_heads=num_heads, 
            hidden_dims=hidden_dims_ffn, 
            residual=residual, 
            dropout=dropout, 
            act=act,
        ))
        
        for _ in range(num_layers-1):
            self.layers.append(MHABlockOracle(
                embed_size=self.embed_size, 
                use_grid=use_grid, 
                num_heads=num_heads, 
                hidden_dims=hidden_dims_ffn,
                residual=residual, 
                dropout=dropout, 
                act=act,
            ))
            
        self.ffn = nn.Linear(self.embed_size, out_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.device = device
        
    def forward(self, x: Tensor, edges: List):
        masks = []
        attn_matrices = []
        
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
        
        for layer in self.layers:
            x_attn, mask, attn = layer(x_attn, edges)
            masks.append(mask)
            attn_matrices.append(attn)
            
        x_attn = x_attn.max(dim=1)[0]
        out = self.ffn(x_attn)
        
        return out, torch.concat(masks, dim=1).mean(dim=1), torch.concat(attn_matrices, dim=1).mean(dim=1)
    
    def fit(self, dataloader: DataLoader, num_epochs: int):
        
        for step in (pbar := tqdm(range(1, num_epochs+1))): 
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_idx, (x, y, edges) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                out, mask, attn = self(x, edges)
                loss = self.loss(out, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                with torch.no_grad():
                    acc = self.accuracy(out, y)
                    epoch_acc += acc.item()
                    
                self.global_step += 1
            
            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            
            pbar.set_description(f"Epoch: {step} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
            
            self.logger.log_metrics(
                {'train/loss_epoch': epoch_loss},
                step=step
            )
            
            self.logger.log_metrics(
                {'train/acc_epoch': epoch_acc},
                step=step
            )
            
    def test(self, name: str, dataloader: DataLoader):
        self.eval()
        masks = []
        attns = []
        epoch_acc = 0.0
        epoch_loss = 0.0
        for batch_idx, (x, y, edges) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            out, mask, attn = self(x, edges)
            loss = self.loss(out, y)
            
            masks.append(mask)
            attns.append(attn)
            
            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                
        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
                
        masks = torch.cat(masks, dim=0)
        attns = torch.cat(attns, dim=0)
        
        self.logger.log_metrics(
            {f'test/loss_epoch_{name}': epoch_loss},
            step=self.global_step
        )
        
        self.logger.log_metrics(
            {f'test/acc_epoch_{name}': epoch_acc},
            step=self.global_step
        )
    
        self.train()
        
        return {'loss': epoch_loss, 'acc': epoch_acc}, masks, attns