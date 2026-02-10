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

class SPARTAN(nn.Module):
    
    def __init__(
        self, 
        embed_sizes,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        
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
            
        x_attn = x_attn.mean(dim=1)
        out = self.ffn(x_attn)
        
        return out, torch.concat(masks, dim=1).mean(dim=1), torch.concat(attn_matrices, dim=1).mean(dim=1)
    
    def fit(self, dataloader: DataLoader, num_epochs: int):
        
        for step in (pbar := tqdm(range(1, num_epochs+1))): 
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_idx, (x, y, edges) in enumerate(dataloader):
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
            out, mask, attn = self(x, edges)
            loss = self.loss(out, y)
            
            masks.append(mask)
            attns.append(attn)
            
            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                
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