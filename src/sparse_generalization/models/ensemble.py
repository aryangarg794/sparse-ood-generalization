import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from functools import partial
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from sparse_generalization.models.blocks import MHABlockBern, MHABlock
from sparse_generalization.layers.agg_attention import AggregationAttention
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.utils.util_funcs import (
    positionalencoding2d,
    compute_attn_mean_ens,
    compute_mask_mean,
    compute_max_paths,
)

class EnsembleMember(nn.Module):

    def __init__(
        self, 
        inp_dim: int = 3,
        model_dim: int = 64,
        out_dim: int = 1,
        num_heads: int = 1,
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        num_layers: int = 4,
        embedding_inp: bool = True,
        residual: bool = True,
        num_embeddings: int = 64,
        layernorm: bool = True,
        agg_pool: bool = False,
        sinusoidal: bool = True,
        positional_encoding: bool = True,
        device: str = 'cpu', 
        spartan: bool = False, 
        *args, 
        **kwargs
    ):
        super().__init__(
            *args, 
            **kwargs
        )

        self.agg_pool = agg_pool
        self.residual = residual
        self.device = device

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)

        bottleneck = 128
        self.feature_map = nn.Sequential(
            nn.Linear(model_dim if embedding_inp else inp_dim, bottleneck),
            act(),
            nn.Linear(bottleneck, model_dim),
            # nn.Identity()
        )

        if positional_encoding:
            if sinusoidal:
                model_dim = model_dim
            else:
                model_dim += 2

        self.embed_size = model_dim
        self.pe = positional_encoding
        self.sinusoidal = sinusoidal
        self.embedding_inp = embedding_inp
        self.spartan = spartan

        self.layers = nn.ModuleList()

        if spartan:
            mha_block = partial(MHABlockBern, 
                                embed_size=self.embed_size, 
                                layernorm=layernorm, 
                                num_heads=num_heads,
                                zeros=False, 
                                residual=residual,
                                dropout=dropout,
                                mask_res=False,
                                separate_mask=False,
                                act=act,
                                alpha_res=False
                                )
            use_mask = True
        else:
            mha_block = partial(MHABlock, 
                                embed_size=self.embed_size, 
                                layernorm=layernorm, 
                                num_heads=num_heads,
                                residual=residual,
                                dropout=dropout,
                                act=act,
                                mha_layer=nn.MultiheadAttention,
                                )
            use_mask = False


        for _ in range(num_layers):
            self.layers.append(
                mha_block()
            )

        if self.agg_pool:
            self.out = AggregationAttention(
                num_heads=num_heads,
                embed_size=self.embed_size,
                out_dim=out_dim,
                residual=False,
                act=act,
                use_mask=use_mask,
                device=device,
                separate_mask=False,
                dropout=dropout,
                layernorm=layernorm,
            )
        else:
            self.out = nn.Linear(self.embed_size, out_dim)

    def forward(self, x: Tensor):
        attn_matrices = []
        mask_attn_matrices = []
        mask_matrices = []
        
        batch_size, width, height, _ = x.size()
        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)

        x_features = self.feature_map(x)
        masks = torch.eye(width * height, device=self.device).repeat(batch_size, 1, 1)
        if self.pe:
            device = x.device
            if self.sinusoidal:
                embeddings = positionalencoding2d(
                    self.embed_size, height=height, width=width, device=self.device
                ).permute(2, 1, 0)
                x_attn = x_features + embeddings.repeat(batch_size, 1, 1, 1)
                x_attn = x_attn.view(-1, width * height, self.embed_size)
            else:
                xs = torch.arange(width, device=device)
                ys = torch.arange(height, device=device)
                coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
                coords = coords.expand(batch_size, width, height, 2)
                x_attn = torch.cat([x_features, coords], dim=-1)
                x_attn = x_attn.view(-1, width * height, self.embed_size)
        else:
            x_attn = x_features.view(-1, width * height, self.embed_size)

        if self.spartan:
            for layer in self.layers:
                x_attn, mask, mask_attn, attn = layer(x_attn, forced_expl=False)
                attn_matrices.append(attn.detach())
                mask_attn_matrices.append(mask_attn.detach())
                masks = torch.bmm(mask, masks)
                mask_matrices.append(mask.detach())
        else:
            for layer in self.layers:
                x_attn, attn = layer(x_attn)
                mask_attn_matrices.append(attn)
                mask = torch.ones_like(attn)
                masks = torch.bmm(mask, masks)

        if self.agg_pool:
            out, final_mask, mask_attn, agg_attn = self.out(x_attn)
        else:
            out = self.out(x_attn.max(dim=1)[0])

        if self.agg_pool:
            attn_matrices.append(agg_attn.detach())
            mask_attn_matrices.append(mask_attn.detach())
            masks = torch.bmm(final_mask, masks)
            mask_matrices.append(final_mask.detach())

        return (
            out,
            masks,
            torch.stack(mask_attn_matrices, dim=0),
            attn_matrices,
            mask_matrices,
        ) 

class Ensemble(nn.Module):

    def __init__(
        self, 
        inp_dim: int = 3,
        out_dim: int = 1,
        num_models: int = 5, 
        spartan: bool = False, 
        model_dim: int = 64,
        num_heads: int = 1,
        num_layers: int = 4,
        agg_pool: bool = False,
        residual: bool = True,
        include_sparsity: bool = False,
        ensemble_loss: str = "mean",
        alpha: float = 0.1,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        pe: bool = True,
        sinusoidal: bool = True,
        embedding_inp: bool = True,
        lr: float = 1e-3,
        dropout: float = 0.0,
        val_freq: int = 10, 
        layernorm: bool = False,
        act: nn.Module = nn.ReLU,
        logger: WandbLogger = None,
        num_embeddings: int = 64,
        device: str = "cuda",
        beta1: float = 0.9,
        beta2: float = 0.999,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.models = nn.ModuleList()
        self.include_sparsity = include_sparsity
        assert not include_sparsity or spartan, "Ensemble sparsity only for spartan"
        self.alpha = alpha
        self.device = device
        self.logger = logger
        self.ensemble_loss = ensemble_loss
        self.val_to_name = val_to_name
        self.val_freq = val_freq

        for _ in range(num_models):
            self.models.append(
                EnsembleMember(
                    inp_dim=inp_dim, 
                    model_dim=model_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    act=act,
                    dropout=dropout,
                    num_layers=num_layers,
                    embedding_inp=embedding_inp,
                    residual=residual,
                    num_embeddings=num_embeddings,
                    layernorm=layernorm,
                    agg_pool=agg_pool,
                    sinusoidal=sinusoidal,
                    positional_encoding=pe,
                    device=device,
                    spartan=spartan
                )
            )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.global_step = 0
        self.sparse_loss = L1SparsityAdjacency()
        self.max_paths = None
        self.num_models = num_models
        self.num_heads = num_heads
        self.residual = residual
        self.agg_pool = agg_pool
        self.num_layers = num_layers

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def forward(self, x: Tensor):
        _, width, height, _ = x.size()
        if self.max_paths is None:
            self.max_paths = compute_max_paths(width * height)

        outputs = []
        masks = []
        mask_attns = []
        attns = []
        for model in self.models:
            out, mask, mask_attn, attn, _ = model(x)
            outputs.append(out)
            masks.append(mask)
            mask_attns.append(mask_attn)
            attns.append(attn)
        
        return torch.stack(outputs, dim=1), torch.stack(masks, dim=1), torch.stack(mask_attns, dim=0), attns

    def predict(self, x: Tensor):
        out, mask, mask_attn, attn = self(x)
        return out.mean(dim=1), mask, mask_attn, attn

    def fit(self, dataloader: DataLoader, num_epochs: int, testloaders: List):
        losses = []
        accs = []
        attn_edges = []
        mask_edges = []
        sparses = []

        attn_test = {i: [] for i in self.val_to_name.values()}
        masks_test = deepcopy(attn_test)
        losses_test = deepcopy(attn_test)
        accs_test = deepcopy(attn_test)

        for step in (pbar := tqdm(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_sparse = 0.0
            attn_running = 0.0
            mask_running = 0.0

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, mask_attns, attns = self(x)  # list of (b, l, l)

                rec_loss = self.loss(out, y.unsqueeze(dim=1).expand(-1, 3, -1))
                if self.ensemble_loss == "mean":
                    rec_loss = rec_loss.mean()
                elif self.ensemble_loss == "sum":
                    rec_loss = rec_loss.mean(dim=0).sum()

                if self.include_sparsity:
                    sparse_loss = self._enforce_sparsity(masks.mean(dim=1) if self.ensemble_loss == "mean" else masks.sum(dim=1))
                    loss = rec_loss + sparse_loss
                    epoch_sparse += sparse_loss.item()
                else:
                    loss = rec_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += rec_loss.item()
                with torch.no_grad():
                    acc = self.accuracy(self.predict(x)[0], y)
                    epoch_acc += acc.item()

                    threshold = 1 / x.size(1)
                    attn_running += compute_attn_mean_ens(mask_attns, threshold=threshold, device=self.device)
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            epoch_sparse /= len(dataloader)
            attn_running /= len(dataloader)
            mask_running /= len(dataloader)

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            sparses.append(epoch_sparse)
            attn_edges.append(attn_running)
            mask_edges.append(mask_running)

            postfix = {
                "loss": epoch_loss,
                "acc": epoch_acc,
            }

            pbar.set_description(f"Epoch: {step}")

            self.logger.log_metrics({"train/loss_epoch": epoch_loss}, step=step)

            self.logger.log_metrics({"train/acc_epoch": epoch_acc}, step=step)

            if self.include_sparsity:
                self.logger.log_metrics({"train/sparse_loss": epoch_sparse}, step=step)
                postfix["sparse_loss"] = epoch_sparse


            self.logger.log_metrics(
                {f"train/attn_edges_train": attn_running}, step=self.global_step
            )

            self.logger.log_metrics(
                {f"train/mask_edges_train": mask_running}, step=self.global_step
            )

            if step % self.val_freq == 0:
                for loader, name in zip(testloaders, self.val_to_name.values()):
                    test_metrics = self.test(name, loader, folder="val")
                    if "id" in name:
                        postfix["val_id"] = test_metrics["acc"]
                    elif "a" in name:
                        postfix["val_a"] = test_metrics["acc"]
                    elif "b" in name:
                        postfix["val_b"] = test_metrics["acc"]
    
                    masks_test[name].append(test_metrics["mask"])
                    attn_test[name].append(test_metrics["attn"])
                    losses_test[name].append(test_metrics["loss"])
                    accs_test[name].append(test_metrics["acc"])

            postfix["mask_edges"] = mask_running
            postfix["attn_edges"] = attn_running

            pbar.set_postfix(postfix)

        return (
            losses,
            accs,
            sparses,
            mask_edges,
            attn_edges,
            losses_test,
            accs_test,
            attn_test,
            masks_test,
        )

    @torch.no_grad()
    def test(self, name: str, dataloader: DataLoader, folder: str = "test"):
        self.eval()
        attn_running = 0.0
        mask_running = 0.0
        epoch_acc = 0.0
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out, mask, mask_attn, attn = self.predict(x)

            loss = self.loss(out, y).mean() 
            epoch_loss += loss.item()

            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()

                threshold = 1 / x.size(1)
                attn_running += compute_attn_mean_ens(mask_attn, threshold=threshold, device=self.device)
                mask_running += compute_mask_mean(mask)

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)

        self.logger.log_metrics(
            {f"{folder}/loss_epoch_{name}": epoch_loss}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/acc_epoch_{name}": epoch_acc}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/attn_edges_{name}": attn_running}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/mask_edges_{name}": mask_running}, step=self.global_step
        )

        self.train()

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "attn": attn_running,
            "mask": mask_running,
        }

    @torch.no_grad()
    def test_anti(self, anti_dataset: DataLoader):
        # total acc, acc a, acc b, conf a, conf b
        results = {}
        labels = []
        true_labels = []
        for batch_idx, (x, y) in enumerate(anti_dataset):
            x = x.to(self.device)
            y = y.to(self.device)
            out, mask, mask_attn, attn = self(x)
            probs = F.sigmoid(out)
            labels.append(probs)
            true_labels.append(y)

        preds = torch.cat(labels, dim=0)
        trues = torch.cat(true_labels, dim=0)
        size = preds.size(0)
        midpoint = size // 2

        total_acc = self.accuracy(preds, trues)
        results["total_acc"] = total_acc.item()

        acc_a = self.accuracy(preds[:midpoint], trues[:midpoint])
        acc_b = self.accuracy(preds[midpoint:], trues[midpoint:])
        conf_a = preds[:midpoint].mean()
        conf_b = preds[:midpoint].mean()

        results["acc_a"] = acc_a.item()
        results["acc_b"] = acc_b.item()
        results["conf_a"] = conf_a.item()
        results["conf_b"] = conf_b.item()

        return results
