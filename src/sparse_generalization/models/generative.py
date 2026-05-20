import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from sparse_generalization.models.blocks import MHABlockGen
from sparse_generalization.layers.agg_attention import AggregationAttention
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.utils.util_funcs import positionalencoding2d


class FlowFormer(nn.Module):

    def __init__(
        self,
        inp_dim: int = 3,
        seq_len: int = 25, 
        out_dim: int = 1,
        model_dim: int = 64,
        latent_dim: int = 32, 
        num_feature_layers: int = 3,
        lstm_layers: int = 1, 
        num_heads: int = 1,
        num_layers: int = 4,
        agg_pool: bool = False,
        residual: bool = True,
        include_sparsity: bool = False,
        alpha: float = 0.1, 
        token_pool: bool = False,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        step_size: float = 1e-1,
        pe: bool = True,
        sinusoidal: bool = True,
        bidirectional: bool = True, 
        flow_params: dict = {'n_flows' : 2, 'hidden_features' : (128, 128)},
        prior_params: dict = {'n_flows' : 3, 'hidden_features' : (256, 256)},
        nf_prior: bool = True, 
        embedding_inp: bool = True,
        lr: float = 1e-3,
        dropout: float = 0.1, 
        layernorm: bool = True,
        act: nn.Module = nn.ReLU,
        logger: WandbLogger = None,
        num_embeddings: int = 25,
        separate_mask: bool = False, 
        device: str = "cuda",
        beta1: float = 0.9,
        beta2: float = 0.999,
        threshold: float = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.logger = logger
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.agg_pool = agg_pool
        self.token_pool = token_pool

        self.feature_map = nn.Sequential()
        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, 4*model_dim)

        self.feature_map.extend(
            [nn.Conv2d(in_channels=4*model_dim if embedding_inp else inp_dim, 
                       out_channels=model_dim, kernel_size=1),
            act()]
        )
        
        for i in range(num_feature_layers-1):
            if i == num_feature_layers-1:
                self.feature_map.extend(
                    [
                        nn.Conv2d(
                            in_channels=model_dim, out_channels=model_dim, kernel_size=1
                        ),
                    ]
                )
            else:
                self.feature_map.extend(
                    [
                        nn.Conv2d(
                            in_channels=model_dim, out_channels=model_dim, kernel_size=1
                        ),
                        act(),
                    ]
                )

        if pe:
            if sinusoidal:
                model_dim = model_dim
            else:
                model_dim += 2
            embed_size = model_dim

        self.embed_size = embed_size
        self.pe = pe
        self.sinusoidal = sinusoidal
        self.embedding_inp = embedding_inp

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                MHABlockGen(
                    embed_size,
                    seq_len,
                    latent_dim=latent_dim, 
                    num_heads=num_heads,
                    dropout=dropout,
                    lstm_layers=lstm_layers, 
                    bidirectional=bidirectional, 
                    prior_params=prior_params,
                    flow_params=flow_params,
                    nf_prior=nf_prior, 
                    residual=residual, 
                )
            )

        if self.agg_pool:
            self.out = AggregationAttention(
                num_heads=num_heads,
                embed_size=self.embed_size,
                out_dim=out_dim,
                residual=False,
                act=act,
                use_mask=True, 
                device=device, 
                separate_mask=separate_mask, 
                dropout=dropout,
                layernorm=layernorm,
            )
        elif self.token_pool:
            self.cls = nn.Parameter(torch.rand(1, self.embed_size, device=self.device))
            self.out = nn.Linear(self.embed_size, out_dim)
        else:
            self.out = nn.Linear(self.embed_size, out_dim)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.threshold = threshold

        self.sparse_loss = L1SparsityAdjacency()
        self.alpha = alpha
        self.include_sparsity = include_sparsity if not self.lagrangian else True
        self.max_paths = None
        self.step_size = step_size
        self.val_to_name = val_to_name
        self.args = locals()

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def forward(self, x: Tensor):
        attn_matrices = []
        batch_size, width, height, _ = x.size()
        if self.max_paths is None:
            self.max_paths = self._compute_max_paths(width * height)

        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)

        x_features = self.feature_map(x.permute(0, 3, 1, 2))  # (b, e, w, h)
        x_features = x_features.permute(0, 2, 3, 1)

        masks = (
            torch.eye(width * height, device=self.device).repeat(batch_size, 1, 1)
        )

        if self.pe:
            device = x.device
            if self.sinusoidal:
                embeddings = positionalencoding2d(self.embed_size, width, height)
                embeddings = embeddings.view(width, height, -1).to(
                    device=device
                )  # (w, h, e)
                x_attn = x_features + embeddings.repeat(batch_size, 1, 1, 1)
                x_attn = x_attn.view(-1, width * height, self.embed_size)
            else:
                xs = torch.arange(width, device=device)
                ys = torch.arange(height, device=device)
                coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
                coords = coords.expand(batch_size, width, height, 2)
                x_attn = torch.cat([x_features, coords], dim=-1)
                x_attn = x_attn.view(-1, width * height, self.embed_size)

        if self.token_pool:
            clses = self.cls.repeat(batch_size, 1, 1)
            x_attn = torch.cat([x_attn, clses], dim=1)

        for layer in self.layers:
            if self.training:
                x_attn, mask, attn, gen_loss = layer(x_attn)
            else:
                x_attn, mask, attn = layer(x_attn)
            attn_matrices.append(attn)

            if self.token_pool: 
                mask = mask[:, :-1, :-1]
            if self.path_sparsity:  
                masks = torch.bmm(mask, masks)
            else:
                masks.append(mask)

        if self.agg_pool:
            out, final_mask, agg_attn = self.out(x_attn)
        elif self.token_pool:
            out = self.out(x_attn[:, -1, :])
        else:
            out = self.out(x_attn.max(dim=1)[0])

        if self.agg_pool:
            attn_matrices.append(agg_attn)
            masks = torch.bmm(final_mask, masks)

        return out, masks, attn_matrices, gen_loss if self.training else None  

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
            epoch_masks = []
            epochs_trues = []

            for batch_idx, batch in enumerate(dataloader):
                if self.compute_mask:
                    x, y, mask = batch
                    mask = mask.to(self.device)
                else:
                    x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, attns = self(x)  # list of (b, l, l)
                rec_loss = self.loss(out, y)

                if self.path_sparsity:
                    path_matrix = masks
                else:
                    path_matrix = torch.stack(masks, dim=1).mean(dim=1)

                if self.compute_mask:
                    epoch_masks.append(path_matrix)
                    epochs_trues.append(mask)

                if self.include_sparsity:
                    if self.lagrangian:
                        if self.global_step == 0:
                            self.ema_loss = (rec_loss - self.target_loss).detach()
                        else:
                            self.ema_loss = (
                                self.ema_step * self.ema_loss
                                + (1 - self.ema_step)
                                * (rec_loss - self.target_loss).detach()
                            )

                    if self.lagrangian:
                        sparse_loss = self.sparse_loss(path_matrix)
                        loss = rec_loss + sparse_loss / self.lambd
                    else:
                        sparse_loss = self._enforce_sparsity(path_matrix)
                        loss = rec_loss + sparse_loss

                    epoch_sparse += sparse_loss.item()

                else:
                    loss = rec_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lagrangian:
                    self.lambd = torch.exp(self.step_size * self.ema_loss) * self.lambd
                    self.lambd = torch.clamp(self.lambd, min=5e3, max=1e15)

                epoch_loss += rec_loss.item()
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

            if self.compute_mask:
                epoch_masks = torch.cat(epoch_masks, dim=0)
                epochs_trues = torch.cat(epochs_trues, dim=0)
                mask_score = self._mask_score(epochs_trues, epoch_masks).item()
                postfix["mask_score"] = mask_score
                self.logger.log_metrics({"train/mask_score": mask_score}, step=step)

            self.logger.log_metrics({"train/loss_epoch": epoch_loss}, step=step)

            self.logger.log_metrics({"train/acc_epoch": epoch_acc}, step=step)

            if self.include_sparsity:
                self.logger.log_metrics({"train/sparse_loss": epoch_sparse}, step=step)
                postfix["sparse_loss"] = epoch_sparse

            if self.lagrangian:
                log_lam = self.lambd.log().item()
                self.logger.log_metrics({"train/log_lambda": log_lam}, step=step)
                postfix["lambd"] = log_lam

            if self.alpha_res:
                with torch.no_grad():
                    for i, layer in enumerate(self.layers):
                        postfix[f"alpha_lay{i}"] = nn.functional.sigmoid(
                            layer.alpha
                        ).item()

            self.logger.log_metrics(
                {f"train/attn_edges_train": attn_running}, step=self.global_step
            )

            self.logger.log_metrics(
                {f"train/mask_edges_train": mask_running}, step=self.global_step
            )

            for loader, name in zip(testloaders, self.val_to_name.values()):
                test_metrics = self.test(name, loader, folder="val")
                if 'id' in name:
                    postfix['val_id'] = test_metrics["acc"]
                masks_test[name].append(test_metrics["mask"])
                attn_test[name].append(test_metrics["attn"])
                losses_test[name].append(test_metrics["loss"])
                accs_test[name].append(test_metrics["acc"])

            postfix["edges"] = mask_running

            # if self.agg_pool:
            #     self.out.temp_decay(step, num_epochs)
            #     postfix["temp"] = self.out.temp

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

    def test(self, name: str, dataloader: DataLoader, folder: str = "test"):
        self.eval()
        attn_running = 0.0
        mask_running = 0.0
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_masks = []
        epochs_trues = []

        for batch_idx, batch in enumerate(dataloader):
            if self.compute_mask:
                x, y, mask = batch
                mask = mask.to(self.device)
            else:
                x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out, masks, attn = self(x)
            loss = self.loss(out, y)

            if self.compute_mask:
                epoch_masks.append(masks)
                epochs_trues.append(mask)

            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                attn_running += self._compute_attn_mean(attn)
                mask_running += self._compute_mask_mean(masks)

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)

        if self.compute_mask:
            epoch_masks = torch.cat(epoch_masks, dim=0)
            epochs_trues = torch.cat(epochs_trues, dim=0)
            mask_score = self._mask_score(epochs_trues, epoch_masks).item()
            self.logger.log_metrics({f"{folder}/mask_score_{name}": mask_score}, step=self.global_step)

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
            out, mask, attn = self(x)
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

    def _compute_attn_mean(self, all_attn: Tensor):
        thresh_list = [
            (attn > self.threshold).float() for attn in all_attn
        ]  # list of (b, l, l)
        batch_size, seq_len, _ = thresh_list[0].size()
        path = torch.eye(seq_len, device=self.device).repeat(batch_size, 1, 1)
        for attn in thresh_list:
            path = attn @ path

        return path.sum(dim=(1, 2)).mean().item()

    def _compute_mask_mean(self, all_masks: Tensor):
        return all_masks.sum(dim=(1, 2)).mean().item()

    def _compute_max_paths(self, seq_len: int):
        paths = torch.ones((seq_len, seq_len)) * self.num_heads
        for l in range(self.num_layers-1):
            multiplier = torch.ones((seq_len, seq_len)) * self.num_heads
            paths = paths @ multiplier

        if self.agg_pool:
            multiplier = torch.ones((1, seq_len)) * self.num_heads
            paths = multiplier @ paths

        return paths.sum().item()
    
    def _mask_score(self, masks, paths):
        paths_bool = (paths.squeeze() > 1).int()
        masks = masks.view(-1, paths.size(-1)) 

        mask1 = (paths_bool == 1)
        mask2 = (masks == 1) 
        batch_result = (mask1 | ~mask2).all(dim=1).float()
        return batch_result.mean()