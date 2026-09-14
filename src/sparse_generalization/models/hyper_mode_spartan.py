"""SPARTAN with K independent mask hypernetworks, one per mode.

Same trunk, heads, bookkeeping (path/attention matrices), training and evaluation as
ConditionalSPARTAN, but the modes are not FiLM contexts: every mode owns a MaskNet
that emits the (L x L) attention masks for each layer, and the transformer itself is
fully shared and unconditioned. Each MaskNet is a deterministic function m(x) of the
input tokens, so every sample gets its own mask per mode per layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from sparse_generalization.layers.masknet import MaskNet, HyperMaskAggAttention
from sparse_generalization.models.blocks import HyperMaskBlock
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.utils.util_funcs import (
    positionalencoding2d,
    compute_mask_mean,
    compute_max_paths,
)
from sparse_generalization.layers.diversity_losses import (
    CosineDiv, CosineRepDiv, L2DistanceDiv, MaskOverlapDiv
)


class HyperModeSPARTAN(nn.Module):

    def __init__(
        self,
        inp_dim: int = 3,
        out_dim: int = 1,
        include_sparsity: bool = False,
        alpha: float = 0.1,
        num_modes: int = 3,
        num_mha_layers: int = 1,
        num_eval_samples: int = 5,
        model_dim: int = 32,
        num_heads: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
        device: str = "cuda",
        layernorm: bool = True,
        act: nn.Module = nn.ReLU,
        val_freq: int = 10,
        div_coeff: float = 0.0,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        pe: bool = True,
        pe_type: str = "sin",  # 'sin' | 'coord' | 'learned'
        max_grid_size: int = 5,
        embedding_inp: bool = True,
        lr: float = 1e-3,
        logger: WandbLogger = None,
        div_loss: nn.Module = MaskOverlapDiv,
        num_embeddings: int = 25,
        use_optimal_test: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        threshold: float = 0.01,
        temp: float = 1.0,
        output_type: str = "linear",  # 'agg' | 'linear'
        head_pool: str = "mean",  # 'mean' | 'max' | 'concat'
        seq_len: int = None,  # number of tokens (grid cells); defaults to inp_dim
        # MaskNet (hypernetwork) settings
        hyper_hidden_dim: int = 128,
        hyper_num_hidden: int = 2,
        mask_bias: float = 0.5,
        hard: bool = True,
        *args,
        **kwargs
    ):
        self.hyper_params = locals()

        for key in ["self", "__class__", "args", "kwargs"]:
            del self.hyper_params[key]

        super().__init__(*args, **kwargs)

        if output_type not in ("agg", "linear"):
            raise ValueError(f"output_type must be 'agg' or 'linear', got {output_type!r}")
        if head_pool not in ("mean", "max", "concat"):
            raise ValueError(f"head_pool must be 'mean', 'max' or 'concat', got {head_pool!r}")
        if pe_type not in ("sin", "coord", "learned"):
            raise ValueError(f"pe_type must be 'sin', 'coord' or 'learned', got {pe_type!r}")

        self.device = device
        self.logger = logger
        self.model_dim = model_dim
        self.val_freq = val_freq
        self.num_heads = num_heads
        self.div_coeff = div_coeff
        self.use_optimal_test = use_optimal_test
        self.num_mha_layers = num_mha_layers
        self.output_type = output_type
        self.head_pool = head_pool
        self.num_eval_samples = num_eval_samples
        self.out_dim = out_dim
        self.num_modes = num_modes
        self.seq_len = inp_dim if seq_len is None else seq_len
        self.avg_heads = div_coeff == 0

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)

        bottleneck = 128
        self.feature_map = nn.Sequential(
            nn.Linear(model_dim if embedding_inp else inp_dim, bottleneck),
            act(),
            nn.Linear(bottleneck, model_dim),
        )

        embed_size = model_dim
        if pe and pe_type == "coord":
            embed_size = model_dim + 2
        if pe and pe_type == "learned":
            self.pe_row = nn.Embedding(max_grid_size, model_dim)
            self.pe_col = nn.Embedding(max_grid_size, model_dim)

        self.embed_size = embed_size
        self.pe = pe
        self.pe_type = pe_type
        self.embedding_inp = embedding_inp
        self.div_loss = div_loss()

        # K independent hypernetworks, one per mode; the trunk below is fully shared
        self.masknets = nn.ModuleList(
            [
                MaskNet(
                    seq_len=self.seq_len,
                    embed_size=self.embed_size,
                    num_layers=num_mha_layers,
                    num_heads=num_heads,
                    hidden_dim=hyper_hidden_dim,
                    num_hidden=hyper_num_hidden,
                    temp=temp,
                    hard=hard,
                    bias=mask_bias,
                    agg_layer=output_type == "agg",
                    act=act,
                )
                for _ in range(num_modes)
            ]
        )

        self.layers = nn.ModuleList(
            [
                HyperMaskBlock(
                    embed_size=self.embed_size,
                    act=act,
                    dropout=dropout,
                    layernorm=layernorm,
                    residual=residual,
                    num_heads=num_heads,
                )
                for _ in range(num_mha_layers)
            ]
        )

        if output_type == "agg":
            self.out = HyperMaskAggAttention(self.embed_size, out_dim, num_heads=num_heads, residual=residual)
        else:
            head_dim = self.embed_size * self.seq_len if head_pool == "concat" else self.embed_size
            self.out = nn.Linear(head_dim, out_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.threshold = threshold

        self.sparse_loss = L1SparsityAdjacency()
        self.alpha = alpha
        self.include_sparsity = include_sparsity
        self.max_paths = None
        self.val_to_name = val_to_name

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def compute_masks(self, x_attn: Tensor, deterministic: bool = None):
        """m_k(x) for every mode k: layer masks (b, m, n, h, L, L), agg masks (b, m, h, L) | None."""
        layer_masks, agg_masks = [], []
        for net in self.masknets:
            lm, am, _, _ = net(x_attn, deterministic=deterministic)
            layer_masks.append(lm)
            agg_masks.append(am)
        layer_masks = torch.stack(layer_masks, dim=1)
        agg_masks = torch.stack(agg_masks, dim=1) if agg_masks[0] is not None else None
        return layer_masks, agg_masks

    def _pool(self, x_attn: Tensor):
        if self.head_pool == "mean":
            return x_attn.mean(dim=2)
        if self.head_pool == "max":
            return x_attn.max(dim=2)[0]
        return x_attn.flatten(2)  # (b, e, l * d)

    def forward(
        self,
        x: Tensor,
        evaluate: bool = False,
        ret_mean: bool = True
    ):
        batch_size, width, height, _ = x.size()
        seq_len = width * height
        num_evals = self.num_modes
        compute_div = self.div_coeff != 0.0

        if self.max_paths is None:
            self.max_paths = compute_max_paths(
                seq_len, self.num_heads, self.num_mha_layers, self.output_type == "agg"
            )
            print(f"MAX PATHS: {self.max_paths}")

        self.threshold = 1 / seq_len

        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)

        x_features = self.feature_map(x)
        if self.pe and self.pe_type == "sin":
            embeddings = positionalencoding2d(
                self.embed_size, height=height, width=width, device=self.device
            ).permute(2, 1, 0)  # (dim, h, w) -> (w, h, dim)
            x_attn = x_features + embeddings.unsqueeze(0)
        elif self.pe and self.pe_type == "learned":
            rows = self.pe_row(torch.arange(width, device=self.device))  # (w, d)
            cols = self.pe_col(torch.arange(height, device=self.device))  # (h, d)
            x_attn = x_features + (rows.unsqueeze(1) + cols.unsqueeze(0)).unsqueeze(0)
        elif self.pe:  # coord
            xs = torch.arange(width, device=self.device)
            ys = torch.arange(height, device=self.device)
            coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
            coords = coords.expand(batch_size, width, height, 2)
            x_attn = torch.cat([x_features, coords], dim=-1)
        else:
            x_attn = x_features
        x_attn = x_attn.reshape(batch_size, seq_len, self.embed_size)

        # masks are a deterministic function of the tokens; only the edge binarisation
        # is stochastic in training (hard Gumbel), thresholded in eval
        layer_masks, agg_masks = self.compute_masks(x_attn)  # (b, m, n, h, l, l), (b, m, h, l)

        div = torch.tensor([0.0], device=self.device)
        attn_maps = []
        eye = torch.eye(seq_len, device=self.device)
        path_matrix = eye.expand(batch_size, num_evals, seq_len, seq_len).clone()
        attn_matrix = eye.expand(batch_size, num_evals, seq_len, seq_len).clone()

        x_attn = x_attn.unsqueeze(1).expand(-1, num_evals, -1, -1)  # (b, e, l, d)

        for layer_idx, layer in enumerate(self.layers):
            x_attn, mask, mask_attn, attn = layer(
                x_attn, layer_masks[:, :, layer_idx], avg_heads=self.avg_heads
            )
            attn_maps.append(attn)
            if not self.avg_heads:  # (b, h, e, l, l) -> (b, e, l, l)
                mask = mask.sum(dim=1)
                mask_attn = mask_attn.sum(dim=1)
            thresh = (mask_attn > self.threshold).float()
            attn_matrix = torch.matmul(thresh, attn_matrix)
            path_matrix = torch.matmul(mask, path_matrix)

        if self.output_type == "agg":
            out, final_mask, mask_attn, agg_attn = self.out(x_attn, agg_masks)  # (b, e, o), (b, e, l)
            # agg map is (b, e, l), so it is not stacked with the (b, e, l, l) layer maps for CosineDiv
            thresh = (mask_attn > self.threshold).float().unsqueeze(2)  # (b, e, 1, l)
            attn_matrix = torch.matmul(thresh, attn_matrix)
            path_matrix = torch.matmul(final_mask.unsqueeze(2), path_matrix)
        else:
            out = self.out(self._pool(x_attn))  # (b, e, o)

        if num_evals > 1:  # every div loss needs at least two modes to compare
            match self.div_loss:
                case CosineDiv():
                    attn_maps = [a if a.dim() == 5 else a.unsqueeze(1) for a in attn_maps]
                    # list of (b, h, e, l, l) -> (e, b, n, h, l, l)
                    attns_probs = torch.stack(attn_maps, dim=1).permute(3, 0, 1, 2, 4, 5)
                    div = self.div_loss(attns_probs)
                case CosineRepDiv() | L2DistanceDiv():
                    div = self.div_loss(x_attn.transpose(0, 1))  # (e, b, l, d)
                case MaskOverlapDiv():
                    div = self.div_loss(path_matrix.transpose(0, 1))  # (e, b, l, l)

        if not compute_div:
            div = div.detach()

        out = out.transpose(0, 1).reshape(-1, out.size(-1))
        path_matrix = path_matrix.transpose(0, 1).reshape(-1, path_matrix.size(-2), seq_len)
        attn_matrix = attn_matrix.transpose(0, 1).reshape(-1, attn_matrix.size(-2), seq_len)

        if evaluate:
            out = torch.sigmoid(out).view(num_evals, batch_size, -1)
            path_matrix = path_matrix.view(num_evals, batch_size, -1, seq_len)
            attn_matrix = attn_matrix.view(num_evals, batch_size, -1, seq_len)
            if ret_mean:
                return out.mean(dim=0), path_matrix, attn_matrix
            return out, path_matrix, attn_matrix

        return out, path_matrix, attn_matrix, div

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

        postfix = {"loss": 0.0, "acc": 0.0}

        for step in (pbar := tqdm(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = 0.0
            epoch_div = 0.0
            epoch_acc = 0.0
            epoch_sparse = 0.0
            attn_running = 0.0
            mask_running = 0.0

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, attns, div = self(x)
                out = out.view(self.num_modes, -1, self.out_dim)
                pointwise_losses = F.binary_cross_entropy_with_logits(
                    out,
                    y.unsqueeze(0).expand_as(out),
                    reduction="none",
                )
                loss_per_model = pointwise_losses.mean(dim=(1, 2))
                rec_loss = loss_per_model.mean()

                if self.include_sparsity:
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.item()
                    loss = rec_loss + sparse_loss + self.div_coeff * div
                else:
                    loss = rec_loss + self.div_coeff * div

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += rec_loss.item()
                epoch_div += div.item()

                with torch.no_grad():
                    acc = self.accuracy(out, y.expand_as(out))
                    epoch_acc += acc.item()

                    attn_running += compute_mask_mean(attns)
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            epoch_sparse /= len(dataloader)
            epoch_div /= len(dataloader)
            attn_running /= len(dataloader)
            mask_running /= len(dataloader)

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            sparses.append(epoch_sparse)
            attn_edges.append(attn_running)
            mask_edges.append(mask_running)

            postfix["loss"] = epoch_loss
            postfix["acc"] = epoch_acc
            postfix["div"] = epoch_div

            pbar.set_description(f"Epoch: {step}")
            self.logger.log_metrics({"train/loss_epoch": epoch_loss}, step=step)
            self.logger.log_metrics({"train/acc_epoch": epoch_acc}, step=step)
            self.logger.log_metrics({"train/div_epoch": epoch_div}, step=step)

            if self.include_sparsity:
                self.logger.log_metrics({"train/sparse_loss": epoch_sparse}, step=step)
                postfix["sparse_loss"] = epoch_sparse

            self.logger.log_metrics(
                {f"train/attn_edges_train": attn_running}, step=self.global_step
            )

            self.logger.log_metrics(
                {f"train/mask_edges_train": mask_running}, step=self.global_step
            )

            if not self.use_optimal_test and step % self.val_freq == 0:
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

            postfix["mask"] = mask_running

            if self.use_optimal_test and step % self.val_freq == 0:
                for loader, name in zip(testloaders, self.val_to_name.values()):
                    test_metrics = self.optimal_test(name, loader, folder="val")
                    if "id" in name:
                        postfix["ens_id"] = test_metrics["acc"]
                    elif "a" in name:
                        postfix["ens_a"] = test_metrics["acc"]
                    elif "b" in name:
                        postfix["ens_b"] = test_metrics["acc"]

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

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                out, masks, attns = self(x, evaluate=True)
                loss = F.binary_cross_entropy(out, y)

                epoch_loss += loss.item()
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                attn_running += compute_mask_mean(attns)
                mask_running += compute_mask_mean(masks)

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

    @torch.inference_mode()
    def test_anti(self, anti_dataset: DataLoader):
        self.eval()
        # total acc, acc a, acc b, conf a, conf b
        results = {}
        labels = []
        true_labels = []
        for batch_idx, (x, y) in enumerate(anti_dataset):
            x = x.to(self.device)
            y = y.to(self.device)
            probs, masks, attns = self(x, evaluate=True)
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
        conf_b = preds[midpoint:].mean()

        results["acc_a"] = acc_a.item()
        results["acc_b"] = acc_b.item()
        results["conf_a"] = conf_a.item()
        results["conf_b"] = conf_b.item()

        self.train()
        return results

    @torch.inference_mode()
    def optimal_test(self, name: str, dataloader: DataLoader, folder: str = "test"):
        self.eval()
        attn_running = 0.0
        mask_running = 0.0
        epoch_acc = 0.0
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            outs, masks, attns = self(x, evaluate=True, ret_mean=False)
            loss = float("inf")
            acc = float("-inf")
            for out in outs:
                loss = min(F.binary_cross_entropy(out, y), loss)
                acc = max(self.accuracy(out, y), acc)

            attn_running += compute_mask_mean(attns)
            mask_running += compute_mask_mean(masks)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)

        self.logger.log_metrics(
            {f"{folder}/loss_ens_{name}": epoch_loss}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/acc_ens_{name}": epoch_acc}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/attn_ens_{name}": attn_running}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/mask_ens_{name}": mask_running}, step=self.global_step
        )

        self.train()

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "attn": attn_running,
            "mask": mask_running,
        }
