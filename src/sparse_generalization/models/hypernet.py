import torch
import torch.nn as nn

from copy import deepcopy
from functools import partial
from torch import Tensor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.layers.vae import FlowVAE
from sparse_generalization.layers.hypernet import HyperNet
from sparse_generalization.layers.priors import make_unit_gaussian
from sparse_generalization.utils.util_funcs import (
    get_device,
    positionalencoding2d,
    compute_mask_mean,
    compute_max_paths,
    build_lr_scheduler,
)
from sparse_generalization.losses.criterion import Criterion
from sparse_generalization.layers.diversity_losses import CosineDiv


class HyperNetSpartan(nn.Module):

    def __init__(
        self,
        inp_dim: int = 3,
        out_dim: int = 2,  # head size: 2 for ce (softmax classes), 1 for bce (single sigmoid logit)
        loss_type: str = "ce",  # 'ce' | 'bce'
        weight_gen = partial(FlowVAE, prior_func=make_unit_gaussian),  # partial of FlowVAE | VHyperNet
        prior_type: str = "uniform",
        include_sparsity: bool = False,
        alpha: float = 0.1,
        num_mha_layers: int = 1,
        include_agg_layer: bool = False,
        seq_len: int = 25,
        num_eval_samples: int = 5,
        model_dim: int = 32,
        num_heads: int = 1,
        dropout: float = 0.0,
        hyper_type: str = "qk",
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        device: str | None = None,
        forward_evals: int = 1,
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        act: nn.Module = nn.ReLU,
        val_freq: int = 10,
        div_coeff: float = 0.1,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        pe_type: str = "sin",  # 'sin' | 'coord' | 'learned' | 'none'
        max_grid_size: int = 5,
        embedding_inp: bool = True,
        lr: float = 1e-3,
        lr_decay: str = "none",  # 'none' | 'linear'
        lr_warmup: bool = False,
        beta: float = 1.0,
        logger: WandbLogger = None,
        div_loss: nn.Module = CosineDiv,
        num_embeddings: int = 25,
        use_optimal_test: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        threshold: float = 0.01,
        train_query: str = "train",  # 'fixed' | 'train' | 'ema'
        agg_ema: float = 0.99, 
        *args,
        **kwargs
    ):
        self.hyper_params = locals()

        for key in ["self", "__class__", "args", "kwargs"]:
            del self.hyper_params[key]

        if lr_decay not in ("none", "linear"):
            raise ValueError(f"lr_decay must be 'none' or 'linear', got {lr_decay!r}")

        device = get_device(device)

        super().__init__(*args, **kwargs)

        self.device = device
        self.lr_decay = lr_decay
        self.lr_warmup = lr_warmup
        self.logger = logger
        self.model_dim = model_dim
        self.val_freq = val_freq
        self.num_heads = num_heads
        self.div_coeff = div_coeff
        self.use_optimal_test = use_optimal_test
        self.num_mha_layers = num_mha_layers
        self.include_agg_layer = include_agg_layer
        self.num_eval_samples = num_eval_samples
        self.forward_evals = forward_evals
        self.criterion = Criterion(loss_type)
        self.out_dim = out_dim
        self.avg_heads = div_coeff == 0

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)

        bottleneck = 128
        self.feature_map = nn.Sequential(
            nn.Linear(model_dim if embedding_inp else inp_dim, bottleneck),
            act(),
            nn.Linear(bottleneck, model_dim),
        )

        if pe_type not in ("sin", "coord", "learned", "none"):
            raise ValueError(f"pe_type must be 'sin', 'coord', 'learned' or 'none', got {pe_type!r}")

        embed_size = model_dim
        if pe_type == "coord":
            embed_size = model_dim + 2
        if pe_type == "learned":
            self.pe_row = nn.Embedding(max_grid_size, model_dim)
            self.pe_col = nn.Embedding(max_grid_size, model_dim)

        self.embed_size = embed_size
        self.pe_type = pe_type
        self.embedding_inp = embedding_inp

        self.hyper_net = HyperNet(
            weight_gen=weight_gen,
            prior_type=prior_type,
            num_mha_layers=num_mha_layers,
            include_agg_layer=include_agg_layer,
            seq_len=seq_len,
            embed_size=embed_size,
            out_dim=out_dim,
            criterion=self.criterion,
            num_heads=num_heads,
            dropout=dropout,
            hyper_type=hyper_type,
            prior_params=prior_params,
            residual=residual,
            div_loss=div_loss,
            device=device,
            layernorm=layernorm,
            separate_mask=separate_mask,
            use_mask=use_mask,
            act=act,
            train_query=train_query,
            agg_ema=agg_ema,
            forward_evals=forward_evals,
        )

        if self.hyper_net.fixed_evals:
            self.num_eval_samples = forward_evals

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.scheduler = None
        self.loss = self.criterion.loss
        self.global_step = 0
        self.threshold = threshold

        self.sparse_loss = L1SparsityAdjacency()
        self.alpha = alpha
        self.include_sparsity = include_sparsity
        self.max_paths = None
        self.val_to_name = val_to_name
        self.beta = beta

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def forward(self, x: Tensor, evaluate: bool = False, ret_mean: bool = True):
        batch_size, width, height, _ = x.size()

        if self.max_paths is None:
            self.max_paths = compute_max_paths(
                width * height, self.num_heads, self.num_mha_layers, self.include_agg_layer
            )

            print(f"MAX PATHS: {self.max_paths}")

        self.threshold = 1 / (width * height)

        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)

        x_features = self.feature_map(x)
        if self.pe_type == "sin":
            embeddings = positionalencoding2d(
                self.embed_size, height=height, width=width, device=self.device
            ).permute(  # returns (dim, h, w)
                2, 1, 0
            )
            x_attn = x_features + embeddings.repeat(batch_size, 1, 1, 1)
        elif self.pe_type == "learned":
            rows = self.pe_row(torch.arange(width, device=self.device))  # (w, d)
            cols = self.pe_col(torch.arange(height, device=self.device))  # (h, d)
            embeddings = rows.unsqueeze(1) + cols.unsqueeze(0)  # (w, h, d)
            x_attn = x_features + embeddings.unsqueeze(0)
        elif self.pe_type == "coord":
            xs = torch.arange(width, device=self.device)
            ys = torch.arange(height, device=self.device)
            coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
            coords = coords.expand(batch_size, width, height, 2)
            x_attn = torch.cat([x_features, coords], dim=-1)
        else:
            x_attn = x_features
        x_attn = x_attn.view(-1, width * height, self.embed_size)

        if evaluate:
            return self.hyper_net.evaluate(x_attn, num_eval_samples=self.num_eval_samples, ret_mean=ret_mean)

        return self.hyper_net(
            x_attn, avg_heads=self.avg_heads, num_evals=self.forward_evals, compute_div=self.div_coeff != 0.0
        )

    def fit(self, dataloader: DataLoader, num_epochs: int, testloaders: List):
        losses = []
        accs = []
        attn_edges = []
        mask_edges = []
        sparses = []
        gens = []

        attn_test = {i: [] for i in self.val_to_name.values()}
        masks_test = deepcopy(attn_test)
        losses_test = deepcopy(attn_test)
        accs_test = deepcopy(attn_test)

        postfix = {"loss": 0.0, "acc": 0.0, "gen": 0.0}

        self.scheduler = build_lr_scheduler(
            self.optimizer, num_epochs * len(dataloader), self.lr_decay, self.lr_warmup
        )

        for step in (pbar := tqdm(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = 0.0
            epoch_div = 0.0
            epoch_acc = 0.0
            epoch_sparse = 0.0
            epoch_gen = 0.0
            attn_running = 0.0
            mask_running = 0.0

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, ladj, prior, attns, div = self(x)  # list of (b, l, l)
                gen_loss = (ladj - prior).mean()
                out = out.view(self.forward_evals, -1, self.out_dim)  # (e, b, c) logits
                y_evals = y.unsqueeze(0).expand(self.forward_evals, -1, -1)  # (e, b, 1)
                pointwise_losses = self.criterion.loss(out, y_evals, reduction="none")  # (e, b)
                loss_per_model = pointwise_losses.mean(dim=1)
                rec_loss = loss_per_model.mean()
                epoch_gen += gen_loss.item()

                loss = rec_loss + self.beta * gen_loss + self.div_coeff * div
                if self.include_sparsity:
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.item()
                    loss = loss + sparse_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += rec_loss.item()
                epoch_div += div.item()

                with torch.no_grad():
                    acc = self.criterion.accuracy(out, y_evals)
                    epoch_acc += acc.item()

                    attn_running += compute_mask_mean(attns)
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            epoch_sparse /= len(dataloader)
            epoch_gen /= len(dataloader)
            epoch_div /= len(dataloader)
            attn_running /= len(dataloader)
            mask_running /= len(dataloader)

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            sparses.append(epoch_sparse)
            gens.append(epoch_gen)
            attn_edges.append(attn_running)
            mask_edges.append(mask_running)

            postfix["loss"] = epoch_loss
            postfix["acc"] = epoch_acc
            postfix["gen"] = epoch_gen
            postfix["div"] = epoch_div

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
            gens,
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
            out, masks, attns = self(x, evaluate=True)
            loss = self.criterion.loss_from_probs(out, y)

            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.criterion.accuracy_from_probs(out, y)
                epoch_acc += acc.item()
                attn_running += compute_mask_mean(attns)
                mask_running += compute_mask_mean(masks)

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)

        self.logger.log_metrics({f"{folder}/loss_epoch_{name}": epoch_loss}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/acc_epoch_{name}": epoch_acc}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/attn_edges_{name}": attn_running}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/mask_edges_{name}": mask_running}, step=self.global_step)

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

        acc = self.criterion.accuracy_from_probs
        total_acc = acc(preds, trues)
        results["total_acc"] = total_acc.item()

        acc_a = acc(preds[:midpoint], trues[:midpoint])
        acc_b = acc(preds[midpoint:], trues[midpoint:])
        # confidence = mean probability assigned to the positive class
        conf_a = self.criterion.confidence(preds[:midpoint])
        conf_b = self.criterion.confidence(preds[midpoint:])

        results["acc_a"] = acc_a.item()
        results["acc_b"] = acc_b.item()
        results["conf_a"] = conf_a.item()
        results["conf_b"] = conf_b.item()

        self.train()
        return results

    @torch.inference_mode()
    def optimal_test(self, name: str, dataloader: DataLoader, folder: str = 'test'):
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
            loss = min(self.criterion.loss_from_probs(out, y) for out in outs)
            acc = max(self.criterion.accuracy_from_probs(out, y) for out in outs)

            attn_running += compute_mask_mean(attns)
            mask_running += compute_mask_mean(masks)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)

        self.logger.log_metrics({f"{folder}/loss_ens_{name}": epoch_loss}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/acc_ens_{name}": epoch_acc}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/attn_ens_{name}": attn_running}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/mask_ens_{name}": mask_running}, step=self.global_step)

        self.train()

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "attn": attn_running,
            "mask": mask_running,
        }
