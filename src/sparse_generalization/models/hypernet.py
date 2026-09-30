import torch
import torch.nn as nn

from copy import deepcopy
from functools import partial
from torch import Tensor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from sparse_generalization.utils.parallel import progress_bar
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
    clip_gradients,
    SparsityAnnealer,
)
from sparse_generalization.losses.criterion import Criterion
from sparse_generalization.losses.eta_diversity import EtaDiversity
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
        model_dim: int = 32,
        num_heads: int = 1,
        dropout: float = 0.0,
        hyper_type: str = "qk",
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        uniform_bound: float = 1.0,
        residual: bool = False,
        device: str | None = None,
        num_modes: int = 1,
        mode_onehot: bool = False,  # condition the weight flow on a one-hot of each weight set's mode
        mode_reduction: str = "sum",
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
        grad_clip: float | None = None,
        beta: float = 1.0,
        logger: WandbLogger = None,
        div_loss: nn.Module = CosineDiv,
        num_embeddings: int = 25,
        use_optimal_test: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        sparse_anneal: bool = False,
        sparse_start_coef: float = 1.0,
        sparse_end_coef: float = 1.0,
        sparse_start_decay: float = 0.0,
        sparse_end_decay: float = 1.0,
        threshold: float = 0.01,
        use_proj: bool = True,
        bern_mask: bool = False,
        agg_head: str = "mlp",  # 'mlp' | 'eta' (the eta model's linear_out -> out_net head)
        hyper_bias_init: bool = False,  # Bias-HyperInit for the weight generator (VHyperNet)
        eta_div_coef: float = 0.0,  # weight on the eta diversity between each pair of weight sets
        eta_div_type: str = "neg_l1",  # 'neg_l1' (eta model) | 'abs' | 'product' | 'jsd'
        eta_div_target: float | None = None,  # target overlap; None = minimise outright
        eta_div_floor: str | None = "squared",  # 'squared' | 'hard' | 'abs' | None (no floor)
        *args,
        **kwargs
    ):
        self.hyper_params = locals()

        for key in ["self", "__class__", "args", "kwargs"]:
            del self.hyper_params[key]

        if lr_decay not in ("none", "linear"):
            raise ValueError(f"lr_decay must be 'none' or 'linear', got {lr_decay!r}")
        if mode_reduction not in ("sum", "mean"):
            raise ValueError(f"mode_reduction must be 'sum' or 'mean', got {mode_reduction!r}")
        if eta_div_coef and num_modes < 2:
            raise ValueError("eta_div_coef compares sampled weight sets, so it needs num_modes > 1")
        if eta_div_coef and not include_agg_layer:
            raise ValueError("eta_div_coef needs include_agg_layer=True: eta is the flow into the aggregation query")

        device = get_device(device)
        kwargs.pop("num_eval_samples", None)  # removed (evals = num_modes); still in older checkpoints' hparams

        super().__init__(*args, **kwargs)

        self.device = device
        self.lr_decay = lr_decay
        self.lr_warmup = lr_warmup
        self.grad_clip = grad_clip
        self.logger = logger
        self.model_dim = model_dim
        self.val_freq = val_freq
        self.num_heads = num_heads
        self.div_coeff = div_coeff
        self.use_optimal_test = use_optimal_test
        self.num_mha_layers = num_mha_layers
        self.include_agg_layer = include_agg_layer
        self.num_modes = num_modes
        self.mode_reduction = mode_reduction
        self.criterion = Criterion(loss_type)
        self.out_dim = out_dim
        self.avg_heads = div_coeff == 0

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)
        else:
            bottleneck = 128
            self.feature_map = nn.Sequential(
                nn.Linear(inp_dim, bottleneck),
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
            uniform_bound=uniform_bound,
            residual=residual,
            div_loss=div_loss,
            device=device,
            layernorm=layernorm,
            separate_mask=separate_mask,
            use_mask=use_mask,
            act=act,
            num_modes=num_modes,
            mode_onehot=mode_onehot,
            use_proj=use_proj,
            bern_mask=bern_mask,
            agg_head=agg_head,
            hyper_bias_init=hyper_bias_init,
        )

        # eta has total mass 2 ** num_mha_layers under the residual stream (+I per attention layer)
        self.eta_div_coef = eta_div_coef
        self.eta_div = EtaDiversity(eta_div_type, eta_div_target, eta_div_floor,
                                    mass=2 ** num_mha_layers if residual else 1)

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
        self.sparse_annealer = SparsityAnnealer(
            sparse_anneal,
            sparse_start_coef,
            sparse_end_coef,
            sparse_start_decay,
            sparse_end_decay,
        )
        self.beta = beta

    def rec_loss(self, out: Tensor, y_evals: Tensor):
        loss_per_model = self.criterion.loss(out, y_evals, reduction="none").mean(dim=1)
        return loss_per_model.sum() if self.mode_reduction == "sum" else loss_per_model.mean()

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def forward(self, x: Tensor, evaluate: bool = False, ret_mean: bool = True):
        batch_size, width, height, _ = x.size()

        if self.max_paths is None:
            self.max_paths = compute_max_paths(
                width * height, self.num_heads, self.num_mha_layers, self.include_agg_layer
            )

            # print(f"MAX PATHS: {self.max_paths}")

        self.threshold = 1 / (width * height)

        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x_features = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)
        else:
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
            return self.hyper_net.evaluate(x_attn, ret_mean=ret_mean)

        return self.hyper_net(x_attn, avg_heads=self.avg_heads, compute_div=self.div_coeff != 0.0)

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
        self.split_history = []  # (epoch, split) at each val step

        self.scheduler = build_lr_scheduler(
            self.optimizer, num_epochs * len(dataloader), self.lr_decay, self.lr_warmup
        )
        self.sparse_annealer.total_steps = num_epochs * len(dataloader)

        for step in (pbar := progress_bar(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = torch.zeros((), device=self.device)
            epoch_div = torch.zeros((), device=self.device)
            epoch_acc = torch.zeros((), device=self.device)
            epoch_sparse = torch.zeros((), device=self.device)
            sparse_coef = 1.0
            epoch_gen = torch.zeros((), device=self.device)
            epoch_eta = torch.zeros((), device=self.device)
            attn_running = torch.zeros((), device=self.device)
            mask_running = torch.zeros((), device=self.device)

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, ladj, prior, attns, div = self(x)  # list of (b, l, l)
                gen_loss = (ladj - prior).mean()
                out = out.view(self.num_modes, -1, self.out_dim)  # (e, b, c) logits
                y_evals = y.unsqueeze(0).expand(self.num_modes, -1, -1)  # (e, b, 1)
                rec_loss = self.rec_loss(out, y_evals)
                epoch_gen += gen_loss.detach()

                loss = rec_loss + self.beta * gen_loss + self.div_coeff * div
                if self.eta_div_coef:
                    eta_loss, eta_raw = self.eta_div(self.hyper_net.eta, self.num_modes)
                    epoch_eta += eta_raw
                    loss = loss + self.eta_div_coef * eta_loss
                if self.include_sparsity:
                    sparse_coef = self.sparse_annealer.coef(self.global_step)
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.detach()
                    loss = loss + sparse_coef * sparse_loss

                self.optimizer.zero_grad()
                loss.backward()
                clip_gradients(self.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += rec_loss.detach()
                epoch_div += div.detach()

                with torch.no_grad():
                    acc = self.criterion.accuracy(out, y_evals)
                    epoch_acc += acc

                    attn_running += compute_mask_mean(attns)
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss = (epoch_loss / len(dataloader)).item()
            epoch_acc = (epoch_acc / len(dataloader)).item()
            epoch_sparse = (epoch_sparse / len(dataloader)).item()
            epoch_gen = (epoch_gen / len(dataloader)).item()
            epoch_div = (epoch_div / len(dataloader)).item()
            epoch_eta = (epoch_eta / len(dataloader)).item()
            attn_running = (attn_running / len(dataloader)).item()
            mask_running = (mask_running / len(dataloader)).item()

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

            if self.eta_div_coef:
                self.logger.log_metrics({"train/eta_overlap": epoch_eta}, step=step)
                postfix["eta"] = epoch_eta

            if self.include_sparsity:
                self.logger.log_metrics({"train/sparse_loss": epoch_sparse}, step=step)
                self.logger.log_metrics({"train/sparse_coef": sparse_coef}, step=step)
                postfix["sparse_loss"] = epoch_sparse
                postfix["sparse_coef"] = sparse_coef

            self.logger.log_metrics(
                {f"train/attn_edges_train": attn_running}, step=self.global_step
            )

            self.logger.log_metrics(
                {f"train/mask_edges_train": mask_running}, step=self.global_step
            )

            mode_accs = {}
            if not self.use_optimal_test and step % self.val_freq == 0:
                for loader, name in zip(testloaders, self.val_to_name.values()):
                    test_metrics = self.test(name, loader, folder="val")
                    mode_accs[name] = test_metrics["mode_accs"]
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
                    mode_accs[name] = test_metrics["mode_accs"]
                    if "id" in name:
                        postfix["ens_id"] = test_metrics["acc"]
                    elif "a" in name:
                        postfix["ens_a"] = test_metrics["acc"]
                    elif "b" in name:
                        postfix["ens_b"] = test_metrics["acc"]

            if "a" in mode_accs and "b" in mode_accs:
                postfix["A"] = "/".join(f"{acc:.2f}" for acc in mode_accs["a"])
                postfix["B"] = "/".join(f"{acc:.2f}" for acc in mode_accs["b"])
                split = self.mode_split(mode_accs["a"], mode_accs["b"])
                self.split_history.append((step, split))
                self.logger.log_metrics({"val/split": split}, step=self.global_step)
                postfix["split"] = split

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

    def _mode_accs(self, outs: Tensor, y: Tensor):
        # accuracy of each sampled weight set: outs (e, b, c) class probabilities -> (e,)
        return torch.stack([self.criterion.accuracy_from_probs(out, y) for out in outs])

    @staticmethod
    def mode_split(acc_a, acc_b):
        # best mean accuracy when two different weight sets take set a and set b (1.0 = one set
        # solves a and another solves b); nan with a single weight set
        acc_a, acc_b = torch.as_tensor(acc_a, dtype=torch.float), torch.as_tensor(acc_b, dtype=torch.float)
        if acc_a.numel() < 2:
            return float("nan")
        pair = (acc_a[:, None] + acc_b[None, :]) / 2
        pair.fill_diagonal_(-float("inf"))
        return pair.max().item()

    def _log_mode_accs(self, folder: str, name: str, mode_acc: Tensor):
        for k, acc in enumerate(mode_acc.tolist()):
            self.logger.log_metrics({f"{folder}/acc_{name}_mode{k}": acc}, step=self.global_step)

    def test(self, name: str, dataloader: DataLoader, folder: str = "test"):
        self.eval()
        mode_acc = torch.zeros(self.num_modes, device=self.device)
        attn_running = torch.zeros((), device=self.device)
        mask_running = torch.zeros((), device=self.device)
        epoch_acc = torch.zeros((), device=self.device)
        epoch_loss = torch.zeros((), device=self.device)

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            outs, masks, attns = self(x, evaluate=True, ret_mean=False)
            out = outs.mean(dim=0)  # ensemble over the weight sets
            loss = self.criterion.loss_from_probs(out, y)

            epoch_loss += loss.detach()
            with torch.no_grad():
                acc = self.criterion.accuracy_from_probs(out, y)
                epoch_acc += acc
                mode_acc = mode_acc + self._mode_accs(outs, y)
                attn_running += compute_mask_mean(attns)
                mask_running += compute_mask_mean(masks)

        epoch_loss = (epoch_loss / len(dataloader)).item()
        epoch_acc = (epoch_acc / len(dataloader)).item()
        attn_running = (attn_running / len(dataloader)).item()
        mask_running = (mask_running / len(dataloader)).item()

        self.logger.log_metrics({f"{folder}/loss_epoch_{name}": epoch_loss}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/acc_epoch_{name}": epoch_acc}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/attn_edges_{name}": attn_running}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/mask_edges_{name}": mask_running}, step=self.global_step)
        mode_acc = mode_acc / len(dataloader)
        self._log_mode_accs(folder, name, mode_acc)

        self.train()

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "attn": attn_running,
            "mask": mask_running,
            "mode_accs": mode_acc.tolist(),
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
        mode_acc = torch.zeros(self.num_modes, device=self.device)
        attn_running = torch.zeros((), device=self.device)
        mask_running = torch.zeros((), device=self.device)
        epoch_acc = torch.zeros((), device=self.device)
        epoch_loss = torch.zeros((), device=self.device)

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            outs, masks, attns = self(x, evaluate=True, ret_mean=False)
            loss = torch.stack([self.criterion.loss_from_probs(out, y) for out in outs]).min()
            batch_mode_acc = self._mode_accs(outs, y)
            acc = batch_mode_acc.max()
            mode_acc += batch_mode_acc

            attn_running += compute_mask_mean(attns)
            mask_running += compute_mask_mean(masks)

            epoch_loss += loss
            epoch_acc += acc

        epoch_loss = (epoch_loss / len(dataloader)).item()
        epoch_acc = (epoch_acc / len(dataloader)).item()
        attn_running = (attn_running / len(dataloader)).item()
        mask_running = (mask_running / len(dataloader)).item()

        self.logger.log_metrics({f"{folder}/loss_ens_{name}": epoch_loss}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/acc_ens_{name}": epoch_acc}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/attn_ens_{name}": attn_running}, step=self.global_step)
        self.logger.log_metrics({f"{folder}/mask_ens_{name}": mask_running}, step=self.global_step)
        mode_acc = mode_acc / len(dataloader)
        self._log_mode_accs(folder, name, mode_acc)

        self.train()

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "attn": attn_running,
            "mask": mask_running,
            "mode_accs": mode_acc.tolist(),
        }
