import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader
from sparse_generalization.utils.parallel import progress_bar
from typing import List

from sparse_generalization.models.blocks import MHABlockGen
from sparse_generalization.layers.gen_mha import (
    FlowMasking,
    FlowMHA,
    FlowDirectA,
    FlowOnlyQK,
)
from sparse_generalization.layers.gen_agg_attn import (
    AggregationFlowMHA,
    AggregationFlowMask,
    AggregationFlowDirectA,
    AggregationFlowOnlyQK,
)
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.utils.util_funcs import (
    get_device,
    positionalencoding2d,
    compute_attn_mean,
    compute_mask_mean,
    compute_max_paths,
    build_lr_scheduler,
    SparsityAnnealer,
)
from sparse_generalization.losses.criterion import Criterion
from sparse_generalization.layers.priors import LaplacePrior, make_unit_gaussian


class FlowSpartan(nn.Module):

    def __init__(
        self,
        inp_dim: int = 3,
        seq_len: int = 25,
        out_dim: int = 2,  # head size: 2 for ce (softmax classes), 1 for bce (single sigmoid logit)
        loss_type: str = "ce",  # 'ce' | 'bce'
        model_dim: int = 32,
        num_heads: int = 1,
        num_layers: int = 4,
        use_mask: bool = False,
        separate_mask: bool = False,
        agg_pool: bool = False,
        residual: bool = True,
        include_sparsity: bool = False,
        alpha: float = 0.1,
        token_pool: bool = False,
        mha_layer: nn.Module = FlowMasking,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        step_size: float = 1e-1,
        pe_type: str = "sin",  # 'sin' | 'coord' | 'learned' | 'none'
        max_grid_size: int = 5,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (128, 128)},
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        embedding_inp: bool = True,
        beta: float = 1.0,
        lr: float = 1e-3,
        lr_decay: str = "none",  # 'none' | 'linear'
        lr_warmup: bool = False,
        prior_func = make_unit_gaussian,
        dropout: float = 0.1,
        layernorm: bool = True,
        act: nn.Module = nn.ReLU,
        logger: WandbLogger = None,
        force_vae_gaussian: bool = False,
        num_embeddings: int = 25,
        device: str | None = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        sparse_anneal: bool = False,
        sparse_start_coef: float = 1.0,
        sparse_end_coef: float = 1.0,
        sparse_start_decay: float = 0.0,
        sparse_end_decay: float = 1.0,
        threshold: float = 0.01,
        *args,
        **kwargs,
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
        self.criterion = Criterion(loss_type)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.agg_pool = agg_pool
        self.per_mask_prior = per_mask_prior
        self.token_pool = token_pool

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)
        else:
            bottleneck = 128
            self.feature_map = nn.Sequential(
                nn.Linear(inp_dim, bottleneck),
                act(),
                nn.Linear(bottleneck, model_dim),
                # nn.Identity()
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

        self.layers = nn.ModuleList()

        if mha_layer.func == FlowMHA:
            base_dist_size = 4 * self.embed_size
            agg_dist_size = base_dist_size
        elif mha_layer.func == FlowMasking or mha_layer.func == FlowDirectA:
            base_dist_size = seq_len
            agg_dist_size = 1
        elif mha_layer.func == FlowOnlyQK:
            base_dist_size = 2 * self.embed_size
            agg_dist_size = base_dist_size

        for _ in range(num_layers):
            self.layers.append(
                MHABlockGen(
                    embed_size,
                    seq_len=seq_len,
                    base_dist=prior_func(base_dist_size),
                    mha_layer=mha_layer,
                    num_heads=num_heads,
                    dropout=dropout,
                    act=act,
                    force_vae_gaussian=force_vae_gaussian, 
                    per_mask_prior=per_mask_prior,
                    separate_mask=separate_mask,
                    use_mask=use_mask,
                    prior_params=prior_params,
                    flow_params=flow_params,
                    prior_type=prior_type,
                    residual=residual,
                    layernorm=layernorm,
                    device=device,
                )
            )

        self.prior_type = prior_type
        non_mask_flow = (
            mha_layer.func == FlowMHA
            or mha_layer.func == FlowOnlyQK
            or mha_layer.func == FlowDirectA
        )
        assert (
            (self.prior_type != "a_laplace" and non_mask_flow) or (not non_mask_flow)
        ), "non mask flow doesn't support laplace"
        if self.prior_type == "laplace":
            self.prior = LaplacePrior()

        if self.agg_pool:
            if mha_layer.func == FlowMHA:
                print("Using FlowMHA")
                agg_layer = AggregationFlowMHA
            elif mha_layer.func == FlowMasking:
                print("Using FlowMasking")
                agg_layer = AggregationFlowMask
            elif mha_layer.func == FlowDirectA:
                print("Using FlowDirectA")
                agg_layer = AggregationFlowDirectA
            elif mha_layer.func == FlowOnlyQK:
                print("Using FlowOnlyQK")
                agg_layer = AggregationFlowOnlyQK

            self.out = agg_layer(
                out_dim=out_dim,
                act=act,
                base_dist=prior_func(agg_dist_size),
                dropout=dropout,
                embed_size=embed_size,
                seq_len=seq_len,
                separate_mask=separate_mask,
                use_mask=use_mask,
                per_mask_prior=per_mask_prior,
                num_heads=num_heads,
                prior_params=prior_params,
                flow_params=flow_params,
                prior_type=prior_type,
                residual=residual,
                layernorm=layernorm,
                device=device,
            )
        else:
            self.out = nn.Linear(self.embed_size, out_dim)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.scheduler = None
        self.loss = self.criterion.loss
        self.global_step = 0
        self.threshold = threshold
        self.residual = residual

        self.sparse_loss = L1SparsityAdjacency()
        self.alpha = alpha
        self.include_sparsity = include_sparsity
        self.max_paths = None
        self.step_size = step_size
        self.sparse_annealer = SparsityAnnealer(
            sparse_anneal,
            sparse_start_coef,
            sparse_end_coef,
            sparse_start_decay,
            sparse_end_decay,
        )
        self.val_to_name = val_to_name
        self.beta = beta

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def forward(self, x: Tensor):
        priors = 0
        ladjs = 0
        attn_matrices = []
        batch_size, width, height, _ = x.size()
        if self.max_paths is None:
            self.max_paths = compute_max_paths(
                width * height, self.num_heads, self.num_layers, self.agg_pool
            )

            print(f"MAX PATHS: {self.max_paths}")

        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x_features = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)
        else:
            x_features = self.feature_map(x)

        masks = torch.eye(width * height, device=self.device).repeat(batch_size, 1, 1)

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

        if self.token_pool:
            clses = self.cls.repeat(batch_size, 1, 1)
            x_attn = torch.cat([x_attn, clses], dim=1)

        for layer in self.layers:
            if self.training:
                x_attn, mask, attn, prior, ladj = layer(x_attn)
                if self.per_mask_prior:
                    priors += prior
                ladjs += ladj
            else:
                x_attn, mask, attn = layer(x_attn)
            attn_matrices.append(attn)

            if self.token_pool:
                mask = mask[:, :-1, :-1]

            masks = torch.bmm(mask, masks)

        if self.agg_pool:
            if self.training:
                out, final_mask, agg_attn, prior, ladj = self.out(x_attn)
                if self.per_mask_prior:
                    priors += prior
                ladjs += ladj
            else:
                out, final_mask, agg_attn = self.out(x_attn)
        elif self.token_pool:
            out = self.out(x_attn[:, -1, :])
        else:
            out = self.out(x_attn.max(dim=1)[0])

        if self.agg_pool:
            attn_matrices.append(agg_attn)
            masks = torch.bmm(final_mask, masks)

        if not self.per_mask_prior and self.training and self.prior_type == "laplace":
            priors = self.prior().log_prob(masks.sum(dim=(1, 2))) / self.max_paths
        if not self.per_mask_prior and self.training and self.prior_type == "a_laplace":
            priors = -self._enforce_sparsity(masks)
        elif not self.per_mask_prior and self.training and self.prior_type == "uniform":
            priors = torch.tensor([1.0], device=self.device).expand_as(ladjs)

        if self.training:
            gen_loss = (ladjs - priors).mean()
        else:
            gen_loss = None
        
        return out, masks, attn_matrices, gen_loss if self.training else None

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

        self.scheduler = build_lr_scheduler(
            self.optimizer, num_epochs * len(dataloader), self.lr_decay, self.lr_warmup
        )
        self.sparse_annealer.total_steps = num_epochs * len(dataloader)

        for step in (pbar := progress_bar(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = torch.zeros((), device=self.device)
            epoch_acc = torch.zeros((), device=self.device)
            epoch_sparse = torch.zeros((), device=self.device)
            sparse_coef = 1.0
            epoch_gen = torch.zeros((), device=self.device)
            attn_running = torch.zeros((), device=self.device)
            mask_running = torch.zeros((), device=self.device)
            epoch_masks = []
            epochs_trues = []

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, attns, gen_loss = self(x)  # list of (b, l, l)
                rec_loss = self.loss(out, y)
                epoch_gen += gen_loss.detach()

                if self.include_sparsity:
                    sparse_coef = self.sparse_annealer.coef(self.global_step)
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.detach()
                    loss = rec_loss + self.beta * gen_loss + sparse_coef * sparse_loss
                else:
                    loss = rec_loss + self.beta * gen_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += rec_loss.detach()
                with torch.no_grad():
                    acc = self.criterion.accuracy(out, y)
                    epoch_acc += acc

                    attn_running += compute_attn_mean(
                        attns, self.threshold, self.device, self.residual
                    )
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss = (epoch_loss / len(dataloader)).item()
            epoch_acc = (epoch_acc / len(dataloader)).item()
            epoch_sparse = (epoch_sparse / len(dataloader)).item()
            epoch_gen = (epoch_gen / len(dataloader)).item()
            attn_running = (attn_running / len(dataloader)).item()
            mask_running = (mask_running / len(dataloader)).item()

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            sparses.append(epoch_sparse)
            gens.append(epoch_gen)
            attn_edges.append(attn_running)
            mask_edges.append(mask_running)

            postfix = {"loss": epoch_loss, "acc": epoch_acc, "gen_loss": epoch_gen}

            pbar.set_description(f"Epoch: {step}")
            self.logger.log_metrics({"train/loss_epoch": epoch_loss}, step=step)
            self.logger.log_metrics({"train/acc_epoch": epoch_acc}, step=step)

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
            # postfix["attn_edges"] = attn_running

            # if self.agg_pool:
            #     self.out.temp_decay(step, num_epochs)
            #     postfix["temp"] = self.out.temp

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
        attn_running = torch.zeros((), device=self.device)
        mask_running = torch.zeros((), device=self.device)
        epoch_acc = torch.zeros((), device=self.device)
        epoch_loss = torch.zeros((), device=self.device)
        epoch_masks = []
        epochs_trues = []

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out, masks, attn, _ = self(x)
            loss = self.loss(out, y)

            epoch_masks.append(masks)

            epoch_loss += loss.detach()
            with torch.no_grad():
                acc = self.criterion.accuracy(out, y)
                epoch_acc += acc
                attn_running += compute_attn_mean(attn, self.threshold, self.device, self.residual)
                mask_running += compute_mask_mean(masks)

        epoch_loss = (epoch_loss / len(dataloader)).item()
        epoch_acc = (epoch_acc / len(dataloader)).item()
        attn_running = (attn_running / len(dataloader)).item()
        mask_running = (mask_running / len(dataloader)).item()

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
        self.eval()
        # total acc, acc a, acc b, conf a, conf b
        results = {}
        labels = []
        true_labels = []
        for batch_idx, (x, y) in enumerate(anti_dataset):
            x = x.to(self.device)
            y = y.to(self.device)
            out, mask, attn, _ = self(x)
            probs = self.criterion.probs(out)
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
