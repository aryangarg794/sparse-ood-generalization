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
    positionalencoding2d,
    compute_attn_mean,
    compute_mask_mean,
    compute_max_paths,
)
from sparse_generalization.layers.priors import LaplacePrior, make_unit_gaussian


class FlowSpartan(nn.Module):

    def __init__(
        self,
        inp_dim: int = 3,
        seq_len: int = 25,
        out_dim: int = 1,
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
        pe: bool = True,
        sinusoidal: bool = True,
        flow_params: dict = {"n_flows": 2, "hidden_features": (128, 128)},
        prior_params: dict = {"n_flows": 3, "hidden_features": (128, 128)},
        prior_type: str = "laplace",
        per_mask_prior: bool = False,
        embedding_inp: bool = True,
        beta: float = 1.0,
        lr: float = 1e-3,
        dropout: float = 0.1,
        layernorm: bool = True,
        act: nn.Module = nn.ReLU,
        logger: WandbLogger = None,
        force_vae_gaussian: bool = False,
        num_embeddings: int = 25,
        device: str = "cuda",
        beta1: float = 0.9,
        beta2: float = 0.999,
        threshold: float = 0.01,
        *args,
        **kwargs,
    ):
        self.hyper_params = locals()

        for key in ["self", "__class__", "args", "kwargs"]:
            del self.hyper_params[key]

        super().__init__(*args, **kwargs)
        self.device = device
        self.logger = logger
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.agg_pool = agg_pool
        self.per_mask_prior = per_mask_prior
        self.token_pool = token_pool

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)

        bottleneck = 128
        self.feature_map = nn.Sequential(
            nn.Linear(model_dim if embedding_inp else inp_dim, bottleneck),
            act(),
            nn.Linear(bottleneck, model_dim),
            # nn.Identity()
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
                    base_dist=make_unit_gaussian(base_dist_size),
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
                base_dist=make_unit_gaussian(agg_dist_size),
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
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.threshold = threshold

        self.sparse_loss = L1SparsityAdjacency()
        self.alpha = alpha
        self.include_sparsity = include_sparsity
        self.max_paths = None
        self.step_size = step_size
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
            x = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)

        x_features = self.feature_map(x)

        masks = torch.eye(width * height, device=self.device).repeat(batch_size, 1, 1)

        if self.pe:
            device = x.device
            if self.sinusoidal:
                embeddings = positionalencoding2d(
                    self.embed_size, height=height, width=width, device=self.device
                ).permute(  # returns (dim, h, w)
                    2, 1, 0
                )
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

        for step in (pbar := tqdm(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_sparse = 0.0
            epoch_gen = 0.0
            attn_running = 0.0
            mask_running = 0.0
            epoch_masks = []
            epochs_trues = []

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, attns, gen_loss = self(x)  # list of (b, l, l)
                rec_loss = self.loss(out, y)
                epoch_gen += gen_loss.item()

                if self.include_sparsity:
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.item()
                    loss = rec_loss + self.beta * gen_loss + sparse_loss
                else:
                    loss = rec_loss + self.beta * gen_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += rec_loss.item()
                with torch.no_grad():
                    acc = self.accuracy(out, y)
                    epoch_acc += acc.item()
                    
                    attn_running += compute_attn_mean(
                        attns, self.threshold, self.device
                    )
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            epoch_sparse /= len(dataloader)
            epoch_gen /= len(dataloader)
            attn_running /= len(dataloader)
            mask_running /= len(dataloader)

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
                postfix["sparse_loss"] = epoch_sparse

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
        attn_running = 0.0
        mask_running = 0.0
        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_masks = []
        epochs_trues = []

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out, masks, attn, _ = self(x)
            loss = self.loss(out, y)

            epoch_masks.append(masks)

            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                attn_running += compute_attn_mean(attn, self.threshold, self.device)
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

        self.train()
        return results
