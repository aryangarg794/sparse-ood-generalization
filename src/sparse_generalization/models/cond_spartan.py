import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch import Tensor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from sparse_generalization.utils.parallel import progress_bar
from typing import List

from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.models.blocks import MHABlockCond
from sparse_generalization.layers.film_agg_attn import FiLMAggAttention
from sparse_generalization.layers.film_attn import FiLMHead
from sparse_generalization.utils.util_funcs import (
    get_device,
    positionalencoding2d,
    compute_mask_mean,
    compute_max_paths,
    build_lr_scheduler,
    with_residual_edges,
    SparsityAnnealer,
)
from sparse_generalization.losses.criterion import Criterion
from sparse_generalization.layers.diversity_losses import (
    CosineDiv, CosineRepDiv, L2DistanceDiv, MaskOverlapDiv, 
    MutualInfDiv
)

class ConditionalSPARTAN(nn.Module):

    def __init__(
        self, 
        inp_dim: int = 3,
        out_dim: int = 2,  # head size: 2 for ce (softmax classes), 1 for bce (single sigmoid logit)
        loss_type: str = "ce",  # 'ce' | 'bce'
        include_sparsity: bool = False,
        alpha: float = 0.1,
        num_modes: int = 3, 
        num_mha_layers: int = 1,
        num_eval_samples: int = 5, 
        model_dim: int = 32, 
        num_heads: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
        device: str | None = None,
        layernorm: bool = True,
        act: nn.Module = nn.ReLU,
        val_freq: int = 10, 
        div_coeff: float = 0.0, 
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        pe_type: str = "sin",  # 'sin' | 'coord' | 'learned' | 'none'
        max_grid_size: int = 5, 
        embedding_inp: bool = True,
        lr: float = 1e-3,
        lr_decay: str = "none",  # 'none' | 'linear'
        lr_warmup: bool = False,
        logger: WandbLogger = None,
        div_loss: nn.Module = CosineDiv,
        num_embeddings: int = 25,
        use_optimal_test: bool = False, 
        beta1: float = 0.9,
        beta2: float = 0.999,
        threshold: float = 0.01,
        context_dim: int = 8,
        temp: float = 1.0,
        num_layers_film: int = 2,
        film_mlp: bool = False,
        film_values: bool = False,
        agg_residual: bool = False,
        agg_res_coeff: float = 1.0,
        output_type: str = "linear",  # 'agg' | 'film' | 'linear'
        head_pool: str = "mean",  # 'mean' | 'max' | 'concat', token pooling for the film/linear heads
        seq_len: int = None,  # number of tokens (grid cells); needed for head_pool='concat', defaults to inp_dim
        sparse_anneal: bool = False,
        sparse_start_coef: float = 1.0,
        sparse_end_coef: float = 1.0,
        sparse_start_decay: float = 0.0,
        sparse_end_decay: float = 1.0,
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
        if output_type not in ("agg", "film", "linear"):
            raise ValueError(f"output_type must be 'agg', 'film' or 'linear', got {output_type!r}")
        self.output_type = output_type
        if head_pool not in ("mean", "max", "concat"):
            raise ValueError(f"head_pool must be 'mean', 'max' or 'concat', got {head_pool!r}")
        self.head_pool = head_pool
        self.seq_len = inp_dim if seq_len is None else seq_len
        self.num_eval_samples = num_eval_samples
        self.criterion = Criterion(loss_type)
        self.out_dim = out_dim

        if div_coeff != 0:
            self.avg_heads = False
        else:
            self.avg_heads = True

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
        self.num_modes = num_modes

        self.context_dim = context_dim
        self.div_loss = div_loss()
        self.modes = nn.Embedding(num_modes, context_dim)

        self.layers = nn.ModuleList()

        for _ in range(num_mha_layers):
            self.layers.append(
                MHABlockCond(
                    embed_size=self.embed_size,
                    context_dim=context_dim,
                    act=act,
                    dropout=dropout,
                    layernorm=layernorm,
                    residual=residual,
                    num_heads=num_heads,
                    temp=temp,
                    num_layers_film=num_layers_film,
                    film_mlp=film_mlp,
                    film_values=film_values,
                    device=device,
                )
            )

        if output_type == "agg":
            self.out = FiLMAggAttention(
                embed_size=self.embed_size,
                context_dim=context_dim,
                out_dim=out_dim,
                num_modes=num_modes,  
                num_heads=num_heads,
                dropout=dropout,
                temp=temp,
                layernorm=layernorm,
                device=device,
                act=act,
                num_layers_film=num_layers_film,
                residual=residual,
                agg_residual=agg_residual,
                agg_res_coeff=agg_res_coeff,
            )
        elif output_type == "film":
            self.out = FiLMHead(
                embed_size=self.embed_size,
                context_dim=context_dim,
                out_dim=out_dim,
                act=act,
                dropout=dropout,
                num_layers_film=num_layers_film,
                layernorm=layernorm,
                pool=head_pool,
                seq_len=self.seq_len,
            )
        else:
            head_dim = self.embed_size * self.seq_len if head_pool == "concat" else self.embed_size
            self.out = nn.Linear(head_dim, out_dim)

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
        self.val_to_name = val_to_name
        self.sparse_annealer = SparsityAnnealer(
            sparse_anneal,
            sparse_start_coef,
            sparse_end_coef,
            sparse_start_decay,
            sparse_end_decay,
        )

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def get_context(self):
        inp = torch.arange(self.num_modes, device=self.device)
        return self.modes(inp)

    def forward(
        self,
        x: Tensor,
        evaluate: bool = False,
        ret_mean: bool = True
    ):
        batch_size, width, height, _ = x.size()
        seq_len = width * height
        num_evals = self.num_eval_samples if evaluate else self.num_modes
        compute_div = True if self.div_coeff != 0.0 else False

        if self.max_paths is None:
            self.max_paths = compute_max_paths(
                width * height, self.num_heads, self.num_mha_layers, self.output_type == "agg"
            )

            print(f"MAX PATHS: {self.max_paths}")

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
        context = self.get_context()
        num_evals = context.size(0)
        div = torch.tensor([0.0], device=self.device)

        attn_maps = []
        eye = torch.eye(seq_len, device=self.device)
        path_matrix = eye.expand(batch_size, num_evals, seq_len, seq_len).clone()
        attn_matrix = eye.expand(batch_size, num_evals, seq_len, seq_len).clone()

        x_attn = x_attn.unsqueeze(1).expand(-1, num_evals, -1, -1)  # (b, e, l, d)

        for layer in self.layers:
            x_attn, mask, mask_attn, attn = layer(x_attn, context, avg_heads=self.avg_heads)
            attn_maps.append(attn)
            if not self.avg_heads:  # (b, h, e, l, l) -> (b, e, l, l)
                mask = mask.sum(dim=1)
                mask_attn = mask_attn.sum(dim=1)
            thresh = with_residual_edges((mask_attn > self.threshold).float(), self.residual)
            attn_matrix = torch.matmul(thresh, attn_matrix)
            path_matrix = torch.matmul(mask, path_matrix)

        if self.output_type == "agg":
            out, final_mask, mask_attn, agg_attn = self.out(x_attn, context, sum_heads=True)  # (b, e, o), (b, e, l)
            attn_maps.append(agg_attn)
            thresh = (mask_attn > self.threshold).float().unsqueeze(2)  # (b, e, 1, l)
            attn_matrix = torch.matmul(thresh, attn_matrix)
            path_matrix = torch.matmul(final_mask.unsqueeze(2), path_matrix)
        elif self.output_type == "film":
            out = self.out(x_attn, context)  # (b, e, o)
        else:
            if self.head_pool == "mean":
                pooled = x_attn.mean(dim=2)
            elif self.head_pool == "max":
                pooled = x_attn.max(dim=2)[0]
            else:
                pooled = x_attn.flatten(2)  # (b, e, l * d)
            out = self.out(pooled)  # (b, e, o)


        if not compute_div:
            div = div.detach()

        if num_evals > 1:  
            match self.div_loss:
                case CosineDiv():
                    # list of (b, h, e, l, l) -> (e, b, n, h, l, l)
                    attns_probs = torch.stack(attn_maps, dim=1).permute(3, 0, 1, 2, 4, 5)
                    div = self.div_loss(attns_probs)
                case CosineRepDiv():
                    div = self.div_loss(x_attn.transpose(0, 1))  # (e, b, l, d)
                case MaskOverlapDiv():
                    div = self.div_loss(path_matrix.transpose(0, 1))  # (e, b, l, l)
                case MutualInfDiv():
                    # (b, e, c) class probabilities; a single sigmoid column is expanded to [1-p, p] inside
                    div = self.div_loss(
                        self.criterion.probs(out).view(batch_size, num_evals, -1),
                        binary=self.criterion.loss_type == "bce",
                    )

        out = out.transpose(0, 1).reshape(-1, out.size(-1))
        path_matrix = path_matrix.transpose(0, 1).reshape(-1, path_matrix.size(-2), seq_len)
        attn_matrix = attn_matrix.transpose(0, 1).reshape(-1, attn_matrix.size(-2), seq_len)

        if evaluate:
            out = self.criterion.probs(out).view(num_evals, batch_size, -1)  # (e, b, c) class probabilities
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
            attn_running = torch.zeros((), device=self.device)
            mask_running = torch.zeros((), device=self.device)

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, attns, div = self(x)  # list of (b, l, l)
                out = out.view(self.num_modes, -1, self.out_dim)  # (e, b, c) logits
                y_modes = y.unsqueeze(0).expand(self.num_modes, -1, -1)  # (e, b, 1)
                pointwise_losses = self.criterion.loss(out, y_modes, reduction="none")  # (e, b)
                loss_per_model = pointwise_losses.mean(dim=1)
                rec_loss = loss_per_model.mean()

                if self.include_sparsity:
                    sparse_coef = self.sparse_annealer.coef(self.global_step)
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.detach()
                    loss = rec_loss + sparse_coef * sparse_loss + self.div_coeff * div
                else:
                    loss = rec_loss + self.div_coeff * div

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += rec_loss.detach()
                epoch_div += div.detach()

                with torch.no_grad():
                    acc = self.criterion.accuracy(out, y_modes)
                    epoch_acc += acc

                    attn_running += compute_mask_mean(attns)
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss = (epoch_loss / len(dataloader)).item()
            epoch_acc = (epoch_acc / len(dataloader)).item()
            epoch_sparse = (epoch_sparse / len(dataloader)).item()
            epoch_div = (epoch_div / len(dataloader)).item()
            attn_running = (attn_running / len(dataloader)).item()
            mask_running = (mask_running / len(dataloader)).item()

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
        attn_running = torch.zeros((), device=self.device)
        mask_running = torch.zeros((), device=self.device)
        epoch_acc = torch.zeros((), device=self.device)
        epoch_loss = torch.zeros((), device=self.device)

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out, masks, attns = self(x, evaluate=True)
            loss = self.criterion.loss_from_probs(out, y)

            epoch_loss += loss.detach()
            with torch.no_grad():
                acc = self.criterion.accuracy_from_probs(out, y)
                epoch_acc += acc
                attn_running += compute_mask_mean(attns)
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
            acc = torch.stack([self.criterion.accuracy_from_probs(out, y) for out in outs]).max()

            attn_running += compute_mask_mean(attns)
            mask_running += compute_mask_mean(masks)

            epoch_loss += loss
            epoch_acc += acc

        epoch_loss = (epoch_loss / len(dataloader)).item()
        epoch_acc = (epoch_acc / len(dataloader)).item()
        attn_running = (attn_running / len(dataloader)).item()
        mask_running = (mask_running / len(dataloader)).item()

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