import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Self

from sparse_generalization.models.cnn import BasicCNN
from sparse_generalization.utils.util_funcs import positionalencoding2d, get_device


class BasicMLP(nn.Module):
    """Basic MLP class

    Args:
        input_dim (int): Size of Input.
        out_dim (int): Size of output.
        hidden_dims (List): Model architecture.
        act (nn.Module): Activation function.
    """

    def __init__(
        self: Self,
        input_dim: int,
        out_dim: int,
        hidden_dims: List,
        act: nn.Module,
        dropout: float,
        *args,
        **kwargs,
    ):
        super(BasicMLP, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential()
        self.layers.extend([nn.Linear(input_dim, hidden_dims[0]), act()])

        for dim1, dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(dim1, dim2), act(), nn.Dropout(dropout)])
        self.layers.append(nn.Linear(hidden_dims[-1], out_dim))

    def forward(self: Self, x: Tensor):
        return self.layers(x)


class MLPBaseline(nn.Module):

    def __init__(
        self: Self,
        inp_dim: int = 1,  # per-cell feature size when not embedding (1 for raw ids, 25 for one-hot)
        out_dim: int = 1,
        seq_len: int = 25,
        hidden_dims: List = [64, 128, 64],
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        lr: float = 1e-3,
        wd: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        val_freq: int = 10,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        embedding_inp: bool = True,
        num_embeddings: int = 25,
        model_dim: int = 16,
        device: str | None = None,
        logger: WandbLogger = None,
        module: nn.Module = BasicMLP,
        input_method: str = "concat",  
        pe: bool = False,  
        inp_size: int = 5,  
        use_optimal_test: bool = False,  
        *args,
        **kwargs,
    ):
        self.hyper_params = locals()
        for key in ["self", "__class__", "args", "kwargs"]:
            del self.hyper_params[key]

        device = get_device(device)

        super().__init__(*args, **kwargs)

        self.device = device
        self.logger = logger
        self.val_freq = val_freq
        self.val_to_name = val_to_name
        self.embedding_inp = embedding_inp
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.use_optimal_test = use_optimal_test
        self.pe = pe
        self.input_method = input_method

        if input_method not in ("concat", "mean", "max"):
            raise ValueError(f"input_method must be 'concat', 'mean' or 'max', got {input_method!r}")

        module_cls = module.func if hasattr(module, "func") else module  # unwrap hydra partials
        self.is_cnn = isinstance(module_cls, type) and issubclass(module_cls, BasicCNN)

        if self.is_cnn:
            self.model = module(
                input_dim=inp_dim,
                out_dim=out_dim,
                embedding_inp=embedding_inp,
                hidden_dims=hidden_dims,
                act=act,
                dropout=dropout,
                model_dim=model_dim,
                inp_size=inp_size,
            )
        else:
            token_dim = model_dim if embedding_inp else inp_dim
            if embedding_inp:
                self.embed_layer = nn.Embedding(num_embeddings, model_dim)
            bottleneck = 128
            self.feature_map = nn.Sequential(
                nn.Linear(token_dim, bottleneck),
                act(),
                nn.Linear(bottleneck, token_dim),
            )
            self.token_dim = token_dim
            mlp_in = seq_len * token_dim if input_method == "concat" else token_dim
            self.model = module(
                input_dim=mlp_in,
                out_dim=out_dim,
                hidden_dims=hidden_dims,
                act=act,
                dropout=dropout,
            )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd
        )
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0

    def _tokens(self: Self, x: Tensor):
        batch_size, width, height, _ = x.size()
        if self.embedding_inp:
            assert x.size(-1) == 1, "channels is not 1 for shapes input"
            x = self.embed_layer(x.squeeze(-1).int())  # (b, w, h, d)
        x = x.float()
        if self.pe:
            x = x + positionalencoding2d(
                self.token_dim, height=height, width=width, device=x.device
            ).permute(2, 1, 0)  # (dim, h, w) -> (w, h, dim)
        x = self.feature_map(x)
        return x.reshape(batch_size, width * height, self.token_dim)

    def forward(self: Self, x: Tensor, evaluate: bool = False):
        if self.is_cnn:
            out = self.model(x)  # (b, o)
        else:
            tokens = self._tokens(x)  # (b, l, d)
            if self.input_method == "concat":
                inp = tokens.flatten(1)
            elif self.input_method == "mean":
                inp = tokens.mean(dim=1)
            else:
                inp = tokens.max(dim=1)[0]
            out = self.model(inp)  # (b, o)
        if evaluate:
            return torch.sigmoid(out)
        return out

    def fit(self: Self, dataloader: DataLoader, num_epochs: int, testloaders: List):
        losses = []
        accs = []
        sparses = []
        mask_edges = []
        attn_edges = []

        attn_test = {i: [] for i in self.val_to_name.values()}
        masks_test = deepcopy(attn_test)
        losses_test = deepcopy(attn_test)
        accs_test = deepcopy(attn_test)

        postfix = {"loss": 0.0, "acc": 0.0}

        for step in (pbar := tqdm(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out = self(x)
                loss = self.loss(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    epoch_acc += self.accuracy(out, y).item()

                self.global_step += 1

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            sparses.append(0.0)
            mask_edges.append(0.0)
            attn_edges.append(0.0)

            postfix["loss"] = epoch_loss
            postfix["acc"] = epoch_acc

            pbar.set_description(f"Epoch: {step}")
            self.logger.log_metrics({"train/loss_epoch": epoch_loss}, step=step)
            self.logger.log_metrics({"train/acc_epoch": epoch_acc}, step=step)

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
    def test(self: Self, name: str, dataloader: DataLoader, folder: str = "test"):
        self.eval()
        epoch_acc = 0.0
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out = self(x, evaluate=True)
            epoch_loss += F.binary_cross_entropy(out, y).item()
            epoch_acc += self.accuracy(out, y).item()

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)

        self.logger.log_metrics(
            {f"{folder}/loss_epoch_{name}": epoch_loss}, step=self.global_step
        )
        self.logger.log_metrics(
            {f"{folder}/acc_epoch_{name}": epoch_acc}, step=self.global_step
        )

        self.train()

        return {"loss": epoch_loss, "acc": epoch_acc, "attn": 0.0, "mask": 0.0}

    @torch.inference_mode()
    def test_anti(self: Self, anti_dataset: DataLoader):
        self.eval()
        # total acc, acc a, acc b, conf a, conf b
        results = {}
        labels = []
        true_labels = []
        for batch_idx, (x, y) in enumerate(anti_dataset):
            x = x.to(self.device)
            y = y.to(self.device)
            probs = self(x, evaluate=True)
            labels.append(probs)
            true_labels.append(y)

        preds = torch.cat(labels, dim=0)
        trues = torch.cat(true_labels, dim=0)
        size = preds.size(0)
        midpoint = size // 2

        results["total_acc"] = self.accuracy(preds, trues).item()
        results["acc_a"] = self.accuracy(preds[:midpoint], trues[:midpoint]).item()
        results["acc_b"] = self.accuracy(preds[midpoint:], trues[midpoint:]).item()
        results["conf_a"] = preds[:midpoint].mean().item()
        results["conf_b"] = preds[midpoint:].mean().item()

        self.train()
        return results
