import numpy as np
import torch
import math

from torch import Tensor


def get_device(device: str | torch.device | None = None) -> str:
    if device is not None:
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width, device=None):
    device = get_device(device)
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width, device=device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2, device=device) * -(math.log(10000.0) / d_model)
    )
    pos_w = torch.arange(0.0, width, device=device).unsqueeze(1)
    pos_h = torch.arange(0.0, height, device=device).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe.detach()


def noise_scheduler(start_eta: float, step: int, gamma: float = 0.55):
    return start_eta / (1 + step) ** gamma

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    lr_decay: str = "none",
    warmup: bool = False,
    warmup_ratio: float = 0.05,
):
    if lr_decay not in ("none", "linear"):
        raise ValueError(f"lr_decay must be 'none' or 'linear', got {lr_decay!r}")

    if lr_decay == "none" and not warmup:
        return None

    warmup_steps = max(1, int(warmup_ratio * total_steps)) if warmup else 0

    def lr_lambda(step: int):
        min_lr_ratio = 0.1  
        if warmup and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if lr_decay == "linear":
            return 1.0 - progress * (1.0 - min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_gradients(parameters, max_norm: float | None = None):
    """Clip the global grad norm to max_norm; no-op if max_norm is None."""
    if max_norm is None:
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def enforce_sparsity(
    paths: Tensor,
    max_paths: float,
    alpha: float,
    alpha_min: float | None = None,
) -> Tensor:
    """Squared penalty on path density outside [alpha_min, alpha].

    relu(d - alpha)^2 + relu(lo - d)^2 with lo = alpha when alpha_min is None
    (equivalent to the old (alpha - d)^2); alpha_min=0 gives an upper cap only.
    """
    if alpha_min is not None and alpha_min > alpha:
        raise ValueError(f"alpha_min ({alpha_min}) must be <= alpha ({alpha})")
    density = paths.sum(dim=(-1, -2)) / max_paths
    lo = alpha if alpha_min is None else alpha_min
    return (torch.relu(density - alpha).pow(2) + torch.relu(lo - density).pow(2)).mean()


def reparametrize(mu: Tensor, sig: Tensor):
    std = torch.exp(0.5 * sig)
    eps = torch.randn_like(std)
    return mu + eps * std


def vae_log_prob(x: Tensor, mu: Tensor, sig: Tensor):
    return (
        -0.5 * math.log(2 * math.pi)
        - 0.5 * sig
        - ((x - mu) ** 2) / (2 * torch.exp(sig))
    ).sum(-1)


def with_residual_edges(edges: Tensor, residual: bool) -> Tensor:
    if residual and edges.size(-2) == edges.size(-1):
        return edges + torch.eye(edges.size(-1), device=edges.device, dtype=edges.dtype)
    return edges


def compute_attn_mean(all_attn: Tensor, threshold: float = 0.01, device: str | None = None, residual: bool = False):
    device = get_device(device)
    thresh_list = [with_residual_edges((attn > threshold).float(), residual) for attn in all_attn]  # list of (b, l, l)
    batch_size, seq_len, _ = thresh_list[0].size()
    path = torch.eye(seq_len, device=device).repeat(batch_size, 1, 1)
    for attn in thresh_list:
        path = torch.bmm(attn, path)

    return path.sum(dim=(1, 2)).mean()


@torch.no_grad()
def compute_attn_mean_ens(
    all_attn: list[list[Tensor]],
    threshold: float = 0.01,
    device: str | None = None,
    residual: bool = False,
    agg_pool: bool = False,
):
    """all_attn: per ensemble member, a list of (b, l, l) layer attns followed by the (b, h, 1, l) agg attn if agg_pool."""
    device = get_device(device)
    model_means = []
    for model_layers in all_attn:
        batch_size, seq_len, _ = model_layers[0].size()
        path = torch.eye(seq_len, device=device).repeat(batch_size, 1, 1)

        for i, layer_attn in enumerate(model_layers):
            if layer_attn.dim() == 4:
                layer_attn = layer_attn.sum(dim=1)  # sum heads, as for the layer attns
            edges = (layer_attn > threshold).float().to(device)
            is_agg_layer = agg_pool and i == len(model_layers) - 1
            path = torch.bmm(edges if is_agg_layer else with_residual_edges(edges, residual), path)
        model_means.append(path.sum(dim=(1, 2)).mean())

    return sum(model_means) / len(model_means)


def compute_mask_mean(all_masks: Tensor):
    return all_masks.sum(dim=(-2, -1)).mean()


def compute_max_paths(
    seq_len: int, num_heads: int = 1, num_layers: int = 1, agg_pool: bool = True
):
    paths = torch.ones((seq_len, seq_len)) * num_heads
    for l in range(num_layers - 1):
        multiplier = torch.ones((seq_len, seq_len)) * num_heads
        paths = paths @ multiplier

    if agg_pool:
        multiplier = torch.ones((1, seq_len)) * num_heads
        paths = multiplier @ paths

    return paths.sum().item()


def mask_score(masks, paths):
    paths_bool = (paths.squeeze() > 1).int()
    masks = masks.view(-1, paths.size(-1))

    mask1 = paths_bool == 1
    mask2 = masks == 1
    batch_result = (mask1 | ~mask2).all(dim=1).float()
    return batch_result.mean()
