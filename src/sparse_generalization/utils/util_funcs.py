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


def compute_attn_mean(all_attn: Tensor, threshold: float = 0.01, device: str | None = None):
    device = get_device(device)
    thresh_list = [(attn > threshold).float() for attn in all_attn]  # list of (b, l, l)
    batch_size, seq_len, _ = thresh_list[0].size()
    path = torch.eye(seq_len, device=device).repeat(batch_size, 1, 1)
    for attn in thresh_list:
        path = torch.bmm(attn, path)

    return path.sum(dim=(1, 2)).mean().item()


@torch.no_grad()
def compute_attn_mean_ens(all_attn: Tensor, threshold: float = 0.01, device: str | None = None):
    device = get_device(device)
    model_means = []
    for model_layers in all_attn:
        batch_size, seq_len, _ = model_layers[0].size()
        path = torch.eye(seq_len, device=device).repeat(batch_size, 1, 1)

        for layer_attn in model_layers:
            thresh = (layer_attn > threshold).float().to(device).squeeze(1)
            path = torch.bmm(thresh, path)
        model_means.append(path.sum(dim=(1, 2)).mean().item())

    return sum(model_means) / len(model_means)


def compute_mask_mean(all_masks: Tensor):
    return all_masks.sum(dim=(-2, -1)).mean().item()


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
