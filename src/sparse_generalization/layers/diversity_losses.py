import torch
import numpy as numpy
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class CosineDiv(nn.Module):

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def forward(self, attns: Tensor):
        # attns = (e, b, n, h, l, l)
        num_evals, batch_size, num_layers, _, _, _ = attns.shape
        assert attns.size(0) > 1 and len(attns.shape) == 6, "need multi-evals to compute dissimilarity, need shape to be exact"
        attns_vec_1 = attns.reshape(num_evals, batch_size, -1).unsqueeze(dim=1)
        attns_vec_2 = attns.reshape(num_evals, batch_size, -1).unsqueeze(dim=0)
        cos_mat = torch.nn.functional.cosine_similarity(attns_vec_1, attns_vec_2, dim=-1).mean(dim=2)
        mask = ~torch.eye(num_evals, dtype=torch.bool, device=attns.device)

        return cos_mat[mask].mean()

class CosineRepDiv(nn.Module):

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def forward(self, reps: Tensor):
        # attns = (e, b, l, d)
        num_evals, batch_size, _, _ = reps.shape
        assert reps.size(0) > 1 and len(reps.shape) == 4, "need multi-evals to compute dissimilarity, need shape to be exact"
        attns_vec_1 = reps.reshape(num_evals, batch_size, -1).unsqueeze(dim=1)
        attns_vec_2 = reps.reshape(num_evals, batch_size, -1).unsqueeze(dim=0)
        cos_mat = torch.nn.functional.cosine_similarity(attns_vec_1, attns_vec_2, dim=-1).mean(dim=-1)

        return cos_mat.mean()


class L2DistanceDiv(nn.Module):

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def forward(self, reps: Tensor):
        # attns = (e, b, l, d)
        num_evals, batch_size, _, _ = reps.shape
        assert reps.size(0) > 1 and len(reps.shape) == 4, "need multi-evals to compute dissimilarity, need shape to be exact"
        attns_vec_1 = reps.reshape(num_evals, batch_size, -1).unsqueeze(dim=1)
        attns_vec_2 = reps.reshape(num_evals, batch_size, -1).unsqueeze(dim=0)
        cos_mat = (attns_vec_1 - attns_vec_2).pow(2).mean(dim=-1).mean(dim=-1)

        return -cos_mat.mean()


class JensenShannonDiv(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, attns: Tensor):
        p = attns
        q = attns.roll(1, dims=0)

        m = 0.5 * (p + q)

        kl_pm = F.kl_div(
            (p + 1e-8).log(),
            m,
            reduction="none",
        ).sum(dim=-1)

        kl_qm = F.kl_div(
            (q + 1e-8).log(),
            m,
            reduction="none",
        ).sum(dim=-1)

        return 0.5 * (kl_pm + kl_qm).mean()