import itertools
import math
import torch
import torch.nn as nn

from torch import Tensor
from torchmetrics.functional.regression import jensen_shannon_divergence


def overlap_neg_l1(m1: Tensor, m2: Tensor, mass: float) -> Tensor:
    return -(m1 - m2).abs().sum(-1)


def overlap_abs(m1: Tensor, m2: Tensor, mass: float) -> Tensor:
    return 1 - (m1 - m2).abs().sum(-1) / (2 * mass)


def overlap_product(m1: Tensor, m2: Tensor, mass: float) -> Tensor:
    return (m1 * m2).sum(-1)

def overlap_jsd(m1: Tensor, m2: Tensor, mass: float, eps: float = 1e-8) -> Tensor:
    p, q = m1 / mass + eps, m2 / mass + eps
    return 1 - jensen_shannon_divergence(p, q, reduction="none") / math.log(2)


class EtaDiversity(nn.Module):
    OVERLAPS = {"neg_l1": overlap_neg_l1, "abs": overlap_abs, "product": overlap_product, "jsd": overlap_jsd}
    FLOORS = ("squared", "hard", "abs", None)

    def __init__(self, kind: str = "neg_l1", target: float | None = None, floor: str | None = "squared", mass: float = 1.0):
        super().__init__()
        if kind not in self.OVERLAPS:
            raise ValueError(f"eta_div_type must be one of {list(self.OVERLAPS)}, got {kind!r}")
        if floor not in self.FLOORS:
            raise ValueError(f"eta_div_floor must be one of {self.FLOORS}, got {floor!r}")
        self.kind, self.target, self.floor, self.mass = kind, target, floor, mass

    def forward(self, eta: Tensor, num_modes: int):
        if num_modes < 2:
            raise ValueError("eta diversity compares sampled weight sets, so it needs num_modes > 1")
        eta = eta.reshape(num_modes, -1, eta.size(-1))
        overlap = torch.stack([self.OVERLAPS[self.kind](eta[i], eta[j], self.mass)
                               for i, j in itertools.combinations(range(num_modes), 2)])  # (pairs, b)
        raw = overlap.detach().mean()
        if self.target is None or self.floor is None:
            return overlap.mean(), raw
        if self.floor == "squared":
            return (overlap - self.target).pow(2).mean(), raw
        if self.floor == "abs":
            return (overlap - self.target).abs().mean(), raw
        return torch.clamp(overlap, min=self.target).mean(), raw
