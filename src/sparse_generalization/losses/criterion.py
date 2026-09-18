import torch
import torch.nn.functional as F

from torch import Tensor


class Criterion:

    def __init__(self, loss_type: str = "ce"):
        if loss_type not in ("ce", "bce"):
            raise ValueError(f"loss_type must be 'ce' or 'bce', got {loss_type!r}")
        self.loss_type = loss_type

    def _check(self, x: Tensor) -> None:
        head = x.size(-1)
        if self.loss_type == "bce" and head != 1:
            raise ValueError(f"loss_type='bce' needs out_dim=1 (a single sigmoid logit), got out_dim={head}")
        if self.loss_type == "ce" and head < 2:
            raise ValueError(f"loss_type='ce' needs out_dim>=2 (softmax classes), got out_dim={head}")

    def _ce_targets(self, y: Tensor, lead_shape: torch.Size) -> Tensor:
        return y.reshape(lead_shape).long()

    def _bce_targets(self, y: Tensor, shape: torch.Size) -> Tensor:
        return y.reshape(shape).float()

    def loss(self, logits: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
        self._check(logits)
        lead_shape = logits.shape[:-1]
        if self.loss_type == "ce":
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                self._ce_targets(y, lead_shape).reshape(-1),
                reduction=reduction,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, self._bce_targets(y, logits.shape), reduction=reduction
            )
        return loss.view(lead_shape) if reduction == "none" else loss

    def loss_from_probs(self, probs: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
        self._check(probs)
        if self.loss_type == "ce":
            return F.nll_loss(
                probs.reshape(-1, probs.size(-1)).clamp_min(eps).log(),
                self._ce_targets(y, probs.shape[:-1]).reshape(-1),
            )
        return F.binary_cross_entropy(probs, self._bce_targets(y, probs.shape))

    def probs(self, logits: Tensor) -> Tensor:
        self._check(logits)
        return F.softmax(logits, dim=-1) if self.loss_type == "ce" else torch.sigmoid(logits)

    def predict(self, preds: Tensor, from_probs: bool = False) -> Tensor:
        self._check(preds)
        if self.loss_type == "ce":
            return preds.argmax(dim=-1)
        threshold = 0.5 if from_probs else 0.0
        return (preds.squeeze(-1) > threshold).long()

    def accuracy(self, logits: Tensor, y: Tensor) -> Tensor:
        return self._accuracy(self.predict(logits), y)

    def accuracy_from_probs(self, probs: Tensor, y: Tensor) -> Tensor:
        return self._accuracy(self.predict(probs, from_probs=True), y)

    def _accuracy(self, hard_preds: Tensor, y: Tensor) -> Tensor:
        return (hard_preds == self._ce_targets(y, hard_preds.shape)).float().mean()

    def positive_prob(self, probs: Tensor) -> Tensor:
        return probs[..., 1] if self.loss_type == "ce" else probs[..., 0]

    def confidence(self, probs: Tensor) -> Tensor:
        return self.positive_prob(probs).mean()
