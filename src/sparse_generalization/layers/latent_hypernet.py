import math
import torch
import torch.nn as nn
import zuko

from torch import Tensor

from sparse_generalization.utils.util_funcs import get_device


class LatentHyperNet(nn.Module):
    """The unit latent model of notebooks/basic_multimodality.ipynb (latent section) as a weight generator for HyperNet.

    Every forward pass draws num_modes latent codes z ~ N(0, I) (z_dim numbers each; one per weight set, as
    train_latent draws one z per "condition") and maps each through a deterministic MLP to a full set of generated
    weights:  z -> Linear(z_dim, h0) -> act -> ... -> Linear(h_last, output_dim). With hidden_features [128, 256]
    and model_dim 32 this is the notebook's generator (4 * dim, 8 * dim).

    Evaluation (eval mode): eval_draws fixed scrambled-Sobol points mapped to N(0, I) (the notebook's mode report,
    unit_cover), so every validation / test call sees the same draws; None = num_modes fresh random draws as in
    training. HyperNetSpartan evaluates the draws in chunks (draw_slice) to bound memory.

    Latent flow (latent_flow=True; the notebook's LatentFlowWeightGenerator, batch-2 E17): z -> u = T(z) (a zuko MAF
    on z_dim dims) -> MLP(u). KL on the flow output, KL = log N(z) - log|det dT/dz| - log N(u) (T = identity: the unit
    model, KL 0). lf_init='identity' zeroes the last layer of every MAF transform, so T = id at init. Per training
    batch the KL is weighted by kl_sched: 'const' (weight 1; HyperNetSpartan's beta multiplies it) or 'bbb' (Bayes by
    backprop, Blundell et al. 2015 eq. 9: batch i of the M per epoch gets 2^(M-1-i) / (2^M - 1), divided by the batch
    size because the NLL is a batch mean; M = num_batches, set by HyperNetSpartan.fit).
    The second output is per weight set and scaled by the number of sets, so that HyperNetSpartan's
    beta * mean(gen) = beta * weight * sum over the sets of their KL, the notebook's kl_coef * kl.
    """

    variable_draws = True   # HyperNet takes the number of weight sets from this generator's output

    def __init__(
        self,
        output_dim: int,
        num_modes: int = 2,
        z_dim: int = 4,
        hidden_features: list = [128, 256],
        act: nn.Module = nn.ReLU,
        eval_draws: int | None = 128,
        latent_flow: bool = False,
        lf_transforms: int = 3,
        lf_hidden: int = 32,
        lf_init: str = "random",
        kl_sched: str = "const",
        device: str | None = None,
        **kwargs,
    ):
        device = get_device(device)
        super().__init__()
        if lf_init not in ("random", "identity"):
            raise ValueError(f"lf_init must be 'random' or 'identity', got {lf_init!r}")
        if kl_sched not in ("const", "bbb"):
            raise ValueError(f"kl_sched must be 'const' or 'bbb', got {kl_sched!r}")
        self.device = device
        self.num_modes = num_modes
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.eval_draws = eval_draws
        self.kl_sched = kl_sched

        dims = [z_dim] + list(hidden_features)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), act()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

        self.lflow = None
        if latent_flow:
            self.lflow = zuko.flows.MAF(features=z_dim, transforms=lf_transforms, hidden_features=(lf_hidden, lf_hidden))
            if lf_init == "identity":
                with torch.no_grad():
                    for m in self.lflow.modules():
                        if isinstance(m, zuko.flows.autoregressive.MaskedAutoregressiveTransform):
                            m.hyper[-1].weight.zero_()
                            m.hyper[-1].bias.zero_()

        if eval_draws:
            u = torch.quasirandom.SobolEngine(z_dim, scramble=True, seed=0).draw(eval_draws).clamp(1e-6, 1 - 1e-6)
            self.register_buffer("eval_z", torch.distributions.Normal(0.0, 1.0).icdf(u))
        self.draw_slice = None      # set by HyperNetSpartan to evaluate the eval draws in chunks
        self.num_batches = 1        # training batches per epoch (bbb), set by HyperNetSpartan.fit
        self.step = 0
        self.parts = {}

    @property
    def last_layer(self) -> nn.Linear:
        return self.net[-1]

    def T(self, z: Tensor) -> Tensor:
        return z if self.lflow is None else self.lflow().transform(z)

    def decode(self, z: Tensor) -> Tensor:
        """weights of the latent codes z (n, z_dim): (n, output_dim); z is the base noise (before the flow)."""
        return self.net(self.T(z))

    def forward(self, x: Tensor = None):
        if not self.training and self.eval_draws:
            z = self.eval_z if self.draw_slice is None else self.eval_z[self.draw_slice]
        else:
            z = torch.randn(self.num_modes, self.z_dim, device=self.device)
        gen = torch.zeros(len(z), device=z.device)
        if self.lflow is None:
            return self.net(z), gen

        dist = self.lflow()
        u, ladj = dist.transform.call_and_ladj(z)
        ladj = zuko.distributions._sum_rightmost(ladj, dist.reinterpreted)
        log_n = lambda v: (-0.5 * v ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
        kl = log_n(z) - ladj - log_n(u)
        if self.training:
            w = 1.0
            if self.kl_sched == "bbb":
                M = max(self.num_batches, 1)
                i, self.step = self.step % M, self.step + 1
                w = 2.0 ** (M - 1 - i) / (2.0 ** M - 1) / (x.size(0) if x is not None else 1)
            gen = w * kl * len(z)
            self.parts = dict(kl=kl.detach().mean().item(), dTz=(u - z).detach().norm(dim=-1).mean().item())
        return self.net(u), gen
