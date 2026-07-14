import numpy as np
import torch
import torch.nn as nn
import zuko

from torch import Tensor
from torch.distributions import Distribution, Independent, Normal

from sparse_generalization.layers.agg_attention import AggregationAttention


class Encoder(zuko.lazy.LazyDistribution):
    def __init__(self, features: int, context: int, hidden_dim: int = 128) -> None:
        super().__init__()

        self.hyper = nn.Sequential(
            nn.Linear(context, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * features),
        )

    def forward(self, c: Tensor) -> Distribution:
        phi = self.hyper(c)
        mu, log_sigma = phi.chunk(2, dim=-1)

        return Independent(Normal(mu, log_sigma.exp()), 1)


class FlowVAE(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_dist: Distribution,
        num_heads: int = 1,
        encoder_heads: bool = False,
        force_vae_gaussian: bool = False, 
        use_encoder: bool = True,
        device: str = "cuda",
        layernorm: bool = True,
        flow_params: dict = {"n_flows": 3, "hidden_features": [256, 256]},
        separate_mask: bool = False,
        use_mask: bool = False,
        act: nn.Module = nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_heads = num_heads
        self.use_encoder = use_encoder
        self.encoder_heads = encoder_heads
        self.base_dist = base_dist
        self.force_vae_gaussian = force_vae_gaussian

        self.encoder_agg = AggregationAttention(
            embed_size=input_dim,
            num_heads=num_heads,
            out_dim=input_dim,
            use_mlp=False,
            device=device,
            layernorm=layernorm,
            separate_mask=separate_mask,
            use_mask=use_mask,
            act=act,
        )

        self.normalizing_flow = zuko.flows.NSF(
            features=output_dim,
            transforms=flow_params["n_flows"],
            hidden_features=flow_params["hidden_features"],
        )

        encoder_latent_dim = output_dim * num_heads if encoder_heads else output_dim
        if self.use_encoder:
            self.encoder = Encoder(encoder_latent_dim, input_dim)

        self.is_lazy = (
            True if isinstance(base_dist, zuko.lazy.LazyDistribution) else False
        )

    def forward(self, x: Tensor = None):
        ladj = 0
        batch_size, seq_len, _ = x.shape

        if self.use_encoder:
            encoding, _, _, _ = self.encoder_agg(x)
            encoding = encoding.squeeze(dim=1)
            q = self.encoder(encoding)
            rep = q.rsample()
            if self.encoder_heads:
                rep = rep.view(batch_size, self.num_heads, -1).reshape(
                    batch_size * self.num_heads, -1
                )
            
            if self.force_vae_gaussian:
                gaussian = torch.distributions.Normal(torch.zeros_like(rep[0]), torch.ones_like(rep[0]))
                vae_prior = gaussian.log_prob(rep.reshape(batch_size, -1)).sum(dim=-1)
        else:
            if self.is_lazy:
                base_dist = self.base_dist()
                rep = base_dist.sample().view(1, -1)
            else:
                base_dist = self.base_dist
                rep = self.prior.sample().view(1, -1)

        log_prior_base = (
            q.log_prob(rep.reshape(batch_size, -1))
            if self.use_encoder
            else base_dist.log_prob(rep.reshape(1, -1))
        ).reshape(-1, 1)
       
        dist = self.normalizing_flow()
        transform = dist.transform
        if self.training:
            output, ladj = transform.call_and_ladj(rep)
            ladj = zuko.distributions._sum_rightmost(ladj, dist.reinterpreted)
        else:
            output = transform(rep)

        log_prob_z = log_prior_base - ladj
        if self.force_vae_gaussian:
            log_prob_z = log_prob_z - vae_prior

        return output, log_prob_z
