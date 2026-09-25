import torch
import zuko

from zuko.lazy import UnconditionalDistribution
from zuko.distributions import DiagNormal
from zuko.mixtures import GMM
from torch.distributions import Independent, Normal


class LaplacePrior(zuko.lazy.LazyDistribution):

    def __init__(self, loc: float = 0.0, scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loc = loc
        self.scale = scale

    def forward(self, c=None):
        return torch.distributions.Laplace(loc=self.loc, scale=self.scale)
    
class NormalPrior(zuko.lazy.LazyDistribution):

    def __init__(self, loc: float = 0.0, scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loc = loc
        self.scale = scale

    def forward(self, c=None):
        return torch.distributions.Normal(loc=self.loc, scale=self.scale)


def make_unit_gaussian(latent_dim: int, k: int = None):
    return UnconditionalDistribution(
        DiagNormal, torch.zeros(latent_dim), torch.ones(latent_dim), buffer=True
    )

def make_gmm(latent_dim: int, k: int = 5):
    gmm = GMM(
        features=latent_dim,
        components=k,
        covariance_type='full'
    )

    for param in gmm.parameters():
        param.requires_grad = False

    return gmm

def make_flow_prior(latent_dim: int, k: int = None):
    flow = zuko.flows.MAF(
        features=latent_dim
    )
    return flow

class DisconnectedPrior(zuko.lazy.LazyDistribution):

    def __init__(
        self,
        latent_dim: int,
        num_modes: int = 2,
        separation: float = 8.0,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if num_modes > latent_dim:
            raise ValueError(
                f"need one orthogonal direction per mode, got num_modes={num_modes} > latent_dim={latent_dim}"
            )

        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.separation = separation

        loc = separation * scale * torch.eye(num_modes, latent_dim)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", torch.full((num_modes, latent_dim), float(scale)))

    def forward(self, c=None):
        # batch shape (num_modes,), event shape (latent_dim,)
        return Independent(Normal(self.loc, self.scale), 1)


def make_disconnected_prior(
    latent_dim: int, k: int = 2, separation: float = 8.0, scale: float = 1.0
):
    return DisconnectedPrior(
        latent_dim, num_modes=k, separation=separation, scale=scale
    )
