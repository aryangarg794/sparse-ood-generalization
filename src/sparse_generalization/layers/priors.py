import torch
import zuko

from zuko.lazy import UnconditionalDistribution
from zuko.distributions import DiagNormal


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


def make_unit_gaussian(latent_dim):
    return UnconditionalDistribution(
        DiagNormal, torch.zeros(latent_dim), torch.ones(latent_dim), buffer=True
    )
