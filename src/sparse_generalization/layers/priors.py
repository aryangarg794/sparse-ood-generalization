import torch
import zuko

class LaplacePrior(zuko.lazy.LazyDistribution):

    def __init__(
            self, 
            loc : float = 0.0, 
            scale: float = 1.0, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.loc = loc
        self.scale = scale

    def forward(self, c = None):
        return torch.distributions.Laplace(loc=self.loc, scale=self.scale)
    